# app/shared/workflow.py
from __future__ import annotations

import json
import textwrap
from pathlib import Path


def build_agent(agent_id: str, inputs: dict, gen_dir: Path) -> Path:
    """
    Generic workflow agent generator.

    Expected inputs:
      - workflow_spec: dict (LLM-generated, validated earlier)
      - docs: optional list[str]
      - policies: optional list[str]
      - tools: optional list[str]

    Output:
      - generated/<agent_id>/workflow_spec.json
      - generated/<agent_id>/config.json
      - generated/<agent_id>/agent.py (thin wrapper around GenericWorkflowEngine + LLM event mapper)
    """
    gen_dir.mkdir(parents=True, exist_ok=True)

    workflow_spec = inputs.get("workflow_spec")
    if not isinstance(workflow_spec, dict):
        raise ValueError("workflow.build_agent requires inputs['workflow_spec'] as dict")

    docs = inputs.get("docs") or []
    policies = inputs.get("policies") or []
    tools = inputs.get("tools") or []

    if isinstance(docs, str):
        docs = [docs]
    if isinstance(policies, str):
        policies = [policies]
    if isinstance(tools, str):
        tools = [tools]

    # Write workflow spec beside the agent for runtime loading
    wf_path = gen_dir / "workflow_spec.json"
    wf_path.write_text(json.dumps(workflow_spec, indent=2), encoding="utf-8")

    # Write a small config that can include doc/policy/tool context
    cfg = {
        "id": agent_id,
        "workflow_spec_path": str(wf_path),
        "docs": docs,
        "policies": policies,
        "tools": tools,
    }
    cfg_path = gen_dir / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Generate the wrapper agent.py
    # NOTE: use .format to avoid f-string brace collisions inside generated code
    agent_src = textwrap.dedent(
        """\
        # Auto-generated Workflow Runner agent ({agent_id})
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Dict, Any, List

        from app.runtime.interfaces import IAgent
        from app.runtime.workflow_engine import GenericWorkflowEngine
        from app.runtime.workflow_mapper import map_query_to_event_and_slots


        class Agent(IAgent):
            def __init__(self) -> None:
                self.ready = False
                self.engine: GenericWorkflowEngine | None = None
                self.context: Dict[str, Any] = {{}}

            def load(self, spec: Dict[str, Any]) -> None:
                cfg_path = Path(__file__).parent / "config.json"
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

                wf_path = Path(cfg["workflow_spec_path"])
                workflow_spec = json.loads(wf_path.read_text(encoding="utf-8"))

                self.context = {{
                    "docs": cfg.get("docs", []),
                    "policies": cfg.get("policies", []),
                    "tools": cfg.get("tools", []),
                }}

                self.engine = GenericWorkflowEngine(
                    agent_id="{agent_id}",
                    workflow_spec=workflow_spec,
                    context=self.context,
                )
                self.ready = True

            def _allowed_events(self) -> List[str]:
                if not self.engine:
                    return []
                state_name = self.engine.current_state
                state = self.engine.states.get(state_name, {{}})
                on = state.get("on") or {{}}
                if isinstance(on, dict):
                    return list(on.keys())
                return []

            def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
                if not self.engine:
                    return {{"error": "engine_not_loaded", "agent_id": "{agent_id}"}}

                # Incoming requests from runtime are typically: {{"query": "..."}}
                query = ""
                if isinstance(request, dict):
                    query = str(request.get("query", "") or "")
                else:
                    query = str(request or "")

                allowed_events = self._allowed_events()

                # Ask LLM to map query -> event + slot updates (generic, non-hardcoded)
                mr = map_query_to_event_and_slots(
                    query=query,
                    current_state=self.engine.current_state,
                    allowed_events=allowed_events,
                    slot_defs=self.engine.slot_defs,
                    model="gpt-5-mini",
                )

                # Drive the workflow engine
                wf_res = self.engine.handle({{
                    "event": mr.event,
                    "slots": mr.slots,
                    "query": query,
                }})

                # Attach explainability/debug info (useful for thesis + UI)
                wf_res["mapper"] = {{
                    "state": self.engine.current_state,
                    "allowed_events": allowed_events,
                    "event": mr.event,
                    "slots": mr.slots,
                    "confidence": mr.confidence,
                    "rationale": mr.rationale,
                }}
                return wf_res

            def metadata(self) -> Dict[str, Any]:
                if not self.engine:
                    return {{
                        "id": "{agent_id}",
                        "type": "workflow_runner",
                        "ready": False,
                        "description": "Workflow runner (engine not loaded)",
                        "capabilities": [],
                    }}
                return self.engine.metadata()
        """
    ).format(agent_id=agent_id)

    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")
    return gen_dir
