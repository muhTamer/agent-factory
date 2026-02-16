# app/shared/workflow.py
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from string import Template


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
    # Generate the wrapper agent.py
    # IMPORTANT: do NOT use .format() here because the generated code contains many { } dict literals.
    # Use Template ($agent_id) to avoid brace collisions.
    agent_src = Template(
        textwrap.dedent(
            """\
        # Auto-generated Workflow Runner agent ($agent_id)
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

                # Thread-aware: one engine per conversation thread
                self.engines: Dict[str, GenericWorkflowEngine] = {}

                # Shared, immutable inputs loaded once
                self.workflow_spec: Dict[str, Any] = {}
                self.context: Dict[str, Any] = {}

            def load(self, spec: Dict[str, Any]) -> None:
                cfg_path = Path(__file__).parent / "config.json"
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

                wf_path = Path(cfg["workflow_spec_path"])
                self.workflow_spec = json.loads(wf_path.read_text(encoding="utf-8"))

                self.context = {
                    "docs": cfg.get("docs", []),
                    "policies": cfg.get("policies", []),
                    "tools": cfg.get("tools", []),
                }

                # Engines are created lazily per thread_id in handle().
                self.engines = {}
                self.ready = True

            def _get_thread_id(self, request: Dict[str, Any]) -> str:
                ctx = request.get("context") if isinstance(request, dict) else None
                if isinstance(ctx, dict) and ctx.get("thread_id"):
                    return str(ctx["thread_id"])
                if isinstance(request, dict) and request.get("thread_id"):
                    return str(request["thread_id"])
                return "default"

            def _engine_for(self, thread_id: str) -> GenericWorkflowEngine:
                eng = self.engines.get(thread_id)
                if eng is None:
                    eng = GenericWorkflowEngine(
                        agent_id="$agent_id",
                        workflow_spec=self.workflow_spec,
                        context=self.context,
                    )
                    self.engines[thread_id] = eng
                return eng

            def _allowed_events(self, engine: GenericWorkflowEngine) -> List[str]:
                state_name = engine.current_state
                state = engine.states.get(state_name, {})
                on = state.get("on") or {}
                if isinstance(on, dict):
                    return list(on.keys())
                return []

            def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
                if not self.ready:
                    return {"error": "engine_not_loaded", "agent_id": "{agent_id}"}

                # Incoming request: {"query": "...", "context": {"thread_id": "..."}}
                query = ""
                if isinstance(request, dict):
                    query = str(request.get("query", "") or "")
                else:
                    query = str(request or "")

                thread_id = self._get_thread_id(request if isinstance(request, dict) else {})
                engine = self._engine_for(thread_id)

                # ✅ Merge request context (runtime) into engine.context WITHOUT gating workflow execution
                req_ctx = request.get("context") if isinstance(request, dict) else None
                merged_ctx = dict(self.context or {})

                if isinstance(req_ctx, dict):
                    # Merge lists (docs/policies/tools) without duplicates
                    for k in ("docs", "policies", "tools"):
                        base = merged_ctx.get(k) or []
                        extra = req_ctx.get(k) or []
                        if not isinstance(base, list):
                            base = [base]
                        if not isinstance(extra, list):
                            extra = [extra]
                        merged_ctx[k] = list(dict.fromkeys([*base, *extra]))

                    # Carry over other keys (thread_id, intent, domain, etc.)
                    for k, v in req_ctx.items():
                        if k not in ("docs", "policies", "tools"):
                            merged_ctx[k] = v

                engine.context = merged_ctx

                allowed_events = self._allowed_events(engine)

                # Ask LLM to map query -> event + slot updates (generic, non-hardcoded)
                mr = map_query_to_event_and_slots(
                    query=query,
                    current_state=engine.current_state,
                    allowed_events=allowed_events,
                    slot_defs=engine.slot_defs,
                    model="gpt-5-mini",
                    current_slots=getattr(engine, "slots", {}) or {},
                )

                # Drive the workflow engine (state is now per-thread)
                wf_res = engine.handle({
                    "event": mr.event,
                    "slots": mr.slots,
                    "query": query,
                })

                # Attach explainability/debug info (useful for thesis + UI)
                wf_res["mapper"] = {
                    "thread_id": thread_id,
                    "state": engine.current_state,
                    "allowed_events": allowed_events,
                    "event": mr.event,
                    "slots": mr.slots,
                    "confidence": mr.confidence,
                    "rationale": mr.rationale,
                }

                # Echo merged context for visibility/debugging
                wf_res["context"] = engine.context

                return wf_res


            def metadata(self) -> Dict[str, Any]:
                return {
                    "id": "$agent_id",
                    "type": "workflow_runner",
                    "ready": self.ready,
                    "description": "Multi-turn workflow agent that can ask follow-up questions and proceed step-by-step.",
                    "capabilities": ["multi_turn", "followups", "workflow"],
                }
        """
        )
    ).substitute(agent_id=agent_id)

    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8")
    return gen_dir
