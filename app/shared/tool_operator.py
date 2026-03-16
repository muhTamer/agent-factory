from pathlib import Path
import json
import textwrap


def build_agent(agent_id: str, inputs: dict, gen_dir: Path) -> Path:
    gen_dir.mkdir(parents=True, exist_ok=True)

    # inputs["tool"] is the canonical tool name (e.g. "verify_identity")
    tool_name = inputs.get("tool") or ""

    cfg = {
        "id": agent_id,
        "tool": tool_name,
    }
    (gen_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    agent_src = textwrap.dedent(
        f"""
    # Auto-generated Tool Operator agent ({agent_id})
    import json
    from pathlib import Path
    from app.runtime.interfaces import IAgent

    class Agent(IAgent):
        def __init__(self):
            self.ready = True
            self.tool_name = ""
            self._stub_response = {{}}

        def load(self, spec):
            cfg_path = Path(__file__).parent / "config.json"
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.tool_name = cfg.get("tool", "")
            try:
                from app.runtime.tools.stub_tools import STUB_RESPONSES
                self._stub_response = STUB_RESPONSES.get(self.tool_name, {{}})
            except ImportError:
                self._stub_response = {{}}

        def handle(self, request):
            resp = dict(self._stub_response)
            resp["agent_id"] = "{agent_id}"
            resp["tool"] = self.tool_name
            if not resp:
                resp["message"] = f"[DEMO] Tool {{self.tool_name}} executed (no stub configured)."
            return resp

        def metadata(self):
            return {{
                "id": "{agent_id}",
                "type": "tool_operator",
                "ready": True,
                "tool": self.tool_name,
                "capabilities": ["tool_execution"]
            }}
    """
    )
    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8", newline="\n")
    return gen_dir
