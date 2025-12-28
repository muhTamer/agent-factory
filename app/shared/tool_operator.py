from pathlib import Path
import json
import textwrap


def build_agent(agent_id: str, inputs: dict, gen_dir: Path) -> Path:
    gen_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "id": agent_id,
        "tools": inputs.get("tools", []),
    }
    (gen_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    agent_src = textwrap.dedent(
        f"""
    # Auto-generated Tool Operator agent ({agent_id})
    from app.runtime.interfaces import IAgent

    class Agent(IAgent):
        def __init__(self):
            self.ready = True

        def load(self, spec):
            pass

        def handle(self, request):
            return {{
                "agent_id": "{agent_id}",
                "action": "noop",
                "message": "Tool operator stub – no tools executed yet."
            }}

        def metadata(self):
            return {{
                "id": "{agent_id}",
                "type": "tool_operator",
                "ready": True,
                "capabilities": ["tool_execution"]
            }}
    """
    )
    (gen_dir / "agent.py").write_text(agent_src, encoding="utf-8", newline="\n")
    return gen_dir
