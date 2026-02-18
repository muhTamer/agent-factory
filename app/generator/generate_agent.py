# app/generator/generate_agent.py
"""
Agent Generator
---------------
Reads a factory spec entry (for one agent) and generates a fully functional
agent package in `generated/<agent_id>/`.

Each blueprint folder (under factory/blueprints/<blueprint_id>/)
may include:
    - blueprint.yaml (metadata)
    - templates/ (optional starter code)
    - prompt.md (LLM-based code synthesis, optional)

The generator:
  1. Loads the blueprint metadata.
  2. Resolves shared library entrypoints (e.g., app.shared.rag.build_agent).
  3. Creates a self-contained Python package: generated/<id>/agent.py
  4. Writes config.yaml and metadata.json.
  5. Runs a local smoke test before registration.
"""

from __future__ import annotations
import importlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any
import yaml


# ------------------------------------------------------------
# 🔧 main entrypoint
# ------------------------------------------------------------
def generate_agent(agent_spec: Dict[str, Any]) -> Path:
    """
    Generate or refresh a domain agent from blueprint and inputs.

    Args:
        agent_spec: one dict from factory_spec["agents"]

    Returns:
        Path to generated/<agent_id>/agent.py
    """
    agent_id = agent_spec["id"]
    blueprint = agent_spec.get("blueprint")
    # bp_meta = agent_spec.get("blueprint_meta", {})
    inputs = agent_spec.get("inputs", {})
    # base_dir = Path(agent_spec.get("inputs", {}).get("base_dir", Path.cwd()))

    print(f"[GEN] Generating agent '{agent_id}' from blueprint '{blueprint}'")

    # Destination folder
    gen_dir = Path("generated") / agent_id
    gen_dir.mkdir(parents=True, exist_ok=True)  # just ensure it exists
    # (No rmtree; we will overwrite files below)

    # Write metadata
    meta_path = gen_dir / "metadata.json"
    meta_path.write_text(json.dumps(agent_spec, indent=2), encoding="utf-8")

    # Resolve blueprint path
    bp_root = Path("factory/blueprints") / blueprint
    bp_yaml = bp_root / "blueprint.yaml"
    if not bp_yaml.exists():
        raise FileNotFoundError(f"Blueprint '{blueprint}' not found at {bp_yaml}")

    blueprint_meta = yaml.safe_load(bp_yaml.read_text(encoding="utf-8"))

    # Step 1: Try direct entrypoint (shared library builder)
    entrypoint = blueprint_meta.get("entrypoint")
    if entrypoint:
        try:
            module_name, func_name = entrypoint.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            build_func = getattr(mod, func_name)
            print(f"[GEN] Using entrypoint: {entrypoint}")
            return build_func(agent_id=agent_id, inputs=inputs, gen_dir=gen_dir)
        except Exception as e:
            print(f"[WARN] Entrypoint failed: {entrypoint} -> {e}")
            traceback.print_exc()

    # Step 2: Fallback — use template copy
    template_dir = bp_root / "templates"
    if template_dir.exists():
        for f in template_dir.glob("*"):
            dest = gen_dir / f.name
            shutil.copy2(f, dest)
        print(f"[GEN] Copied template files for {agent_id}")
    else:
        # Minimal stub if no entrypoint and no template
        (gen_dir / "agent.py").write_text(_default_agent_stub(agent_id), encoding="utf-8")

    print(f"[GEN] Agent '{agent_id}' generated at {gen_dir}")
    return gen_dir


# ------------------------------------------------------------
# 📄 default stub (when no blueprint template or entrypoint found)
# ------------------------------------------------------------
def _default_agent_stub(agent_id: str) -> str:
    return f'''"""
Auto-generated stub agent: {agent_id}
Implements IAgent (load, handle, metadata)
"""

from typing import Dict, Any
from app.runtime.interfaces import IAgent

class Agent(IAgent):
    def __init__(self):
        self.ready = True

    def load(self, spec: Dict[str, Any]) -> None:
        self.spec = spec

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        q = request.get("query", "")
        return {{"answer": f"[{agent_id}] Echo: " + q}}

    def metadata(self) -> Dict[str, Any]:
        return {{"id": "{agent_id}", "ready": self.ready}}
'''


# ------------------------------------------------------------
# 🧪 optional smoke test
# ------------------------------------------------------------
def smoke_test_agent(gen_dir: Path) -> bool:
    """Import the agent and run a basic handle() call."""
    try:
        from app.runtime.registry import AgentRegistry

        reg = AgentRegistry()
        agent = reg.import_generated_agent(gen_dir.name, gen_dir)
        agent.load({"id": gen_dir.name})
        res = agent.handle({"query": "ping"})
        print(f"[SMOKE] {gen_dir.name} -> {res}")
        return True
    except Exception as e:
        print(f"[ERROR] Smoke test failed for {gen_dir.name}: {e}")
        traceback.print_exc()
        return False
