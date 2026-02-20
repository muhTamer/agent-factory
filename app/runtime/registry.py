# app/runtime/registry.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from app.runtime.interfaces import IAgent
from importlib.util import spec_from_file_location, module_from_spec


class AgentRegistry:
    """Keeps track of loaded agents, supports dynamic import of generated packages."""

    def __init__(self) -> None:
        self._agents: Dict[str, IAgent] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def register(self, agent_id: str, agent: IAgent, meta: Optional[Dict[str, Any]] = None) -> None:
        self._agents[agent_id] = agent
        self._meta[agent_id] = meta or {}

    def get(self, agent_id: str) -> Optional[IAgent]:
        return self._agents.get(agent_id)

    def all_ids(self) -> List[str]:
        return list(self._agents.keys())

    def all_meta(self) -> Dict[str, Dict[str, Any]]:
        return {aid: self._safe_metadata(aid) for aid in self._agents}

    def _safe_metadata(self, agent_id: str) -> Dict[str, Any]:
        try:
            base = self._agents[agent_id].metadata() or {}
        except Exception:
            base = {}
        # Merge registered meta (e.g. blueprint_meta from factory spec)
        # over generic agent defaults — rich descriptions take priority.
        stored = self._meta.get(agent_id, {})
        if stored:
            base.update(stored)
        base.update({"id": agent_id})
        return base

    # -------- dynamic import of generated agents --------
    def import_generated_agent(self, agent_id: str, gen_dir: Path):
        """
        Load generated/<agent_id>/agent.py dynamically and return an Agent instance.

        This is resilient to Windows cp1252 output by normalizing to UTF-8 on load.
        """
        from pathlib import Path

        gen_dir = Path(gen_dir)
        agent_py = gen_dir / "agent.py"
        if not agent_py.exists():
            raise FileNotFoundError(f"Generated agent missing: {agent_py}")

        # --- Normalize encoding: if file isn't UTF-8, rewrite as UTF-8 ---
        raw = agent_py.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", errors="replace")
            agent_py.write_text(text, encoding="utf-8", newline="\n")
            print(f"[WARN] Rewrote non-utf8 generated agent to utf-8: {agent_py}")

        # Import as a unique module name per agent_id
        mod_name = f"generated_{agent_id}"
        spec = spec_from_file_location(mod_name, str(agent_py))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for generated agent: {agent_py}")

        mod = module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)  # type: ignore

        if not hasattr(mod, "Agent"):
            raise AttributeError(f"Generated agent module has no Agent class: {agent_py}")

        return mod.Agent()
