# scripts/_bootstrap.py
"""
Shared bootstrap helper — loads agents from the factory spec into a registry.

Usage:
    from scripts._bootstrap import bootstrap_registry
    registry, store = bootstrap_registry()
"""

from __future__ import annotations

import json
from pathlib import Path

from app.orchestration.performance_store import PerformanceStore
from app.runtime.registry import AgentRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTORY_SPEC_PATH = REPO_ROOT / ".factory" / "factory_spec.json"


def bootstrap_registry(
    spec_path: Path = FACTORY_SPEC_PATH,
) -> tuple[AgentRegistry, PerformanceStore]:
    """Load agents from the factory spec into a fresh AgentRegistry.

    Returns (registry, performance_store).
    """
    registry = AgentRegistry()
    store = PerformanceStore()

    if not spec_path.exists():
        raise FileNotFoundError(f"Factory spec not found: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    for agent_spec in spec.get("agents", []):
        a_id = agent_spec["id"]
        a_type = agent_spec.get("type")

        if a_type == "guardrails":
            continue

        gen_dir = REPO_ROOT / "generated" / a_id
        if not (gen_dir / "agent.py").exists():
            print(f"[bootstrap] Skipping {a_id} — not generated yet")
            continue

        try:
            agent = registry.import_generated_agent(a_id, gen_dir)
            agent.load(agent_spec)
            registry.register(a_id, agent, meta=agent_spec.get("blueprint_meta"))
            print(f"[bootstrap] Loaded: {a_id}")
        except Exception as e:
            print(f"[bootstrap] Failed to load {a_id}: {e}")

    print(f"[bootstrap] Registry has {len(registry.all_ids())} agents")
    return registry, store
