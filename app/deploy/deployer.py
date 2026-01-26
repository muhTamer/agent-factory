# app/deploy/deployer.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.deploy.spec_builder import build_factory_spec
from app.generator.generate_agent import generate_agent


def deploy_factory(
    plan: Dict[str, Any],
    data_dir: str,
    dry_run: bool,
    llm_client: Optional[object] = None,
) -> Dict[str, Any]:
    """
    Build spec + generate all autogen agents now (deployment step).
    Returns deployment artifact info for UI.
    """
    build_factory_spec(
        plan=plan,
        data_dir=data_dir,
        dry_run=dry_run,
        llm_client=llm_client,
    )

    # Load spec from disk (single source of truth)
    spec_path = Path(data_dir).resolve() / ".factory" / "factory_spec.json"
    spec_on_disk = json.loads(spec_path.read_text(encoding="utf-8"))

    generated: List[Dict[str, str]] = []
    for a in spec_on_disk.get("agents", []):
        if a.get("type") != "autogen":
            continue
        gen_dir = generate_agent(a)  # writes generated/<agent_id>/...
        generated.append({"id": a.get("id", ""), "path": str(gen_dir)})

    return {
        "spec_path": str(spec_path),
        "generated": generated,
        "agent_ids": [g["id"] for g in generated],
    }
