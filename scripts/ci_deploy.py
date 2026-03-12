#!/usr/bin/env python3
"""
CI Factory Deploy — runs the full quickstart-fintech + deploy flow
without starting a server.

Usage:
    python scripts/ci_deploy.py

Requires LLM credentials (OPENAI_API_KEY or AZURE_OPENAI_* env vars).

Stages:
    1. Copy data files to .workspace/
    2. Run inference (InferCapabilities) to generate plan
    3. Build factory spec (compiles policies, generates agent blueprints)
    4. Generate all agents
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
WORKSPACE = REPO_ROOT / ".workspace"

FINTECH_DATA_FILES = [
    DATA_DIR / "BankFAQs.csv",
    DATA_DIR / "refunds_policy.yaml",
    DATA_DIR / "complaints_policy.yaml",
]


def main() -> int:
    import app.llm_client as llm_client
    from app.concierge.concierge_agent import ConciergeAgent

    # ── Stage 1: Copy data files to workspace ──────────────────────
    print("[1/4] Copying data files to .workspace/ ...")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for src in FINTECH_DATA_FILES:
        if not src.exists():
            print(f"  ERROR: {src} not found")
            return 1
        shutil.copy2(src, WORKSPACE / src.name)
        print(f"  {src.name}")

    # ── Stage 2: Run inference ─────────────────────────────────────
    print("[2/4] Running inference (InferCapabilities) ...")
    agent = ConciergeAgent(
        vertical="fintech",
        data_dir=str(WORKSPACE),
        llm_client=llm_client,
    )
    result = agent.handle_event({"type": "upload_docs", "use_llm": True})
    plan = result.get("plan") or agent.state.get("last_plan")
    if not plan:
        print("  ERROR: Inference returned no plan")
        print(f"  Result: {json.dumps(result, indent=2, default=str)[:500]}")
        return 1
    print(f"  Vertical: {plan.get('vertical')}")
    print(f"  Agents proposed: {len(plan.get('agents', []))}")

    # ── Stage 3 & 4: Build spec + generate agents ─────────────────
    print("[3/4] Building factory spec + compiling policies ...")
    print("[4/4] Generating agents ...")
    deploy_result = agent.handle_event(
        {"type": "user_action", "action": "approve_deploy_dry"}
    )

    generated = deploy_result.get("generated_agents", [])
    errors = deploy_result.get("generation_errors", [])

    print(f"\n{'=' * 60}")
    print("FACTORY DEPLOY COMPLETE")
    print(f"{'=' * 60}")

    # Verify key artifacts
    artifacts = {
        "factory_spec": REPO_ROOT / ".factory" / "factory_spec.json",
        "policy_pack": None,  # find dynamically
        "faq_agent": REPO_ROOT / "generated" / "customer_facing_rag" / "agent.py",
        "refunds_agent": REPO_ROOT / "generated" / "refunds_workflow" / "agent.py",
        "faqs_json": REPO_ROOT / "generated" / "customer_facing_rag" / "faqs.json",
    }

    # Find compiled policy pack
    policy_dir = REPO_ROOT / ".factory" / "compiled_policies"
    if policy_dir.exists():
        packs = list(policy_dir.glob("*.json"))
        if packs:
            artifacts["policy_pack"] = packs[0]

    all_ok = True
    for name, path in artifacts.items():
        if path and path.exists():
            print(f"  OK  {name}: {path.relative_to(REPO_ROOT)}")
        else:
            print(f"  MISSING  {name}: {path}")
            all_ok = False

    print(f"\nGenerated: {len(generated)} agents")
    if errors:
        print(f"Errors: {errors}")

    if not all_ok:
        print(
            "\nWARNING: Some artifacts missing — deploy-dependent tests may still skip."
        )

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
