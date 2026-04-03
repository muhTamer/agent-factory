#!/usr/bin/env python3
"""
Regenerate agents from workspace documents with correct doc_visibility,
then verify compatibility with evaluation scenarios.

Runs the full pipeline:
  1. PlannerInterface.generate_plan_preview() — LLM analyzes workspace docs
  2. build_factory_spec() — creates factory_spec.json with doc_visibility
  3. generate_agent() — creates agent packages in generated/
  4. Verify generated agent IDs match evaluation ground_truth.json

Usage:
    python -m scripts.regenerate_agents
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.concierge.planner_interface import PlannerInterface  # noqa: E402
from app.deploy.spec_builder import build_factory_spec  # noqa: E402
from app.generator.generate_agent import generate_agent  # noqa: E402

WORKSPACE = REPO_ROOT / ".workspace"
GENERATED_DIR = REPO_ROOT / "generated"
GROUND_TRUTH = REPO_ROOT / "evaluation" / "scenarios" / "ground_truth.json"

# User-defined document visibility.
# Internal = agent instructions only (not shareable with customers).
# Customer-facing = content that can be shared with customers.
DOC_VISIBILITY = {
    "refunds_policy.yaml": "internal",
    "complaints_policy.yaml": "internal",
    "accounts_policy.yaml": "internal",
    "BankFAQs.csv": "customer_facing",
}

# Domain keywords expected in agent IDs.
# ground_truth.json uses these for substring matching (e.g. "refunds" in "refunds_agent").
EXPECTED_DOMAINS = {"refunds", "complaints", "accounts", "faq"}


def _on_rm_error(func, path, exc_info):
    """Handle Windows/OneDrive permission errors during rmtree."""
    import stat
    import os
    import time

    os.chmod(path, stat.S_IWRITE)
    time.sleep(0.1)
    func(path)


def main():
    print("=" * 60)
    print("Agent Regeneration Pipeline")
    print("=" * 60)

    # Clean old generated agents
    if GENERATED_DIR.exists():
        print(f"\nCleaning old generated agents: {GENERATED_DIR}")
        shutil.rmtree(GENERATED_DIR, onerror=_on_rm_error)

    # Step 1: Generate plan preview (LLM analyzes workspace docs)
    print("\n[1/3] Analyzing workspace documents via LLM...")
    planner = PlannerInterface(
        vertical="fintech",
        data_dir=str(WORKSPACE),
    )
    plan = planner.generate_plan_preview(use_llm=True)
    print(f"  Vertical: {plan.get('vertical', '?')}")
    agents_planned = plan.get("agents", [])
    print(f"  Agents planned: {len(agents_planned)}")
    for bp in agents_planned:
        print(f"    - {bp.get('id', 'unknown')}: {bp.get('agent_kind', '?')}")

    # Step 2: Build factory spec with doc_visibility
    print("\n[2/3] Building factory spec with doc_visibility...")
    spec = build_factory_spec(
        plan=plan,
        data_dir=str(WORKSPACE),
        dry_run=True,
        doc_visibility=DOC_VISIBILITY,
    )

    # Verify document assignments and visibility
    for agent in spec.get("agents", []):
        if agent.get("type") != "autogen":
            continue
        a_id = agent["id"]
        ks = agent.get("inputs", {}).get("knowledge_sources", [])
        vis_map = agent.get("inputs", {}).get("doc_visibility_map", {})
        meta = agent.get("blueprint_meta", {})
        print(f"  {a_id}:")
        print(f"    knowledge_sources: {[Path(k).name for k in ks]}")
        print(f"    doc_visibility_map: {vis_map}")
        print(f"    has_customer_facing_docs: {meta.get('has_customer_facing_docs')}")
        print(f"    has_internal_policy: {meta.get('has_internal_policy')}")
        print(f"    document_categories: {meta.get('document_categories', [])}")

    # Step 3: Generate agent packages
    print("\n[3/3] Generating agents...")
    generated = []
    errors = []
    for agent in spec.get("agents", []):
        if agent.get("type") != "autogen":
            continue
        a_id = agent["id"]
        try:
            gen_dir = generate_agent(agent)
            generated.append(a_id)
            print(f"  Generated: {a_id} -> {gen_dir}")
        except Exception as e:
            errors.append((a_id, str(e)))
            print(f"  FAILED: {a_id} -> {e}")

    # Step 4: Verify evaluation compatibility
    print(f"\n{'=' * 60}")
    print("Evaluation Compatibility Check")
    print("=" * 60)

    # Check that each expected domain has a matching agent

    missing_domains = []
    domain_agent_map = {}
    for domain in EXPECTED_DOMAINS:
        matches = [a for a in generated if domain in a.lower()]
        if matches:
            domain_agent_map[domain] = matches[0]
            print(f"  {domain} -> {matches[0]}")
        else:
            missing_domains.append(domain)
            print(f"  {domain} -> MISSING!")

    if missing_domains:
        print(f"\n  WARNING: Missing agents for domains: {missing_domains}")
        print("  ground_truth.json scenarios for these domains will FAIL.")
        print("  Consider adjusting the inference prompt or adding naming hints.")
    else:
        print(f"\n  All {len(EXPECTED_DOMAINS)} expected domains have matching agents.")

    # Verify ground_truth.json expected_agents are compatible
    if GROUND_TRUTH.exists():
        gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
        incompatible = []
        for scenario in gt:
            for turn in scenario.get("turns", []):
                for ea in turn.get("expected", {}).get("expected_agents", []):
                    if not any(ea.lower() in a.lower() for a in generated):
                        incompatible.append((scenario["id"], ea))
        if incompatible:
            print(
                f"\n  WARNING: {len(incompatible)} scenario/agent pairs have no match:"
            )
            for sid, ea in incompatible[:10]:
                print(f"    {sid}: expected '{ea}' but no agent contains it")
        else:
            print("  All ground_truth.json expected_agents are compatible.")

    print(f"\n{'=' * 60}")
    print(f"Done. Generated {len(generated)} agents, {len(errors)} errors.")
    if errors:
        for a_id, err in errors:
            print(f"  ERROR {a_id}: {err}")
    print("\nGenerated agent IDs:")
    for a_id in generated:
        print(f"  - {a_id}")


if __name__ == "__main__":
    main()
