# thesis/rq2_case_studies.py
"""
Generate qualitative case study narratives for RQ2.

Selects exemplar scenarios (one per orchestration pattern + edge cases)
and generates annotated narratives showing:
  - How the governance mechanisms operated
  - What IEEE compliance looks like in practice
  - What each explainability level reveals
  - Where gaps remain

Usage:
    python -m thesis.rq2_case_studies

Outputs (to thesis/output/case_studies/):
    case_study_direct.md           — Simple routing example
    case_study_workflow.md         — FSM workflow example
    case_study_aop.md             — AOP delegation example
    case_study_escalation.md      — HITL escalation example
    case_study_guardrail.md       — Guardrail intervention example
    case_studies_combined.md      — All cases in one file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RQ2_RESULTS_PATH = ROOT / "evaluation" / "results" / "rq2" / "rq2_results.json"
SCENARIOS_PATH = ROOT / "evaluation" / "scenarios" / "ground_truth.json"
OUTPUT_DIR = ROOT / "thesis" / "output" / "case_studies"

CATEGORY_LABELS = {
    "simple_routing": "Simple Routing (Direct)",
    "fsm_workflow": "FSM Workflow (Refund)",
    "hierarchical_delegation": "Hierarchical Delegation (AOP)",
    "hitl_escalation": "HITL Escalation",
}


def _load_rq2_results() -> list:
    if not RQ2_RESULTS_PATH.exists():
        return []
    return json.loads(RQ2_RESULTS_PATH.read_text(encoding="utf-8"))


def _load_scenarios() -> dict:
    if not SCENARIOS_PATH.exists():
        return {}
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    return {s["id"]: s for s in scenarios}


def _select_exemplar(results: list, category: str) -> dict | None:
    """Select the best exemplar for a category (highest compliance, no error)."""
    candidates = [r for r in results if r["category"] == category and r.get("error") is None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("overall_compliance", 0))


def _select_guardrail_case(results: list) -> dict | None:
    """Select a scenario where guardrails intervened."""
    for r in results:
        if r.get("guardrail_interventions", 0) > 0:
            return r
    return None


def generate_case_study(result: dict, scenario: dict, title: str) -> str:
    """Generate a detailed case study narrative."""
    lines = [f"## {title}", ""]

    # Scenario context
    lines.append(f"**Scenario:** `{result['scenario_id']}` — {result.get('description', '')}")
    lines.append(f"**Category:** {CATEGORY_LABELS.get(result['category'], result['category'])}")

    # Query
    turns = scenario.get("turns", [])
    if turns:
        lines.append(f"**Query:** \"{turns[0].get('query', 'N/A')}\"")
    lines.append("")

    # IEEE Compliance
    lines.append("### IEEE Standards Compliance")
    lines.append("")
    lines.append("| Standard | Compliance |")
    lines.append("|----------|-----------|")
    lines.append(f"| IEEE P3394 (Message Format) | {result.get('p3394_compliance', 0):.0%} |")
    lines.append(
        f"| IEEE 2894-2024 (Explainability) | {result.get('ieee_2894_compliance', 0):.0%} |"
    )
    lines.append(f"| IEEE 3152-2024 (Transparency) | {result.get('ieee_3152_compliance', 0):.0%} |")
    lines.append(f"| **Overall** | **{result.get('overall_compliance', 0):.0%}** |")
    lines.append("")

    # Explainability
    lines.append("### Explainability Coverage")
    lines.append("")
    levels = result.get("explanation_levels_available", 0)
    lines.append(f"- **Explanation levels available:** {levels}/3")
    lines.append(f"  - Summary (user-facing): {'Yes' if result.get('has_summary') else 'No'}")
    lines.append(f"  - Detailed (auditor): {'Yes' if result.get('has_detailed') else 'No'}")
    lines.append(f"  - Full (developer): {'Yes' if result.get('has_full') else 'No'}")
    lines.append(f"- **Provenance present:** {'Yes' if result.get('provenance_present') else 'No'}")
    lines.append(
        f"- **Agent identity disclosed:** {'Yes' if result.get('agent_identity_disclosed') else 'No'}"
    )
    lines.append(f"- **Decisions documented:** {result.get('decisions_documented', 0)}")
    lines.append("")

    # Governance
    lines.append("### Governance Activity")
    lines.append("")
    lines.append(f"- **Guardrail checks:** {result.get('guardrail_checks', 0)}")
    lines.append(f"- **Guardrail interventions:** {result.get('guardrail_interventions', 0)}")
    lines.append(f"- **Trace events recorded:** {result.get('trace_event_count', 0)}")
    lines.append(f"- **Processing latency:** {result.get('latency_ms', 0):.1f} ms")
    lines.append("")

    # Analysis narrative
    lines.append("### Analysis")
    lines.append("")

    compliance = result.get("overall_compliance", 0)
    if compliance >= 0.8:
        lines.append(
            "This scenario demonstrates **strong** IEEE standards alignment. "
            "The message envelope contains all required fields (P3394), "
            "multi-level explanations are generated (2894-2024), and "
            "agent identity is properly disclosed (3152-2024)."
        )
    elif compliance >= 0.5:
        lines.append(
            "This scenario shows **moderate** IEEE standards alignment. "
            "While core requirements are met, some aspects require improvement."
        )
    else:
        lines.append(
            "This scenario reveals **gaps** in IEEE standards alignment. "
            "Key areas for improvement include the items marked as non-compliant above."
        )

    if result.get("guardrail_interventions", 0) > 0:
        lines.append(
            "\nGuardrail intervention occurred, demonstrating the system's "
            "ability to enforce policy constraints while maintaining transparency "
            "about the intervention."
        )

    if result.get("provenance_present"):
        lines.append(
            "\nProvenance tracking is active: decisions can be traced back to "
            "specific routing scores, solvability estimates, and completeness checks."
        )

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_rq2_results()
    scenarios = _load_scenarios()

    if not results:
        print("  [WARN] No RQ2 results found. Run evaluation first:")
        print("         python -m evaluation.rq2_harness")
        return

    all_parts = [
        "# RQ2 Qualitative Case Studies",
        "",
        "Annotated case studies demonstrating IEEE compliance, "
        "explainability, and governance mechanisms across orchestration patterns.",
        "",
    ]

    case_configs = [
        ("simple_routing", "case_study_direct.md", "Case Study 1: Simple Routing"),
        ("fsm_workflow", "case_study_workflow.md", "Case Study 2: FSM Workflow"),
        ("hierarchical_delegation", "case_study_aop.md", "Case Study 3: AOP Delegation"),
        ("hitl_escalation", "case_study_escalation.md", "Case Study 4: HITL Escalation"),
    ]

    for category, filename, title in case_configs:
        exemplar = _select_exemplar(results, category)
        if exemplar:
            scenario = scenarios.get(exemplar["scenario_id"], {})
            md = generate_case_study(exemplar, scenario, title)
            (OUTPUT_DIR / filename).write_text(md, encoding="utf-8")
            all_parts.append(md + "\n---\n")
            print(f"  [OK] {filename}")
        else:
            print(f"  [SKIP] {filename} — no data for category '{category}'")

    # Guardrail case
    guard_case = _select_guardrail_case(results)
    if guard_case:
        scenario = scenarios.get(guard_case["scenario_id"], {})
        md = generate_case_study(guard_case, scenario, "Case Study 5: Guardrail Intervention")
        (OUTPUT_DIR / "case_study_guardrail.md").write_text(md, encoding="utf-8")
        all_parts.append(md + "\n---\n")
        print("  [OK] case_study_guardrail.md")

    # Combined file
    (OUTPUT_DIR / "case_studies_combined.md").write_text("\n".join(all_parts), encoding="utf-8")
    print("  [OK] case_studies_combined.md")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
