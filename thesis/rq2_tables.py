# thesis/rq2_tables.py
"""
Generate thesis-ready tables for RQ2 evaluation results.

Usage:
    python -m thesis.rq2_tables

Outputs (to thesis/output/tables/):
    rq2_ieee_compliance.md / .tex     — IEEE compliance by standard
    rq2_explainability.md / .tex      — Explainability coverage metrics
    rq2_governance.md / .tex          — Governance mechanism activity
    rq2_compliance_by_category.md/.tex — Compliance by orchestration pattern
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results" / "rq2"
OUTPUT_DIR = ROOT / "thesis" / "output" / "tables"

CATEGORY_LABELS = {
    "simple_routing": "Simple Routing",
    "fsm_workflow": "FSM Workflow",
    "hierarchical_delegation": "Hierarchical Delegation",
    "hitl_escalation": "HITL Escalation",
}


def _load_summary() -> dict:
    path = RESULTS_DIR / "rq2_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_results() -> list:
    path = RESULTS_DIR / "rq2_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val, fmt=".2f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _stdev(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


# ── Table generators ─────────────────────────────────────────────────


def table_ieee_compliance(summary: dict, results: list) -> str:
    """Table: IEEE compliance rate by standard."""
    rows = [
        (
            "IEEE P3394 (Universal Message Format)",
            _fmt(summary.get("p3394_compliance"), ".1%"),
            "Message envelope, sender/receiver, provenance",
        ),
        (
            "IEEE 2894-2024 (Explainable AI)",
            _fmt(summary.get("ieee_2894_compliance"), ".1%"),
            "Multi-level explanations, decision rationale",
        ),
        (
            "IEEE 3152-2024 (Transparent Agency)",
            _fmt(summary.get("ieee_3152_compliance"), ".1%"),
            "AI disclosure, agent identity, audit trail",
        ),
        (
            "**Overall Compliance**",
            _fmt(summary.get("overall_compliance"), ".1%"),
            f"Across {summary.get('total_scenarios', 0)} scenarios",
        ),
    ]

    lines = [
        "# Table: IEEE Standards Compliance (RQ2)",
        "",
        "| Standard | Compliance Rate | Key Requirements |",
        "|----------|----------------:|------------------|",
    ]
    for name, rate, reqs in rows:
        lines.append(f"| {name} | {rate} | {reqs} |")
    return "\n".join(lines)


def table_explainability(summary: dict, results: list) -> str:
    """Table: Explainability coverage metrics."""
    rows = [
        ("Summary-level (user-facing)", _fmt(summary.get("summary_coverage"), ".1%")),
        ("Detailed-level (auditor)", _fmt(summary.get("detailed_coverage"), ".1%")),
        ("Full-level (developer)", _fmt(summary.get("full_coverage"), ".1%")),
        ("Provenance present", _fmt(summary.get("provenance_rate"), ".1%")),
        ("Agent identity disclosed", _fmt(summary.get("agent_identity_rate"), ".1%")),
        (
            "Mean decisions documented",
            _fmt(summary.get("mean_decisions_documented"), ".1f"),
        ),
        (
            "Mean explanation levels",
            _fmt(summary.get("mean_explanation_levels"), ".1f"),
        ),
    ]

    lines = [
        "# Table: Explainability Coverage (RQ2)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for name, val in rows:
        lines.append(f"| {name} | {val} |")
    return "\n".join(lines)


def table_governance(summary: dict, results: list) -> str:
    """Table: Governance mechanism activity."""
    rows = [
        (
            "Mean guardrail checks per request",
            _fmt(summary.get("mean_guardrail_checks"), ".1f"),
        ),
        (
            "Total guardrail interventions",
            str(summary.get("total_guardrail_interventions", 0)),
        ),
        (
            "Mean trace events per request",
            _fmt(summary.get("mean_trace_events"), ".1f"),
        ),
        ("Mean latency (ms)", _fmt(summary.get("mean_latency_ms"), ".1f")),
    ]

    lines = [
        "# Table: Governance Mechanism Activity (RQ2)",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]
    for name, val in rows:
        lines.append(f"| {name} | {val} |")
    return "\n".join(lines)


def table_compliance_by_category(summary: dict, results: list) -> str:
    """Table: Compliance broken down by orchestration pattern."""
    by_cat = summary.get("compliance_by_category", {})

    lines = [
        "# Table: IEEE Compliance by Orchestration Pattern (RQ2)",
        "",
        "| Pattern | N | Overall | P3394 | 2894 | 3152 |",
        "|---------|--:|--------:|------:|-----:|-----:|",
    ]
    for key in [
        "simple_routing",
        "fsm_workflow",
        "hierarchical_delegation",
        "hitl_escalation",
    ]:
        if key not in by_cat:
            continue
        data = by_cat[key]
        label = CATEGORY_LABELS.get(key, key)
        lines.append(
            f"| {label} | {data['n']} "
            f"| {data['overall']:.0%} "
            f"| {data['p3394']:.0%} "
            f"| {data['2894']:.0%} "
            f"| {data['3152']:.0%} |"
        )
    return "\n".join(lines)


# ── LaTeX conversion (reuse from generate_tables.py) ─────────────────


def _md_table_to_latex(md: str) -> str:
    """Convert a markdown table to LaTeX tabular."""
    out = []
    in_table = False
    col_count = 0
    title = ""

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:]
            continue
        if not stripped.startswith("|"):
            continue
        if all(c in "|-: " for c in stripped):
            continue

        cells = [c.strip() for c in stripped.split("|")[1:-1]]
        if not in_table:
            col_count = len(cells)
            col_spec = "l" + "r" * (col_count - 1)
            out.append("\\begin{table}[htbp]")
            out.append("\\centering")
            out.append(f"\\caption{{{title}}}")
            out.append(f"\\begin{{tabular}}{{{col_spec}}}")
            out.append("\\toprule")
            out.append(" & ".join(f"\\textbf{{{c}}}" for c in cells) + " \\\\")
            out.append("\\midrule")
            in_table = True
        else:
            escaped = []
            for c in cells:
                c = c.replace("%", "\\%").replace("**", "")
                c = c.replace("≥", "$\\geq$").replace(">", "$>$")
                escaped.append(c)
            out.append(" & ".join(escaped) + " \\\\")

    if in_table:
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()
    results = _load_results()

    if not summary:
        print("  [WARN] No RQ2 results found. Run evaluation first:")
        print("         python -m evaluation.rq2_harness")
        return

    tables = {
        "rq2_ieee_compliance": table_ieee_compliance,
        "rq2_explainability": table_explainability,
        "rq2_governance": table_governance,
        "rq2_compliance_by_category": table_compliance_by_category,
    }

    for name, fn in tables.items():
        md = fn(summary, results)
        (OUTPUT_DIR / f"{name}.md").write_text(md, encoding="utf-8")
        (OUTPUT_DIR / f"{name}.tex").write_text(
            _md_table_to_latex(md), encoding="utf-8"
        )
        print(f"  [OK] {name}.md + .tex")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
