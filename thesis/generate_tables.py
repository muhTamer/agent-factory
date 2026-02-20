# thesis/generate_tables.py
"""
Generate thesis-ready tables from evaluation results.

Usage:
    python -m thesis.generate_tables

Outputs (to thesis/output/tables/):
    metrics_summary.md / .tex     — Table 1: overall metrics
    latency_by_category.md / .tex — Table 2: efficiency by pattern
    scenario_detail.md / .tex     — Table 3: per-scenario results
    solvability_by_category.md / .tex — Table 4: solvability distribution
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
OUTPUT_DIR = ROOT / "thesis" / "output" / "tables"

CATEGORY_LABELS = {
    "simple_routing": "Simple Routing",
    "fsm_workflow": "FSM Workflow",
    "hierarchical_delegation": "Hierarchical Delegation",
    "hitl_escalation": "HITL Escalation",
}


def _load_summary() -> dict:
    return json.loads((RESULTS_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))


def _load_results() -> list:
    return json.loads((RESULTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))


def _fmt(val, fmt=".2f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _stdev(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


# ── Table generators ─────────────────────────────────────────────────


def table_metrics_summary(summary: dict, results: list) -> str:
    """Table 1: Overall Evaluation Metrics."""
    total = summary.get("total_scenarios", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)

    rows = [
        ("Total Scenarios", str(total), "25", ""),
        ("Passed / Failed", f"{passed} / {failed}", "", ""),
        (
            "Orchestration Accuracy",
            _fmt(summary.get("orchestration_accuracy"), ".1%"),
            "≥ 80%",
            "✓" if (summary.get("orchestration_accuracy") or 0) >= 0.8 else "✗",
        ),
        (
            "Reasoning Accuracy",
            _fmt(summary.get("reasoning_accuracy"), ".1%"),
            "≥ 75%",
            "✓" if (summary.get("reasoning_accuracy") or 0) >= 0.75 else "✗",
        ),
        ("Agent Accuracy", _fmt(summary.get("agent_accuracy"), ".1%"), "", ""),
        (
            "Completeness Rate",
            _fmt(summary.get("completeness_rate"), ".1%"),
            "≥ 85%",
            "✓" if (summary.get("completeness_rate") or 0) >= 0.85 else "✗",
        ),
        (
            "Solvability Correlation (ρ)",
            _fmt(summary.get("solvability_correlation"), ".4f"),
            "> 0.0",
            "✓" if (summary.get("solvability_correlation") or 0) > 0 else "✗",
        ),
        ("Mean Latency", f"{_fmt(summary.get('avg_latency_ms'), '.1f')} ms", "", ""),
    ]

    lines = [
        "# Table 1: Overall Evaluation Metrics",
        "",
        "| Metric | Value | Target | Met? |",
        "|--------|-------|--------|------|",
    ]
    for name, val, target, met in rows:
        lines.append(f"| {name} | {val} | {target} | {met} |")
    return "\n".join(lines)


def table_latency_by_category(summary: dict, results: list) -> str:
    """Table 2: Efficiency by Orchestration Pattern."""
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, {"latencies": [], "calls": []})
        cats[cat]["latencies"].append(r["latency_ms"])
        cats[cat]["calls"].append(r["agent_calls"])

    lines = [
        "# Table 2: Orchestration Efficiency by Pattern",
        "",
        "| Pattern | N | Mean Latency (ms) | Std (ms) | Mean Agent Calls |",
        "|---------|--:|------------------:|---------:|-----------------:|",
    ]
    for key in ["simple_routing", "fsm_workflow", "hierarchical_delegation", "hitl_escalation"]:
        if key not in cats:
            continue
        data = cats[key]
        n = len(data["latencies"])
        mean_lat = statistics.mean(data["latencies"])
        std_lat = _stdev(data["latencies"])
        mean_calls = statistics.mean(data["calls"])
        label = CATEGORY_LABELS.get(key, key)
        lines.append(f"| {label} | {n} | {mean_lat:.1f} | {std_lat:.1f} | {mean_calls:.1f} |")
    return "\n".join(lines)


def table_scenario_detail(summary: dict, results: list) -> str:
    """Table 3: Per-Scenario Results."""
    lines = [
        "# Table 3: Per-Scenario Evaluation Results",
        "",
        "| ID | Category | Success | Pattern ✓ | Agent ✓ | Keywords ✓ | Latency (ms) | Calls | Solvability |",
        "|----|----------|---------|-----------|---------|------------|-------------:|------:|------------:|",
    ]
    for r in results:
        cat = CATEGORY_LABELS.get(r["category"], r["category"])
        succ = "✓" if r["success"] else "✗"
        pat = "✓" if r["pattern_correct"] else "✗"
        agt = "✓" if r["agent_correct"] else "✗"
        kw = "✓" if r["answer_keywords_found"] else "✗"
        solv = _fmt(r.get("solvability_score"), ".2f")
        lines.append(
            f"| {r['scenario_id']} | {cat} | {succ} | {pat} | {agt} | {kw} "
            f"| {r['latency_ms']:.1f} | {r['agent_calls']} | {solv} |"
        )
    return "\n".join(lines)


def table_solvability_by_category(summary: dict, results: list) -> str:
    """Table 4: Solvability Score Distribution by Category."""
    cats = {}
    for r in results:
        score = r.get("solvability_score")
        if score is not None:
            cats.setdefault(r["category"], []).append(score)

    lines = [
        "# Table 4: Solvability Score Distribution by Category",
        "",
        "| Pattern | N | Min | Max | Mean | Std |",
        "|---------|--:|----:|----:|-----:|----:|",
    ]
    for key in ["simple_routing", "fsm_workflow", "hierarchical_delegation", "hitl_escalation"]:
        if key not in cats:
            continue
        vals = cats[key]
        label = CATEGORY_LABELS.get(key, key)
        lines.append(
            f"| {label} | {len(vals)} | {min(vals):.2f} | {max(vals):.2f} "
            f"| {statistics.mean(vals):.2f} | {_stdev(vals):.2f} |"
        )
    return "\n".join(lines)


# ── LaTeX conversion ─────────────────────────────────────────────────


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
        # Skip separator rows
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
            # Escape special chars
            escaped = []
            for c in cells:
                c = c.replace("✓", "\\checkmark").replace("✗", "\\texttimes")
                c = c.replace("≥", "$\\geq$").replace(">", "$>$")
                c = c.replace("ρ", "$\\rho$").replace("%", "\\%")
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

    tables = {
        "metrics_summary": table_metrics_summary,
        "latency_by_category": table_latency_by_category,
        "scenario_detail": table_scenario_detail,
        "solvability_by_category": table_solvability_by_category,
    }

    for name, fn in tables.items():
        md = fn(summary, results)
        (OUTPUT_DIR / f"{name}.md").write_text(md, encoding="utf-8")
        (OUTPUT_DIR / f"{name}.tex").write_text(_md_table_to_latex(md), encoding="utf-8")
        print(f"  [OK] {name}.md + .tex")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
