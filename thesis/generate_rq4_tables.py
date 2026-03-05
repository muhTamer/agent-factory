# thesis/generate_rq4_tables.py
"""
Generate thesis-ready tables from RQ4 evaluation results.

Usage:
    python -m thesis.generate_rq4_tables

Outputs (to thesis/output/tables/):
    rq4_tts_by_strategy.md / .tex      — Table: TTS scores by strategy
    rq4_tts_by_persona.md / .tex       — Table: TTS scores by persona
    rq4_strategy_persona_matrix.md / .tex — Table: Strategy x Persona grid
    rq4_tts_by_category.md / .tex      — Table: TTS scores by scenario category
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RQ4_RESULTS_DIR = ROOT / "evaluation" / "rq4" / "results"
OUTPUT_DIR = ROOT / "thesis" / "output" / "tables"

STRATEGY_LABELS = {
    "baseline": "Baseline",
    "transparent": "Transparent",
    "empathetic": "Empathetic",
    "proactive": "Proactive",
}

CATEGORY_LABELS = {
    "informational": "Informational",
    "transactional": "Transactional",
    "complaint": "Complaint",
    "complex_multi_intent": "Complex Multi-Intent",
    "trust_sensitive": "Trust-Sensitive",
}


def _load_metrics() -> dict:
    path = RQ4_RESULTS_DIR / "rq4_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"RQ4 metrics not found at {path}. "
            "Run: python -m evaluation.rq4.run_rq4_evaluation first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val, fmt=".2f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


# ── Table generators ──────────────────────────────────────────────────


def table_tts_by_strategy(metrics: dict) -> str:
    """Table: Mean TTS scores by orchestration strategy."""
    by_strategy = metrics.get("by_strategy", {})

    lines = [
        "# Table: Perceived Quality Scores by Orchestration Strategy (RQ4)",
        "",
        "| Strategy | Transparency | Trust | Satisfaction | Composite | N |",
        "|----------|------------:|------:|-------------:|----------:|--:|",
    ]

    for slug in ["baseline", "transparent", "empathetic", "proactive"]:
        stats = by_strategy.get(slug, {})
        label = STRATEGY_LABELS.get(slug, slug)
        t = stats.get("transparency", {})
        tr = stats.get("trust", {})
        s = stats.get("satisfaction", {})
        comp = stats.get("composite", {})
        n = t.get("n", 0)
        lines.append(
            f"| {label} "
            f"| {_fmt(t.get('mean'))} ({_fmt(t.get('std'))}) "
            f"| {_fmt(tr.get('mean'))} ({_fmt(tr.get('std'))}) "
            f"| {_fmt(s.get('mean'))} ({_fmt(s.get('std'))}) "
            f"| {_fmt(comp.get('mean'))} "
            f"| {n} |"
        )

    # Add statistical tests if available
    stat_tests = metrics.get("statistical_tests", {})
    if stat_tests and "note" not in stat_tests:
        lines.append("")
        lines.append("*Kruskal-Wallis tests:*")
        for dim in ("transparency", "trust", "satisfaction"):
            test = stat_tests.get(dim, {})
            if "p_value" in test:
                sig = "p < 0.05" if test["significant"] else "n.s."
                lines.append(
                    f"- {dim.capitalize()}: H={test['H_statistic']:.2f}, p={test['p_value']:.4f} ({sig})"
                )

    return "\n".join(lines)


def table_tts_by_persona(metrics: dict) -> str:
    """Table: Mean TTS scores by customer persona."""
    by_persona = metrics.get("by_persona", {})

    lines = [
        "# Table: Perceived Quality Scores by Customer Persona (RQ4)",
        "",
        "| Persona | Transparency | Trust | Satisfaction | Composite | N |",
        "|---------|------------:|------:|-------------:|----------:|--:|",
    ]

    for name, stats in by_persona.items():
        t = stats.get("transparency", {})
        tr = stats.get("trust", {})
        s = stats.get("satisfaction", {})
        comp = stats.get("composite", {})
        n = t.get("n", 0)
        lines.append(
            f"| {name} "
            f"| {_fmt(t.get('mean'))} ({_fmt(t.get('std'))}) "
            f"| {_fmt(tr.get('mean'))} ({_fmt(tr.get('std'))}) "
            f"| {_fmt(s.get('mean'))} ({_fmt(s.get('std'))}) "
            f"| {_fmt(comp.get('mean'))} "
            f"| {n} |"
        )

    return "\n".join(lines)


def table_strategy_persona_matrix(metrics: dict) -> str:
    """Table: Strategy x Persona interaction matrix (composite scores)."""
    matrix = metrics.get("strategy_persona_matrix", {})
    persona_names = list(metrics.get("by_persona", {}).keys())

    # Header
    lines = [
        "# Table: Strategy x Persona Interaction Matrix — Composite Scores (RQ4)",
        "",
    ]

    header = "| Strategy |"
    sep = "|----------|"
    for name in persona_names:
        short = name.split()[0] if len(name) > 12 else name
        header += f" {short} |"
        sep += "------:|"

    lines.append(header)
    lines.append(sep)

    for slug in ["baseline", "transparent", "empathetic", "proactive"]:
        label = STRATEGY_LABELS.get(slug, slug)
        row = f"| {label} |"
        strategy_data = matrix.get(slug, {})
        for name in persona_names:
            cell = strategy_data.get(name, {})
            comp = cell.get("composite", {}).get("mean")
            row += f" {_fmt(comp)} |"
        lines.append(row)

    return "\n".join(lines)


def table_tts_by_category(metrics: dict) -> str:
    """Table: Mean TTS scores by scenario category."""
    by_category = metrics.get("by_category", {})

    lines = [
        "# Table: Perceived Quality Scores by Scenario Category (RQ4)",
        "",
        "| Category | Transparency | Trust | Satisfaction | Composite | N |",
        "|----------|------------:|------:|-------------:|----------:|--:|",
    ]

    for key in [
        "informational",
        "transactional",
        "complaint",
        "complex_multi_intent",
        "trust_sensitive",
    ]:
        stats = by_category.get(key, {})
        if not stats:
            continue
        label = CATEGORY_LABELS.get(key, key)
        t = stats.get("transparency", {})
        tr = stats.get("trust", {})
        s = stats.get("satisfaction", {})
        comp = stats.get("composite", {})
        n = t.get("n", 0)
        lines.append(
            f"| {label} "
            f"| {_fmt(t.get('mean'))} ({_fmt(t.get('std'))}) "
            f"| {_fmt(tr.get('mean'))} ({_fmt(tr.get('std'))}) "
            f"| {_fmt(s.get('mean'))} ({_fmt(s.get('std'))}) "
            f"| {_fmt(comp.get('mean'))} "
            f"| {n} |"
        )

    return "\n".join(lines)


# ── LaTeX conversion ──────────────────────────────────────────────────


def _md_table_to_latex(md: str) -> str:
    """Convert a markdown table to LaTeX tabular (reuses pattern from generate_tables.py)."""
    out = []
    in_table = False
    col_count = 0
    title = ""

    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:]
            continue
        if stripped.startswith("*"):
            # Statistical test notes — add as table footnote
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
                c = c.replace("%", "\\%")
                c = c.replace("≥", "$\\geq$").replace(">", "$>$")
                escaped.append(c)
            out.append(" & ".join(escaped) + " \\\\")

    if in_table:
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

    return "\n".join(out)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        metrics = _load_metrics()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        print("  Run the RQ4 evaluation first:")
        print("    python -m evaluation.rq4.run_rq4_evaluation")
        return

    tables = {
        "rq4_tts_by_strategy": table_tts_by_strategy,
        "rq4_tts_by_persona": table_tts_by_persona,
        "rq4_strategy_persona_matrix": table_strategy_persona_matrix,
        "rq4_tts_by_category": table_tts_by_category,
    }

    for name, fn in tables.items():
        md = fn(metrics)
        (OUTPUT_DIR / f"{name}.md").write_text(md, encoding="utf-8")
        (OUTPUT_DIR / f"{name}.tex").write_text(_md_table_to_latex(md), encoding="utf-8")
        print(f"  [OK] {name}.md + .tex")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
