# thesis/generate_figures.py
"""
Generate thesis-ready matplotlib figures from evaluation and performance data.

Usage:
    python -m thesis.generate_figures

Outputs (to thesis/output/figures/):
    fig_latency_by_pattern.pdf/.png
    fig_agent_calls_by_pattern.pdf/.png
    fig_solvability_distribution.pdf/.png
    fig_learning_curve.pdf/.png
    fig_pipeline_stages.pdf/.png
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
PERF_PATH = ROOT / ".factory" / "performance_history.json"
TRACES_PATH = ROOT / ".factory" / "audit" / "runtime_traces.jsonl"
OUTPUT_DIR = ROOT / "thesis" / "output" / "figures"

CATEGORY_ORDER = [
    "simple_routing",
    "fsm_workflow",
    "hierarchical_delegation",
    "hitl_escalation",
]
CATEGORY_LABELS = {
    "simple_routing": "Simple\nRouting",
    "fsm_workflow": "FSM\nWorkflow",
    "hierarchical_delegation": "Hierarchical\nDelegation",
    "hitl_escalation": "HITL\nEscalation",
}
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


def _load_results() -> list:
    return json.loads((RESULTS_DIR / "evaluation_results.json").read_text(encoding="utf-8"))


def _load_performance() -> list:
    if not PERF_PATH.exists():
        return []
    return json.loads(PERF_PATH.read_text(encoding="utf-8"))


def _load_traces() -> list:
    if not TRACES_PATH.exists():
        return []
    traces = []
    for line in TRACES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


def _setup_style():
    """Set academic-style defaults."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  [OK] {name}.pdf + .png")


# ── Figures ───────────────────────────────────────────────────────────


def fig_latency_by_pattern(results: list):
    """Bar chart: mean latency per orchestration pattern with error bars."""
    import matplotlib.pyplot as plt

    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["latency_ms"])

    names = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER if c in cats]
    means = [statistics.mean(cats[c]) for c in CATEGORY_ORDER if c in cats]
    stds = [
        statistics.stdev(cats[c]) if len(cats[c]) > 1 else 0 for c in CATEGORY_ORDER if c in cats
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        capsize=5,
        color=COLORS[: len(names)],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Mean Latency by Orchestration Pattern")
    ax.set_ylim(bottom=0)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, "fig_latency_by_pattern")


def fig_agent_calls_by_pattern(results: list):
    """Bar chart: mean agent calls per pattern."""
    import matplotlib.pyplot as plt

    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["agent_calls"])

    names = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER if c in cats]
    means = [statistics.mean(cats[c]) for c in CATEGORY_ORDER if c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, color=COLORS[: len(names)], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Agent Calls")
    ax.set_title("Mean Agent Calls by Orchestration Pattern")
    ax.set_ylim(bottom=0, top=max(means) + 0.5)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, "fig_agent_calls_by_pattern")


def fig_solvability_distribution(results: list):
    """Box plot: solvability scores by category."""
    import matplotlib.pyplot as plt

    cats = {}
    for r in results:
        score = r.get("solvability_score")
        if score is not None:
            cats.setdefault(r["category"], []).append(score)

    data = [cats.get(c, []) for c in CATEGORY_ORDER if c in cats]
    labels = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER if c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Solvability Score")
    ax.set_title("Solvability Score Distribution by Pattern")
    ax.set_ylim(0, 1.0)

    _save(fig, "fig_solvability_distribution")


def fig_learning_curve(history: list):
    """Line chart: agent performance over time (feedback loop)."""
    import matplotlib.pyplot as plt

    if not history:
        print("  [SKIP] fig_learning_curve — no performance history data")
        return

    # Group by agent_id, sort by timestamp
    agents = {}
    for rec in history:
        aid = rec.get("agent_id", "unknown")
        agents.setdefault(aid, []).append(rec)
    for aid in agents:
        agents[aid].sort(key=lambda r: r.get("timestamp", 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (aid, records) in enumerate(agents.items()):
        scores = [r.get("score", 0) for r in records]
        # Running average
        running = []
        total = 0
        for j, s in enumerate(scores, 1):
            total += s
            running.append(total / j)
        ax.plot(
            range(1, len(running) + 1),
            running,
            marker="o",
            markersize=4,
            label=aid,
            color=COLORS[i % len(COLORS)],
        )

    ax.set_xlabel("Execution #")
    ax.set_ylabel("Cumulative Average Score")
    ax.set_title("Agent Performance Learning Curve (AOP Feedback Loop)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.0)

    _save(fig, "fig_learning_curve")


def fig_pipeline_stages(traces: list):
    """Stacked horizontal bar: time per pipeline stage for representative traces."""
    import matplotlib.pyplot as plt

    if not traces:
        print("  [SKIP] fig_pipeline_stages — no runtime traces")
        return

    # Pick up to 4 traces with the most events
    traces_sorted = sorted(traces, key=lambda t: len(t.get("events", [])), reverse=True)
    selected = traces_sorted[:4]

    stage_order = [
        "request_received",
        "orchestration_pattern",
        "route",
        "intent_inferred",
        "guard_pre_ok",
        "execute",
        "select",
        "response_ready",
        "guard_post_ok",
    ]
    fig, ax = plt.subplots(figsize=(10, 3 + len(selected)))
    y_labels = []
    for idx, trace in enumerate(selected):
        events = trace.get("events", [])
        base = trace.get("started_ts_ms", 0)
        query = trace.get("query", "")[:40]
        y_labels.append(query)

        prev_ts = base
        for ev in events:
            stage = ev.get("stage", "")
            ts = ev.get("ts_ms", base)
            delta = ts - prev_ts
            if stage in stage_order:
                si = stage_order.index(stage)
                ax.barh(
                    idx,
                    delta,
                    left=prev_ts - base,
                    color=COLORS[si % len(COLORS)],
                    edgecolor="white",
                    linewidth=0.3,
                )
            prev_ts = ts

    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pipeline Stage Breakdown per Request")

    _save(fig, "fig_pipeline_stages")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    import matplotlib

    matplotlib.use("Agg")
    _setup_style()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = _load_results()
    history = _load_performance()
    traces = _load_traces()

    fig_latency_by_pattern(results)
    fig_agent_calls_by_pattern(results)
    fig_solvability_distribution(results)
    fig_learning_curve(history)
    fig_pipeline_stages(traces)

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
