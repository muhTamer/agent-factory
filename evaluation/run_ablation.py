# evaluation/run_ablation.py
"""
Ablation Study Runner — Orchestration Depth Comparison

Runs the full 58-scenario evaluation in three orchestration modes:
  1. direct_only  — always single-agent dispatch (no AOP)
  2. aop_only     — always hierarchical delegation (full AOP)
  3. hybrid       — LLM classifies each query (default architecture)

Each mode writes results to a separate subdirectory under
evaluation/results/ablation/{mode}/.

Usage:
    python -m evaluation.run_ablation                     # all 3 modes
    python -m evaluation.run_ablation --mode direct_only  # single mode
    python -m evaluation.run_ablation --dry-run            # 3 scenarios only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Force UTF-8 stdout/stderr on Windows ──
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ("direct_only", "aop_only", "hybrid")


def run_single_mode(
    mode: str,
    output_base: Path,
    max_scenarios: int | None = None,
) -> dict:
    """Run evaluation in a specific orchestration mode."""
    # Set the env var BEFORE importing spine (which reads it at module level)
    os.environ["ORCHESTRATION_MODE"] = mode

    # Force re-read of the module-level constant in spine
    import app.runtime.spine as spine_mod

    spine_mod.ORCHESTRATION_MODE = mode

    # Import the evaluation runner (lazy to pick up env changes)
    from evaluation.run_evaluation import run_evaluation

    output_dir = output_base / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  ABLATION STUDY — MODE: {mode.upper()}")
    print("=" * 70)

    metrics = run_evaluation(
        output_dir=output_dir,
        max_scenarios=max_scenarios,
    )

    return metrics


def run_ablation(
    modes: tuple[str, ...] = MODES,
    output_base: Path | None = None,
    max_scenarios: int | None = None,
) -> dict:
    """Run ablation study across specified modes."""
    if output_base is None:
        output_base = Path("evaluation/results/ablation")

    all_metrics = {}
    for mode in modes:
        t0 = time.time()
        metrics = run_single_mode(mode, output_base, max_scenarios)
        elapsed = time.time() - t0
        metrics["mode"] = mode
        metrics["wall_time_seconds"] = round(elapsed, 2)
        all_metrics[mode] = metrics

    # Write comparison summary
    output_base.mkdir(parents=True, exist_ok=True)
    comparison = {}
    for mode, m in all_metrics.items():
        total = m.get("total_scenarios", 0)
        passed = m.get("passed", 0)
        comparison[mode] = {
            "passed": passed,
            "failed": m.get("failed", 0),
            "total": total,
            "task_completion_rate": round(passed / total, 4) if total else 0,
            "agent_accuracy": m.get("agent_accuracy", 0),
            "reasoning_accuracy": m.get("reasoning_accuracy", 0),
            "avg_latency_ms": m.get("avg_latency_ms", 0),
            "wall_time_seconds": m.get("wall_time_seconds", 0),
            "by_category": m.get("by_category", {}),
        }

    summary_path = output_base / "ablation_comparison.json"
    summary_path.write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )

    # Print comparison table
    # Note: orchestration_accuracy (pattern correctness) is excluded because
    # ablation forces a single pattern — measuring pattern match is meaningless.
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — COMPARISON")
    print("=" * 70)
    print(
        f"  {'Mode':<15} {'Pass':>6} {'Compl%':>8} {'Agent%':>8} {'Reason%':>8} {'Latency':>10}"
    )
    print("  " + "-" * 60)
    for mode, m in comparison.items():
        print(
            f"  {mode:<15} "
            f"{m['passed']:>3}/{m['total']:<3} "
            f"{m['task_completion_rate']:>7.1%} "
            f"{m['agent_accuracy']:>7.1%} "
            f"{m['reasoning_accuracy']:>7.1%} "
            f"{m['avg_latency_ms']:>8.0f}ms"
        )
    print(f"\n  Results: {output_base.resolve()}")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study: orchestration depth")
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODES,
        default=None,
        help="Run a single mode (default: all three)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/ablation",
        help="Output base directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 3 scenarios per mode",
    )
    args = parser.parse_args()

    modes = (args.mode,) if args.mode else MODES
    max_sc = 3 if args.dry_run else None

    run_ablation(
        modes=modes,
        output_base=Path(args.output),
        max_scenarios=max_sc,
    )
