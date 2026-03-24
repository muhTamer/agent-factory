# evaluation/run_solvability_comparison.py
"""
RQ1 Solvability Estimator Comparison: TF-IDF vs Neural

Runs 45 ground-truth subtask scenarios through both estimators and reports:
  - Overall accuracy (standard match + lexical gap)
  - Per-category breakdown
  - McNemar's test for statistical significance
  - Latency comparison
  - Confusion matrices

Usage:
    python -m evaluation.run_solvability_comparison
    python -m evaluation.run_solvability_comparison --detailed
    python -m evaluation.run_solvability_comparison --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SCENARIOS_PATH = Path("evaluation/scenarios/solvability_scenarios.json")
OUTPUT_DIR = Path("evaluation/results/solvability")


def run_solvability_comparison(
    output_dir: Path = OUTPUT_DIR,
    max_scenarios: int | None = None,
    detailed: bool = False,
) -> dict:
    """Execute the TF-IDF vs Neural solvability comparison."""
    from app.orchestration.neural_solvability_estimator import (
        NeuralSolvabilityEstimator,
    )
    from app.orchestration.solvability_estimator import SolvabilityEstimator
    from evaluation.solvability_comparison import SolvabilityComparison
    from scripts._bootstrap import bootstrap_registry

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
        print(f"Dry-run mode: running {len(scenarios)} scenarios only")

    # Bootstrap
    registry, store = bootstrap_registry()

    tfidf = SolvabilityEstimator(store)
    neural = NeuralSolvabilityEstimator(store, model_path=Path("models/reward_mlp.pt"))

    print(f"Neural MLP trained: {neural.is_trained}")
    print(f"Agents in registry: {registry.all_ids()}")
    print(f"Scenarios: {len(scenarios)}")
    print()

    comparison = SolvabilityComparison(
        tfidf_estimator=tfidf,
        neural_estimator=neural,
        registry=registry,
    )

    # Run comparison
    t0 = time.time()
    results = comparison.compare_on_scenarios(scenarios)
    wall_time = time.time() - t0

    # Print results
    if detailed:
        comparison.print_detailed(results)

    comparison.print_summary(results)

    # Compute summary for export
    summary = comparison.compute_summary(results)
    summary["wall_time_seconds"] = round(wall_time, 2)
    summary["neural_mlp_trained"] = neural.is_trained

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-scenario results
    per_scenario = []
    for r in results:
        per_scenario.append(
            {
                "scenario_id": r.scenario_id,
                "subtask": r.subtask,
                "correct_agent": r.correct_agent,
                "category": r.category,
                "tfidf_agent": r.tfidf_agent,
                "tfidf_score": round(r.tfidf_score, 4),
                "tfidf_correct": r.tfidf_correct,
                "tfidf_latency_ms": r.tfidf_latency_ms,
                "neural_agent": r.neural_agent,
                "neural_score": round(r.neural_score, 4),
                "neural_correct": r.neural_correct,
                "neural_latency_ms": r.neural_latency_ms,
                "agreement": r.agreement,
                "lexical_gap": r.lexical_gap,
            }
        )

    results_path = output_dir / "solvability_comparison_results.json"
    results_path.write_text(
        json.dumps(per_scenario, indent=2, default=str), encoding="utf-8"
    )

    summary_path = output_dir / "solvability_comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nResults saved to: {output_dir.resolve()}")
    return summary


# ── pytest integration ──────────────────────────────────────────────


def test_solvability_comparison_runs():
    """pytest entry point: run comparison and verify no crashes."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="solv_cmp_"))
    summary = run_solvability_comparison(output_dir, max_scenarios=5)

    assert "total_scenarios" in summary
    assert summary["total_scenarios"] == 5
    assert "mcnemar" in summary
    assert "by_category" in summary


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ1 Solvability Comparison — TF-IDF vs Neural"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print per-scenario details",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 5 scenarios for verification",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SOLVABILITY ESTIMATOR COMPARISON: TF-IDF vs Neural")
    print("=" * 70 + "\n")

    max_sc = 5 if args.dry_run else None
    run_solvability_comparison(
        Path(args.output),
        max_scenarios=max_sc,
        detailed=args.detailed,
    )
