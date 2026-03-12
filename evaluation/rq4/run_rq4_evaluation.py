# evaluation/rq4/run_rq4_evaluation.py
"""
RQ4 Evaluation Runner — Perceived Quality in Customer Interactions

Usage:
    python -m evaluation.rq4.run_rq4_evaluation                          # mock mode (default)
    python -m evaluation.rq4.run_rq4_evaluation --real                   # real LLM judge
    python -m evaluation.rq4.run_rq4_evaluation --output results/rq4/    # custom output dir
    pytest evaluation/rq4/run_rq4_evaluation.py::test_rq4_mock -v        # as pytest

Outputs:
    rq4_results.csv          — one row per evaluation (scenario x strategy x persona)
    rq4_results.json         — full results with justifications
    rq4_metrics.json         — aggregate metrics (by_strategy, by_persona, matrix)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# ── Project root on sys.path ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.rq4.harness import RQ4Harness  # noqa: E402
from evaluation.rq4.judge import LLMJudge, MockJudge  # noqa: E402
from evaluation.rq4.personas import ALL_PERSONAS  # noqa: E402
from evaluation.rq4.strategies import ALL_STRATEGIES  # noqa: E402

# ── Scenarios path ────────────────────────────────────────────────────
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "rq4_scenarios.json"


# ── Load scenarios ────────────────────────────────────────────────────


def _load_scenarios() -> list:
    return json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))


# ── Run evaluation ────────────────────────────────────────────────────


def run_rq4_evaluation(
    output_dir: Path,
    use_real_judge: bool = False,
    scenario_filter: str | None = None,
) -> Dict[str, Any]:
    """Execute the full RQ4 evaluation and write results."""

    # Select judge mode
    if use_real_judge:
        judge = LLMJudge()
        print("  Judge mode: REAL LLM (API calls)")
    else:
        judge = MockJudge()
        print("  Judge mode: MOCK (deterministic)")

    # Load scenarios
    scenarios = _load_scenarios()
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        if not scenarios:
            print(f"[ERROR] No scenario with id={scenario_filter}")
            return {}

    print(f"  Scenarios: {len(scenarios)}")
    print(
        f"  Strategies: {len(ALL_STRATEGIES)} ({', '.join(s.slug for s in ALL_STRATEGIES)})"
    )
    print(
        f"  Personas: {len(ALL_PERSONAS)} ({', '.join(p.slug for p in ALL_PERSONAS)})"
    )
    print(
        f"  Total evaluations: "
        f"{len(scenarios)} x {len(ALL_STRATEGIES)} x {len(ALL_PERSONAS)} = "
        f"{len(scenarios) * len(ALL_STRATEGIES) * len(ALL_PERSONAS)}"
    )
    print()

    # Run harness
    harness = RQ4Harness(
        scenarios=scenarios,
        judge=judge,
        personas=ALL_PERSONAS,
        strategies=ALL_STRATEGIES,
    )

    print("Running evaluations...")
    result = harness.run_all()

    # Print progress summary per scenario
    current_scenario = None
    for jr in result.judge_results:
        if jr.scenario_id != current_scenario:
            current_scenario = jr.scenario_id
            # Count results for this scenario
            sc_results = [
                r for r in result.judge_results if r.scenario_id == current_scenario
            ]
            mean_sat = (
                sum(r.satisfaction for r in sc_results) / len(sc_results)
                if sc_results
                else 0
            )
            print(
                f"  [{current_scenario:12s}]  "
                f"evals={len(sc_results):3d}  "
                f"mean_satisfaction={mean_sat:.1f}"
            )

    # Compute metrics
    metrics = harness.compute_metrics(result)

    # Print summary
    print("\n" + "=" * 70)
    print("RQ4 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total evaluations:  {metrics.get('total_evaluations', 0)}")

    overall = metrics.get("overall", {})
    for dim in ("transparency", "trust", "satisfaction"):
        stats = overall.get(dim, {})
        print(
            f"  Overall {dim:14s}: "
            f"mean={stats.get('mean', 0):.2f}  "
            f"std={stats.get('std', 0):.2f}"
        )

    print("\n  By Strategy:")
    for slug, stats in metrics.get("by_strategy", {}).items():
        t = stats.get("transparency", {}).get("mean", 0)
        tr = stats.get("trust", {}).get("mean", 0)
        s = stats.get("satisfaction", {}).get("mean", 0)
        print(f"    {slug:14s}  T={t:.2f}  Tr={tr:.2f}  S={s:.2f}")

    print("\n  By Persona:")
    for name, stats in metrics.get("by_persona", {}).items():
        t = stats.get("transparency", {}).get("mean", 0)
        tr = stats.get("trust", {}).get("mean", 0)
        s = stats.get("satisfaction", {}).get("mean", 0)
        print(f"    {name:22s}  T={t:.2f}  Tr={tr:.2f}  S={s:.2f}")

    stat_tests = metrics.get("statistical_tests", {})
    if stat_tests and "note" not in stat_tests:
        print("\n  Statistical Tests (Kruskal-Wallis):")
        for dim, test in stat_tests.items():
            if isinstance(test, dict) and "p_value" in test:
                sig = "***" if test["significant"] else "n.s."
                print(
                    f"    {dim:14s}: H={test['H_statistic']:.2f}  "
                    f"p={test['p_value']:.4f}  {sig}"
                )

    # Export results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness.export_csv(result, output_dir / "rq4_results.csv")
    harness.export_json(result, output_dir / "rq4_results.json")

    metrics_path = output_dir / "rq4_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── pytest integration ────────────────────────────────────────────────


def test_rq4_mock():
    """pytest entry point: run all RQ4 evaluations in mock mode."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="rq4_eval_"))
    metrics = run_rq4_evaluation(output_dir, use_real_judge=False)

    # Verify structure
    assert (
        metrics["total_evaluations"] == 560
    ), f"Expected 560 evaluations (20x4x7), got {metrics['total_evaluations']}"

    # Verify all strategies present
    assert (
        len(metrics["by_strategy"]) == 4
    ), f"Expected 4 strategies, got {len(metrics['by_strategy'])}"

    # Verify all personas present
    assert (
        len(metrics["by_persona"]) == 7
    ), f"Expected 7 personas, got {len(metrics['by_persona'])}"

    # Verify score ranges (all scores should be 1-5)
    for strat_stats in metrics["by_strategy"].values():
        for dim in ("transparency", "trust", "satisfaction"):
            mean = strat_stats[dim]["mean"]
            assert 1.0 <= mean <= 5.0, f"Mean {dim} out of range: {mean}"

    # Verify CSV was written
    csv_path = output_dir / "rq4_results.csv"
    assert csv_path.exists(), "CSV results file not created"

    # Verify JSON was written
    json_path = output_dir / "rq4_results.json"
    assert json_path.exists(), "JSON results file not created"

    # Verify metrics JSON was written
    metrics_path = output_dir / "rq4_metrics.json"
    assert metrics_path.exists(), "Metrics JSON file not created"


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RQ4 evaluation: perceived quality in customer interactions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/rq4/results",
        help="Output directory for results (default: evaluation/rq4/results)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        default=False,
        help="Use real LLM judge (requires API key; default: mock mode)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by ID",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RQ4: PERCEIVED QUALITY IN CUSTOMER INTERACTIONS")
    print("Agent Factory — LLM-as-Judge Persona Evaluation")
    print("=" * 70 + "\n")

    run_rq4_evaluation(
        Path(args.output),
        use_real_judge=args.real,
        scenario_filter=args.scenario,
    )
