#!/usr/bin/env python3
"""
Run side-by-side comparison of TF-IDF vs Neural solvability estimators.

Usage:
    python scripts/run_comparison.py [--scenarios SCENARIOS_FILE] [--model MODEL_PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.orchestration.neural_solvability_estimator import NeuralSolvabilityEstimator
from app.orchestration.solvability_estimator import SolvabilityEstimator
from evaluation.solvability_comparison import SolvabilityComparison
from scripts._bootstrap import bootstrap_registry

# Default evaluation scenarios with ground truth.
# These cover standard matches and lexical-gap cases.
DEFAULT_SCENARIOS = [
    # Standard matches (keywords overlap with agent descriptions)
    {
        "subtask": "INFORMATIONAL: refund policy details",
        "correct_agent": "agent_faq",
        "lexical_gap": False,
    },
    {
        "subtask": "ACTION: process refund for order #12345",
        "correct_agent": "agent_refunds",
        "lexical_gap": False,
    },
    {
        "subtask": "INFORMATIONAL: account opening requirements",
        "correct_agent": "agent_faq",
        "lexical_gap": False,
    },
    # Lexical gap cases (semantically correct but words don't match)
    {
        "subtask": "INFORMATIONAL: how to get my money back",
        "correct_agent": "agent_faq",
        "lexical_gap": True,
    },
    {
        "subtask": "ACTION: reverse the charge on my card",
        "correct_agent": "agent_refunds",
        "lexical_gap": True,
    },
    {
        "subtask": "INFORMATIONAL: what happens if I'm unhappy with my purchase",
        "correct_agent": "agent_faq",
        "lexical_gap": True,
    },
    {
        "subtask": "ACTION: give me back what I paid for order #999",
        "correct_agent": "agent_refunds",
        "lexical_gap": True,
    },
    {
        "subtask": "INFORMATIONAL: steps to terminate my membership",
        "correct_agent": "agent_faq",
        "lexical_gap": True,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Compare TF-IDF vs Neural")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="JSON file with evaluation scenarios (uses defaults if omitted)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/reward_mlp.pt"),
        help="Path to trained neural model weights",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print per-scenario details",
    )
    args = parser.parse_args()

    # Load scenarios
    if args.scenarios and args.scenarios.exists():
        scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
        print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}")
    else:
        scenarios = DEFAULT_SCENARIOS
        print(f"Using {len(scenarios)} built-in default scenarios")

    # Initialise components
    registry, store = bootstrap_registry()

    tfidf = SolvabilityEstimator(store)
    neural = NeuralSolvabilityEstimator(store, model_path=args.model)

    comparison = SolvabilityComparison(
        tfidf_estimator=tfidf,
        neural_estimator=neural,
        registry=registry,
    )

    # Run comparison
    results = comparison.compare_on_scenarios(scenarios)

    # Print results
    if args.detailed:
        comparison.print_detailed(results)

    comparison.print_summary(results)

    # Save results
    output_path = Path("evaluation/comparison_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [
                {
                    "scenario_id": r.scenario_id,
                    "subtask": r.subtask,
                    "correct_agent": r.correct_agent,
                    "tfidf_agent": r.tfidf_agent,
                    "tfidf_score": r.tfidf_score,
                    "tfidf_correct": r.tfidf_correct,
                    "tfidf_latency_ms": r.tfidf_latency_ms,
                    "neural_agent": r.neural_agent,
                    "neural_score": r.neural_score,
                    "neural_correct": r.neural_correct,
                    "neural_latency_ms": r.neural_latency_ms,
                    "agreement": r.agreement,
                    "lexical_gap": r.lexical_gap,
                }
                for r in results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
