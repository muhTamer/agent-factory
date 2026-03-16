# evaluation/solvability_comparison.py
"""
Side-by-side comparison of TF-IDF vs Neural solvability estimators.

Metrics:
  1. Accuracy — % correct agent assignments vs ground truth
  2. Agreement rate — how often both choose the same agent
  3. Lexical gap performance — accuracy on word-mismatch cases
  4. Latency — execution time comparison
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ComparisonResult:
    """Result for a single (subtask, correct_agent) scenario."""

    scenario_id: str
    subtask: str
    correct_agent: str  # ground truth

    tfidf_agent: str
    tfidf_score: float
    tfidf_correct: bool
    tfidf_latency_ms: float

    neural_agent: str
    neural_score: float
    neural_correct: bool
    neural_latency_ms: float

    agreement: bool  # same agent chosen?
    lexical_gap: bool  # word mismatch case?


class SolvabilityComparison:
    """Compare TF-IDF and Neural solvability estimators."""

    def __init__(self, tfidf_estimator, neural_estimator, registry):
        """
        Args:
            tfidf_estimator: SolvabilityEstimator instance.
            neural_estimator: NeuralSolvabilityEstimator instance.
            registry: AgentRegistry for building agent catalog.
        """
        self.tfidf = tfidf_estimator
        self.neural = neural_estimator
        self.registry = registry

    def compare_on_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> List[ComparisonResult]:
        """
        Run both estimators on a list of scenarios and compare.

        Each scenario dict must contain:
          - "subtask": str — the subtask text
          - "correct_agent": str — ground truth agent_id
          - "lexical_gap": bool (optional) — whether this is a word-mismatch case
          - "scenario_id": str (optional) — identifier
        """
        agent_catalog = self.registry.all_meta()
        results: List[ComparisonResult] = []

        for i, scenario in enumerate(scenarios):
            subtask = scenario["subtask"]
            correct_agent = scenario["correct_agent"]
            is_lexical_gap = scenario.get("lexical_gap", False)
            scenario_id = scenario.get("scenario_id", f"S{i+1:03d}")

            # Run TF-IDF estimator
            t0 = time.perf_counter()
            tfidf_result = self.tfidf.estimate([subtask], agent_catalog)
            tfidf_ms = (time.perf_counter() - t0) * 1000

            tfidf_agent = tfidf_result.assignments.get(subtask, "")
            tfidf_score = tfidf_result.assignment_scores.get(subtask, 0.0)

            # Run Neural estimator
            t0 = time.perf_counter()
            neural_result = self.neural.estimate([subtask], agent_catalog)
            neural_ms = (time.perf_counter() - t0) * 1000

            neural_agent = neural_result.assignments.get(subtask, "")
            neural_score = neural_result.assignment_scores.get(subtask, 0.0)

            result = ComparisonResult(
                scenario_id=scenario_id,
                subtask=subtask,
                correct_agent=correct_agent,
                tfidf_agent=tfidf_agent,
                tfidf_score=tfidf_score,
                tfidf_correct=(tfidf_agent == correct_agent),
                tfidf_latency_ms=round(tfidf_ms, 2),
                neural_agent=neural_agent,
                neural_score=neural_score,
                neural_correct=(neural_agent == correct_agent),
                neural_latency_ms=round(neural_ms, 2),
                agreement=(tfidf_agent == neural_agent),
                lexical_gap=is_lexical_gap,
            )
            results.append(result)

        return results

    def print_summary(self, results: List[ComparisonResult]) -> str:
        """Print and return a formatted comparison summary."""
        total = len(results)
        if total == 0:
            msg = "No results to compare."
            print(msg)
            return msg

        tfidf_correct = sum(1 for r in results if r.tfidf_correct)
        neural_correct = sum(1 for r in results if r.neural_correct)
        agreements = sum(1 for r in results if r.agreement)

        lexical_cases = [r for r in results if r.lexical_gap]
        lex_total = len(lexical_cases)
        lex_tfidf = sum(1 for r in lexical_cases if r.tfidf_correct)
        lex_neural = sum(1 for r in lexical_cases if r.neural_correct)

        avg_tfidf_ms = sum(r.tfidf_latency_ms for r in results) / total
        avg_neural_ms = sum(r.neural_latency_ms for r in results) / total
        speedup = avg_neural_ms / max(0.001, avg_tfidf_ms)

        lines = [
            "=" * 60,
            "SOLVABILITY ESTIMATOR COMPARISON",
            "=" * 60,
            f"Total subtasks evaluated: {total}",
            "",
            f"TF-IDF Accuracy:   {tfidf_correct}/{total} ({100*tfidf_correct/total:.1f}%)",
            f"Neural Accuracy:   {neural_correct}/{total} ({100*neural_correct/total:.1f}%)",
            f"Agreement Rate:    {agreements}/{total} ({100*agreements/total:.1f}%)",
        ]

        if lex_total > 0:
            lines += [
                "",
                f"Lexical Gap Cases: {lex_total}",
                f"  TF-IDF on lexical: {lex_tfidf}/{lex_total} ({100*lex_tfidf/lex_total:.1f}%)",
                f"  Neural on lexical: {lex_neural}/{lex_total} ({100*lex_neural/lex_total:.1f}%)",
            ]

        lines += [
            "",
            "Avg Latency:",
            f"  TF-IDF: {avg_tfidf_ms:.1f}ms",
            f"  Neural: {avg_neural_ms:.1f}ms",
            f"  Speedup: {speedup:.0f}x faster (TF-IDF)",
            "=" * 60,
        ]

        summary = "\n".join(lines)
        print(summary)
        return summary

    def print_detailed(self, results: List[ComparisonResult]) -> None:
        """Print per-scenario details."""
        for r in results:
            marker = "✓" if r.neural_correct else "✗"
            gap_tag = " [LEX-GAP]" if r.lexical_gap else ""
            print(
                f"{r.scenario_id}: {r.subtask[:50]}...{gap_tag}\n"
                f"  Ground truth: {r.correct_agent}\n"
                f"  TF-IDF: {r.tfidf_agent} ({r.tfidf_score:.3f}) "
                f"{'✓' if r.tfidf_correct else '✗'}\n"
                f"  Neural: {r.neural_agent} ({r.neural_score:.3f}) {marker}\n"
            )
