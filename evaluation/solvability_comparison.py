# evaluation/solvability_comparison.py
"""
Side-by-side comparison of TF-IDF vs Neural solvability estimators.

Metrics:
  1. Accuracy — % correct agent assignments vs ground truth
  2. Agreement rate — how often both choose the same agent
  3. Lexical gap performance — accuracy on word-mismatch cases
  4. Latency — execution time comparison
  5. McNemar's test — statistical significance of accuracy difference
  6. Per-category breakdown — accuracy by scenario category
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ComparisonResult:
    """Result for a single (subtask, correct_agent) scenario."""

    scenario_id: str
    subtask: str
    correct_agent: str  # ground truth
    category: str  # scenario category

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


# ── McNemar's test ──────────────────────────────────────────────────


def mcnemar_test(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Compute McNemar's test for paired binary classification.

    Constructs a 2x2 contingency table:
        - a: both correct
        - b: TF-IDF correct, Neural wrong
        - c: TF-IDF wrong, Neural correct
        - d: both wrong

    McNemar statistic (with continuity correction):
        χ² = (|b - c| - 1)² / (b + c)

    Returns dict with contingency table, chi2, p-value, and interpretation.
    """
    a = b = c = d = 0
    for r in results:
        if r.tfidf_correct and r.neural_correct:
            a += 1
        elif r.tfidf_correct and not r.neural_correct:
            b += 1
        elif not r.tfidf_correct and r.neural_correct:
            c += 1
        else:
            d += 1

    discordant = b + c
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        # McNemar's with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / discordant
        # Approximate p-value from chi-squared distribution (1 df)
        p_value = _chi2_sf(chi2, df=1)

    return {
        "contingency": {
            "both_correct": a,
            "tfidf_only": b,
            "neural_only": c,
            "both_wrong": d,
        },
        "discordant_pairs": discordant,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant_at_005": p_value < 0.05,
        "favours": "tfidf" if b > c else ("neural" if c > b else "neither"),
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function approximation.
    For df=1: P(X > x) = erfc(sqrt(x/2)) where erfc is complementary error function.
    """
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    # Fallback for other df (not needed for McNemar's)
    return math.erfc(math.sqrt(x / 2))


# ── Per-category metrics ────────────────────────────────────────────


def per_category_metrics(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Compute accuracy breakdown by scenario category."""
    categories: Dict[str, List[ComparisonResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    breakdown: Dict[str, Any] = {}
    for cat in sorted(categories):
        cat_results = categories[cat]
        n = len(cat_results)
        tfidf_ok = sum(1 for r in cat_results if r.tfidf_correct)
        neural_ok = sum(1 for r in cat_results if r.neural_correct)
        breakdown[cat] = {
            "count": n,
            "tfidf_correct": tfidf_ok,
            "tfidf_accuracy": round(tfidf_ok / n, 4) if n else 0.0,
            "neural_correct": neural_ok,
            "neural_accuracy": round(neural_ok / n, 4) if n else 0.0,
        }
    return breakdown


# ── Confusion matrices ──────────────────────────────────────────────


def confusion_matrix(
    results: List[ComparisonResult], estimator: str = "tfidf"
) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix for one estimator.

    Returns {actual_agent: {predicted_agent: count}}.
    """
    agents = sorted(set(r.correct_agent for r in results))
    matrix: Dict[str, Dict[str, int]] = {a: {b: 0 for b in agents} for a in agents}
    for r in results:
        predicted = r.tfidf_agent if estimator == "tfidf" else r.neural_agent
        if r.correct_agent in matrix and predicted in matrix[r.correct_agent]:
            matrix[r.correct_agent][predicted] += 1
    return matrix


# ── Comparison class ────────────────────────────────────────────────


class SolvabilityComparison:
    """Compare TF-IDF and Neural solvability estimators."""

    def __init__(self, tfidf_estimator, neural_estimator, registry):
        self.tfidf = tfidf_estimator
        self.neural = neural_estimator
        self.registry = registry

    def compare_on_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
    ) -> List[ComparisonResult]:
        """Run both estimators on a list of scenarios and compare."""
        agent_catalog = self.registry.all_meta()
        results: List[ComparisonResult] = []

        for i, scenario in enumerate(scenarios):
            subtask = scenario["subtask"]
            correct_agent = scenario["correct_agent"]
            is_lexical_gap = scenario.get("lexical_gap", False)
            scenario_id = scenario.get("scenario_id", f"S{i+1:03d}")
            category = scenario.get("category", "unknown")

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
                category=category,
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

    def compute_summary(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Compute full summary metrics including McNemar's test."""
        total = len(results)
        if total == 0:
            return {}

        tfidf_correct = sum(1 for r in results if r.tfidf_correct)
        neural_correct = sum(1 for r in results if r.neural_correct)
        agreements = sum(1 for r in results if r.agreement)

        # Lexical gap subset
        lex_cases = [r for r in results if r.lexical_gap]
        std_cases = [r for r in results if not r.lexical_gap]

        # Latency
        avg_tfidf_ms = sum(r.tfidf_latency_ms for r in results) / total
        avg_neural_ms = sum(r.neural_latency_ms for r in results) / total

        return {
            "total_scenarios": total,
            "tfidf_accuracy": round(tfidf_correct / total, 4),
            "neural_accuracy": round(neural_correct / total, 4),
            "tfidf_correct": tfidf_correct,
            "neural_correct": neural_correct,
            "agreement_rate": round(agreements / total, 4),
            "standard_match": {
                "count": len(std_cases),
                "tfidf_accuracy": round(
                    sum(1 for r in std_cases if r.tfidf_correct)
                    / max(1, len(std_cases)),
                    4,
                ),
                "neural_accuracy": round(
                    sum(1 for r in std_cases if r.neural_correct)
                    / max(1, len(std_cases)),
                    4,
                ),
            },
            "lexical_gap": {
                "count": len(lex_cases),
                "tfidf_accuracy": round(
                    sum(1 for r in lex_cases if r.tfidf_correct)
                    / max(1, len(lex_cases)),
                    4,
                ),
                "neural_accuracy": round(
                    sum(1 for r in lex_cases if r.neural_correct)
                    / max(1, len(lex_cases)),
                    4,
                ),
            },
            "latency": {
                "avg_tfidf_ms": round(avg_tfidf_ms, 2),
                "avg_neural_ms": round(avg_neural_ms, 2),
                "speedup_factor": round(avg_neural_ms / max(0.001, avg_tfidf_ms), 1),
            },
            "mcnemar": mcnemar_test(results),
            "by_category": per_category_metrics(results),
            "confusion_tfidf": confusion_matrix(results, "tfidf"),
            "confusion_neural": confusion_matrix(results, "neural"),
        }

    def print_summary(self, results: List[ComparisonResult]) -> str:
        """Print and return a formatted comparison summary."""
        summary = self.compute_summary(results)
        if not summary:
            msg = "No results to compare."
            print(msg)
            return msg

        total = summary["total_scenarios"]
        lines = [
            "=" * 70,
            "SOLVABILITY ESTIMATOR COMPARISON: TF-IDF vs Neural",
            "=" * 70,
            f"Total subtasks evaluated: {total}",
            "",
            "OVERALL ACCURACY:",
            f"  TF-IDF:  {summary['tfidf_correct']}/{total} ({100*summary['tfidf_accuracy']:.1f}%)",
            f"  Neural:  {summary['neural_correct']}/{total} ({100*summary['neural_accuracy']:.1f}%)",
            f"  Agreement: {100*summary['agreement_rate']:.1f}%",
        ]

        std = summary["standard_match"]
        lex = summary["lexical_gap"]
        lines += [
            "",
            f"STANDARD MATCH ({std['count']} cases):",
            f"  TF-IDF: {100*std['tfidf_accuracy']:.1f}%",
            f"  Neural: {100*std['neural_accuracy']:.1f}%",
            "",
            f"LEXICAL GAP ({lex['count']} cases):",
            f"  TF-IDF: {100*lex['tfidf_accuracy']:.1f}%",
            f"  Neural: {100*lex['neural_accuracy']:.1f}%",
        ]

        lat = summary["latency"]
        lines += [
            "",
            "LATENCY:",
            f"  TF-IDF: {lat['avg_tfidf_ms']:.1f}ms",
            f"  Neural: {lat['avg_neural_ms']:.1f}ms",
            f"  Speedup: {lat['speedup_factor']:.0f}x (TF-IDF faster)",
        ]

        mc = summary["mcnemar"]
        ct = mc["contingency"]
        lines += [
            "",
            "McNEMAR'S TEST (paired comparison):",
            f"  Both correct: {ct['both_correct']}  |  TF-IDF only: {ct['tfidf_only']}",
            f"  Neural only:  {ct['neural_only']}  |  Both wrong:   {ct['both_wrong']}",
            f"  chi2={mc['chi2']:.4f}  p={mc['p_value']:.4f}  significant={mc['significant_at_005']}",
            f"  Favours: {mc['favours']}",
        ]

        by_cat = summary["by_category"]
        if by_cat:
            lines += ["", "PER-CATEGORY ACCURACY:"]
            for cat, data in by_cat.items():
                lines.append(
                    f"  {cat:25s}  n={data['count']:2d}  "
                    f"TF-IDF={100*data['tfidf_accuracy']:5.1f}%  "
                    f"Neural={100*data['neural_accuracy']:5.1f}%"
                )

        lines.append("=" * 70)

        text = "\n".join(lines)
        print(text)
        return text

    def print_detailed(self, results: List[ComparisonResult]) -> None:
        """Print per-scenario details."""
        for r in results:
            gap_tag = " [LEX-GAP]" if r.lexical_gap else ""
            print(
                f"{r.scenario_id}: {r.subtask[:50]}...{gap_tag}\n"
                f"  Ground truth: {r.correct_agent}\n"
                f"  TF-IDF: {r.tfidf_agent} ({r.tfidf_score:.3f}) "
                f"{'PASS' if r.tfidf_correct else 'FAIL'}\n"
                f"  Neural: {r.neural_agent} ({r.neural_score:.3f}) "
                f"{'PASS' if r.neural_correct else 'FAIL'}\n"
            )
