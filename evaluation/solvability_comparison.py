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

    llm_agent: str = ""
    llm_score: float = 0.0
    llm_correct: bool = False
    llm_latency_ms: float = 0.0

    agreement: bool = False  # same agent chosen by tfidf+neural?
    lexical_gap: bool = False  # word mismatch case?


# ── McNemar's test ──────────────────────────────────────────────────


def _mcnemar_pair(
    correct_a: List[bool], correct_b: List[bool], name_a: str, name_b: str
) -> Dict[str, Any]:
    """Compute McNemar's test for a pair of estimators.

    Constructs a 2x2 contingency table:
        - both_correct: a correct AND b correct
        - a_only: a correct, b wrong
        - b_only: a wrong, b correct
        - both_wrong: both wrong

    McNemar statistic (with continuity correction):
        χ² = (|b_only - a_only| - 1)² / (a_only + b_only)
    """
    both_ok = a_only = b_only = both_bad = 0
    for ca, cb in zip(correct_a, correct_b):
        if ca and cb:
            both_ok += 1
        elif ca and not cb:
            a_only += 1
        elif not ca and cb:
            b_only += 1
        else:
            both_bad += 1

    discordant = a_only + b_only
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant
        p_value = _chi2_sf(chi2, df=1)

    return {
        "pair": f"{name_a}_vs_{name_b}",
        "contingency": {
            "both_correct": both_ok,
            f"{name_a}_only": a_only,
            f"{name_b}_only": b_only,
            "both_wrong": both_bad,
        },
        "discordant_pairs": discordant,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant_at_005": p_value < 0.05,
        "favours": (
            name_a if a_only > b_only else (name_b if b_only > a_only else "neither")
        ),
    }


def mcnemar_test(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Compute McNemar's test for TF-IDF vs Neural (backward compat)."""
    return _mcnemar_pair(
        [r.tfidf_correct for r in results],
        [r.neural_correct for r in results],
        "tfidf",
        "neural",
    )


def mcnemar_all_pairs(results: List[ComparisonResult]) -> Dict[str, Any]:
    """Compute pairwise McNemar tests for all estimator pairs.

    Returns dict with keys: tfidf_vs_neural, llm_vs_tfidf, llm_vs_neural.
    LLM pairs are only included if LLM results are present.
    """
    pairs = {
        "tfidf_vs_neural": _mcnemar_pair(
            [r.tfidf_correct for r in results],
            [r.neural_correct for r in results],
            "tfidf",
            "neural",
        ),
    }
    has_llm = any(r.llm_agent for r in results)
    if has_llm:
        pairs["llm_vs_tfidf"] = _mcnemar_pair(
            [r.llm_correct for r in results],
            [r.tfidf_correct for r in results],
            "llm",
            "tfidf",
        )
        pairs["llm_vs_neural"] = _mcnemar_pair(
            [r.llm_correct for r in results],
            [r.neural_correct for r in results],
            "llm",
            "neural",
        )
    return pairs


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
    """Compare TF-IDF, Neural, and optionally LLM solvability estimators."""

    def __init__(self, tfidf_estimator, neural_estimator, registry, llm_estimator=None):
        self.tfidf = tfidf_estimator
        self.neural = neural_estimator
        self.llm = llm_estimator
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

            # Run LLM estimator (optional)
            llm_agent, llm_score, llm_ms = "", 0.0, 0.0
            if self.llm is not None:
                t0 = time.perf_counter()
                llm_result = self.llm.estimate([subtask], agent_catalog)
                llm_ms = (time.perf_counter() - t0) * 1000
                llm_agent = llm_result.assignments.get(subtask, "")
                llm_score = llm_result.assignment_scores.get(subtask, 0.0)

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
                llm_agent=llm_agent,
                llm_score=llm_score,
                llm_correct=(llm_agent == correct_agent) if self.llm else False,
                llm_latency_ms=round(llm_ms, 2),
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
        llm_correct = sum(1 for r in results if r.llm_correct)
        agreements = sum(1 for r in results if r.agreement)
        has_llm = any(r.llm_agent for r in results)

        # Lexical gap subset
        lex_cases = [r for r in results if r.lexical_gap]
        std_cases = [r for r in results if not r.lexical_gap]

        # Latency
        avg_tfidf_ms = sum(r.tfidf_latency_ms for r in results) / total
        avg_neural_ms = sum(r.neural_latency_ms for r in results) / total
        avg_llm_ms = sum(r.llm_latency_ms for r in results) / total if has_llm else 0.0

        summary = {
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
            "mcnemar_all_pairs": mcnemar_all_pairs(results),
            "by_category": per_category_metrics(results),
            "confusion_tfidf": confusion_matrix(results, "tfidf"),
            "confusion_neural": confusion_matrix(results, "neural"),
        }
        if has_llm:
            summary["llm_accuracy"] = round(llm_correct / total, 4)
            summary["llm_correct"] = llm_correct
            summary["latency"]["avg_llm_ms"] = round(avg_llm_ms, 2)
            summary["standard_match"]["llm_accuracy"] = round(
                sum(1 for r in std_cases if r.llm_correct) / max(1, len(std_cases)), 4
            )
            summary["lexical_gap"]["llm_accuracy"] = round(
                sum(1 for r in lex_cases if r.llm_correct) / max(1, len(lex_cases)), 4
            )
        return summary

    def print_summary(self, results: List[ComparisonResult]) -> str:
        """Print and return a formatted comparison summary."""
        summary = self.compute_summary(results)
        if not summary:
            msg = "No results to compare."
            print(msg)
            return msg

        total = summary["total_scenarios"]
        has_llm = "llm_accuracy" in summary
        title = (
            "SOLVABILITY ESTIMATOR COMPARISON: TF-IDF vs Neural vs LLM"
            if has_llm
            else "SOLVABILITY ESTIMATOR COMPARISON: TF-IDF vs Neural"
        )
        lines = [
            "=" * 70,
            title,
            "=" * 70,
            f"Total subtasks evaluated: {total}",
            "",
            "OVERALL ACCURACY:",
            f"  TF-IDF:  {summary['tfidf_correct']}/{total} ({100*summary['tfidf_accuracy']:.1f}%)",
            f"  Neural:  {summary['neural_correct']}/{total} ({100*summary['neural_accuracy']:.1f}%)",
        ]
        if has_llm:
            lines.append(
                f"  LLM:     {summary['llm_correct']}/{total} ({100*summary['llm_accuracy']:.1f}%)"
            )
        lines.append(
            f"  Agreement (TF-IDF/Neural): {100*summary['agreement_rate']:.1f}%"
        )

        std = summary["standard_match"]
        lex = summary["lexical_gap"]
        std_llm = f"\n  LLM:    {100*std['llm_accuracy']:.1f}%" if has_llm else ""
        lex_llm = f"\n  LLM:    {100*lex['llm_accuracy']:.1f}%" if has_llm else ""
        lines += [
            "",
            f"STANDARD MATCH ({std['count']} cases):",
            f"  TF-IDF: {100*std['tfidf_accuracy']:.1f}%",
            f"  Neural: {100*std['neural_accuracy']:.1f}%" + std_llm,
            "",
            f"LEXICAL GAP ({lex['count']} cases):",
            f"  TF-IDF: {100*lex['tfidf_accuracy']:.1f}%",
            f"  Neural: {100*lex['neural_accuracy']:.1f}%" + lex_llm,
        ]

        lat = summary["latency"]
        lines += [
            "",
            "LATENCY:",
            f"  TF-IDF: {lat['avg_tfidf_ms']:.1f}ms",
            f"  Neural: {lat['avg_neural_ms']:.1f}ms",
            f"  Speedup: {lat['speedup_factor']:.0f}x (TF-IDF faster)",
        ]

        all_pairs = summary.get("mcnemar_all_pairs", {})
        lines += ["", "McNEMAR'S PAIRWISE TESTS:"]
        for pair_key, mc in all_pairs.items():
            ct = mc["contingency"]
            lines += [
                f"  {mc['pair']}:",
                f"    Contingency: {ct}",
                f"    chi2={mc['chi2']:.4f}  p={mc['p_value']:.4f}  "
                f"significant={mc['significant_at_005']}  favours={mc['favours']}",
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
