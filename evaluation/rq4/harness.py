# evaluation/rq4/harness.py
"""
RQ4 Evaluation Harness — Perceived Quality in Customer Interactions

Orchestrates the full RQ4 evaluation loop:

  For each scenario (20):
    For each strategy (4):
      1. Retrieve strategy-specific response text
      2. For each persona (7):
        a. LLM judge evaluates response from persona perspective
        b. Collect TTS scores (transparency, trust, satisfaction)

Total: 20 x 4 x 7 = 560 evaluations.

Computes aggregate metrics and exports results for thesis tables.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.rq4.judge import Judge, JudgeResult
from evaluation.rq4.personas import ALL_PERSONAS, Persona
from evaluation.rq4.strategies import ALL_STRATEGIES, Strategy

# ── Result dataclasses ────────────────────────────────────────────────


@dataclass
class RQ4EvaluationResult:
    """Complete RQ4 evaluation result set."""

    judge_results: List[JudgeResult] = field(default_factory=list)
    total_evaluations: int = 0
    scenarios_count: int = 0
    strategies_count: int = 0
    personas_count: int = 0


# ── Harness ───────────────────────────────────────────────────────────


class RQ4Harness:
    """Runs RQ4 persona-based evaluation across scenarios, strategies, and personas."""

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        judge: Judge,
        personas: Optional[List[Persona]] = None,
        strategies: Optional[List[Strategy]] = None,
    ):
        self.scenarios = scenarios
        self.judge = judge
        self.personas = personas or ALL_PERSONAS
        self.strategies = strategies or ALL_STRATEGIES

    # ── Execution ─────────────────────────────────────────────────────

    def run_all(self) -> RQ4EvaluationResult:
        """Run the full evaluation matrix: scenarios x strategies x personas."""

        total_expected = len(self.scenarios) * len(self.strategies) * len(self.personas)
        result = RQ4EvaluationResult(
            scenarios_count=len(self.scenarios),
            strategies_count=len(self.strategies),
            personas_count=len(self.personas),
        )

        for scenario in self.scenarios:
            sc_id = scenario["id"]
            description = scenario.get("description", "")
            query = scenario["turns"][0]["query"]
            strategy_responses = scenario.get("strategy_responses", {})

            for strategy in self.strategies:
                response_text = strategy_responses.get(
                    strategy.slug,
                    f"[No response configured for strategy '{strategy.slug}']",
                )

                for persona in self.personas:
                    try:
                        judge_result = self.judge.evaluate(
                            persona=persona,
                            strategy=strategy,
                            scenario_id=sc_id,
                            scenario_description=description,
                            query=query,
                            response_text=response_text,
                        )
                    except Exception as exc:
                        # Log error but continue with remaining evaluations
                        print(
                            f"  [ERROR] {sc_id}/{strategy.slug}/{persona.slug}: {exc}",
                            flush=True,
                        )
                        from evaluation.rq4.judge import JudgeResult

                        judge_result = JudgeResult(
                            scenario_id=sc_id,
                            strategy_name=strategy.slug,
                            persona_name=persona.name,
                            transparency=3,
                            trust=3,
                            satisfaction=3,
                            justification=f"[ERROR] Judge call failed: {exc}",
                        )

                    result.judge_results.append(judge_result)
                    result.total_evaluations += 1

                    # Progress indicator
                    if result.total_evaluations % 7 == 0:
                        print(
                            f"  [{result.total_evaluations:3d}/{total_expected}] "
                            f"{sc_id} / {strategy.slug}",
                            flush=True,
                        )

        return result

    # ── Metrics ───────────────────────────────────────────────────────

    def compute_metrics(self, result: RQ4EvaluationResult) -> Dict[str, Any]:
        """Compute aggregate RQ4 metrics from evaluation results."""
        results = result.judge_results
        if not results:
            return {}

        metrics: Dict[str, Any] = {
            "total_evaluations": result.total_evaluations,
            "scenarios_count": result.scenarios_count,
            "strategies_count": result.strategies_count,
            "personas_count": result.personas_count,
        }

        # Overall means
        metrics["overall"] = self._compute_tts_stats(results)

        # By strategy
        metrics["by_strategy"] = {}
        for strategy in self.strategies:
            strat_results = [r for r in results if r.strategy_name == strategy.slug]
            if strat_results:
                metrics["by_strategy"][strategy.slug] = self._compute_tts_stats(
                    strat_results
                )

        # By persona
        metrics["by_persona"] = {}
        for persona in self.personas:
            persona_results = [r for r in results if r.persona_name == persona.name]
            if persona_results:
                metrics["by_persona"][persona.name] = self._compute_tts_stats(
                    persona_results
                )

        # By scenario category
        metrics["by_category"] = {}
        category_map = self._build_category_map()
        for category, sc_ids in category_map.items():
            cat_results = [r for r in results if r.scenario_id in sc_ids]
            if cat_results:
                metrics["by_category"][category] = self._compute_tts_stats(cat_results)

        # Strategy x Persona interaction matrix
        metrics["strategy_persona_matrix"] = {}
        for strategy in self.strategies:
            metrics["strategy_persona_matrix"][strategy.slug] = {}
            for persona in self.personas:
                cell_results = [
                    r
                    for r in results
                    if r.strategy_name == strategy.slug
                    and r.persona_name == persona.name
                ]
                if cell_results:
                    metrics["strategy_persona_matrix"][strategy.slug][persona.name] = (
                        self._compute_tts_stats(cell_results)
                    )

        # Kruskal-Wallis test for strategy differences (if scipy available)
        metrics["statistical_tests"] = self._compute_kruskal_wallis(results)

        return metrics

    # ── Export ─────────────────────────────────────────────────────────

    def export_csv(self, result: RQ4EvaluationResult, path: str | Path) -> None:
        """Write one row per judge evaluation to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "scenario_id",
            "strategy",
            "persona",
            "transparency",
            "trust",
            "satisfaction",
            "justification",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in result.judge_results:
                writer.writerow(
                    {
                        "scenario_id": r.scenario_id,
                        "strategy": r.strategy_name,
                        "persona": r.persona_name,
                        "transparency": r.transparency,
                        "trust": r.trust,
                        "satisfaction": r.satisfaction,
                        "justification": r.justification,
                    }
                )

    def export_json(self, result: RQ4EvaluationResult, path: str | Path) -> None:
        """Write full results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_evaluations": result.total_evaluations,
            "scenarios_count": result.scenarios_count,
            "strategies_count": result.strategies_count,
            "personas_count": result.personas_count,
            "results": [r.to_dict() for r in result.judge_results],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _compute_tts_stats(results: List[JudgeResult]) -> Dict[str, Any]:
        """Compute mean, std, min, max for each TTS dimension."""
        if not results:
            return {}

        t_scores = [r.transparency for r in results]
        tr_scores = [r.trust for r in results]
        s_scores = [r.satisfaction for r in results]

        def _stats(scores: List[int]) -> Dict[str, float]:
            return {
                "mean": round(statistics.mean(scores), 2),
                "std": round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 2),
                "min": min(scores),
                "max": max(scores),
                "n": len(scores),
            }

        return {
            "transparency": _stats(t_scores),
            "trust": _stats(tr_scores),
            "satisfaction": _stats(s_scores),
            "composite": _stats(
                [
                    round((t + tr + s) / 3, 2)
                    for t, tr, s in zip(t_scores, tr_scores, s_scores)
                ]
            ),
        }

    def _build_category_map(self) -> Dict[str, set]:
        """Map scenario categories to scenario IDs."""
        cat_map: Dict[str, set] = {}
        for sc in self.scenarios:
            cat = sc.get("category", "unknown")
            cat_map.setdefault(cat, set()).add(sc["id"])
        return cat_map

    def _compute_kruskal_wallis(self, results: List[JudgeResult]) -> Dict[str, Any]:
        """Run Kruskal-Wallis test comparing strategies on each TTS dimension."""
        try:
            from scipy.stats import kruskal
        except ImportError:
            return {"note": "scipy not available; statistical tests skipped"}

        tests: Dict[str, Any] = {}
        strategy_slugs = [s.slug for s in self.strategies]

        for dimension in ("transparency", "trust", "satisfaction"):
            groups = []
            for slug in strategy_slugs:
                scores = [
                    getattr(r, dimension) for r in results if r.strategy_name == slug
                ]
                if scores:
                    groups.append(scores)

            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                stat, p_value = kruskal(*groups)
                tests[dimension] = {
                    "H_statistic": round(float(stat), 4),
                    "p_value": round(float(p_value), 6),
                    "significant": p_value < 0.05,
                }
            else:
                tests[dimension] = {"note": "insufficient data"}

        return tests
