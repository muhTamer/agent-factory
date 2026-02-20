# evaluation/harness.py
"""
Evaluation Harness — DSRM Stage 5 (Week 3)

Runs ground-truth scenarios against the RuntimeSpine and collects the 6
quantitative metrics committed to in the thesis Methods chapter:

  1. Orchestration Accuracy   — % correct pattern selection
  2. Orchestration Efficiency  — latency (ms) and agent-call count per category
  3. Reasoning Accuracy        — % successful task completion
  4. Solvability Correlation   — Spearman ρ(predicted confidence, actual success)
  5. Completeness Rate         — mean completeness score for delegation scenarios
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.runtime.spine import RuntimeSpine


# ── Result dataclasses ──────────────────────────────────────────────


@dataclass
class TurnResult:
    """Outcome of a single conversation turn within a scenario."""

    turn_index: int
    query: str
    pattern_correct: bool = False
    agent_correct: bool = False
    answer_keywords_found: bool = False
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregate result for a complete scenario (possibly multi-turn)."""

    scenario_id: str
    category: str
    description: str
    success: bool = False
    pattern_correct: bool = False
    agent_correct: bool = False
    answer_keywords_found: bool = False
    latency_ms: float = 0.0
    agent_calls: int = 0
    completeness_score: Optional[float] = None
    solvability_score: Optional[float] = None
    error: Optional[str] = None
    turns: List[TurnResult] = field(default_factory=list)


# ── Harness ─────────────────────────────────────────────────────────


class EvaluationHarness:
    """Runs ground-truth scenarios and collects metrics."""

    def __init__(self, spine: RuntimeSpine, scenarios_path: str | Path):
        self.spine = spine
        self.scenarios = self._load_scenarios(scenarios_path)

    # ── Execution ───────────────────────────────────────────────────

    def run_all(self) -> List[ScenarioResult]:
        """Run every scenario and return results."""
        results: List[ScenarioResult] = []
        for sc in self.scenarios:
            results.append(self.run_scenario(sc))
        return results

    def run_scenario(self, scenario: Dict[str, Any]) -> ScenarioResult:
        """Execute a single scenario (possibly multi-turn)."""
        sc_id = scenario["id"]
        category = scenario["category"]
        description = scenario.get("description", "")
        turns_spec = scenario.get("turns", [])

        thread_id = f"eval_{sc_id}"
        result = ScenarioResult(
            scenario_id=sc_id,
            category=category,
            description=description,
        )

        total_latency = 0.0
        total_agent_calls = 0
        all_pattern_ok = True
        all_agent_ok = True
        all_keywords_ok = True
        last_response: Dict[str, Any] = {}

        for i, turn in enumerate(turns_spec):
            query = turn["query"]
            expected = turn.get("expected", {})

            t0 = time.perf_counter()
            try:
                resp = self.spine.handle_chat(
                    query,
                    request_id=f"{sc_id}_turn{i}",
                    context={"thread_id": thread_id},
                )
            except Exception as e:
                result.error = f"Turn {i} raised: {e}"
                result.turns.append(
                    TurnResult(turn_index=i, query=query, raw_response={"error": str(e)})
                )
                break

            latency = (time.perf_counter() - t0) * 1000.0
            total_latency += latency
            last_response = resp or {}

            # Count agent calls
            subtask_results = last_response.get("subtask_results")
            if isinstance(subtask_results, list):
                total_agent_calls += len(subtask_results)
            else:
                total_agent_calls += 1

            # Check pattern
            actual_pattern = self._detect_pattern(last_response)
            expected_pattern = expected.get("pattern")
            pattern_ok = (expected_pattern is None) or (actual_pattern == expected_pattern)

            # Check agent
            actual_agent = last_response.get("agent_id", "")
            expected_agent_contains = expected.get("agent_contains")
            agent_ok = (expected_agent_contains is None) or (
                expected_agent_contains.lower() in (actual_agent or "").lower()
            )

            # Check answer keywords
            answer_text = self._extract_answer(last_response)
            expected_keywords = expected.get("answer_contains", [])
            keywords_ok = (
                all(kw.lower() in answer_text.lower() for kw in expected_keywords)
                if expected_keywords
                else True
            )

            if not pattern_ok:
                all_pattern_ok = False
            if not agent_ok:
                all_agent_ok = False
            if not keywords_ok:
                all_keywords_ok = False

            result.turns.append(
                TurnResult(
                    turn_index=i,
                    query=query,
                    pattern_correct=pattern_ok,
                    agent_correct=agent_ok,
                    answer_keywords_found=keywords_ok,
                    latency_ms=latency,
                    raw_response=last_response,
                )
            )

        # Aggregate
        result.latency_ms = total_latency
        result.agent_calls = total_agent_calls
        result.pattern_correct = all_pattern_ok
        result.agent_correct = all_agent_ok
        result.answer_keywords_found = all_keywords_ok

        # Completeness (for AOP scenarios)
        comp = last_response.get("completeness")
        if isinstance(comp, dict):
            result.completeness_score = comp.get("coverage_ratio")

        # Solvability (for AOP scenarios — use average assignment score)
        solv = last_response.get("solvability")
        if isinstance(solv, dict):
            scores = solv.get("assignment_scores", {})
            if scores:
                result.solvability_score = sum(scores.values()) / len(scores)

        # For direct routing, use agent score as solvability proxy
        if result.solvability_score is None:
            try:
                result.solvability_score = float(last_response.get("score", 0))
            except (TypeError, ValueError):
                pass

        # Success: all turns passed + no error
        success_criteria = scenario.get("success_criteria", "all_turns_pass")
        if success_criteria == "answer_not_empty":
            answer = self._extract_answer(last_response)
            result.success = bool(answer.strip()) and result.error is None
        elif success_criteria == "final_state":
            expected_state = scenario.get("expected_final_state")
            actual_state = last_response.get("current_state")
            result.success = actual_state == expected_state and result.error is None
        else:  # "all_turns_pass"
            result.success = (
                all_pattern_ok and all_agent_ok and all_keywords_ok and result.error is None
            )

        return result

    # ── Metrics ─────────────────────────────────────────────────────

    def compute_metrics(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Compute the 6 thesis evaluation metrics."""
        if not results:
            return {}

        n = len(results)

        # 1. Orchestration Accuracy
        orchestration_accuracy = sum(1 for r in results if r.pattern_correct) / n

        # 2. Orchestration Efficiency — latency and steps by category
        categories = sorted(set(r.category for r in results))
        latency_by_cat = {}
        steps_by_cat = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            latency_by_cat[cat] = sum(r.latency_ms for r in cat_results) / len(cat_results)
            steps_by_cat[cat] = sum(r.agent_calls for r in cat_results) / len(cat_results)

        # 3. Reasoning Accuracy
        reasoning_accuracy = sum(1 for r in results if r.success) / n

        # 4. Solvability Correlation (Spearman ρ)
        solvability_pairs = [
            (r.solvability_score, 1.0 if r.success else 0.0)
            for r in results
            if r.solvability_score is not None
        ]
        solvability_rho = self._spearman_rho(solvability_pairs)

        # 5. Completeness Rate (delegation scenarios only)
        comp_scores = [r.completeness_score for r in results if r.completeness_score is not None]
        completeness_rate = sum(comp_scores) / len(comp_scores) if comp_scores else None

        # 6. Agent accuracy
        agent_accuracy = sum(1 for r in results if r.agent_correct) / n

        return {
            "orchestration_accuracy": round(orchestration_accuracy, 4),
            "reasoning_accuracy": round(reasoning_accuracy, 4),
            "agent_accuracy": round(agent_accuracy, 4),
            "solvability_correlation": (
                round(solvability_rho, 4) if solvability_rho is not None else None
            ),
            "completeness_rate": (
                round(completeness_rate, 4) if completeness_rate is not None else None
            ),
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 2),
            "latency_by_category": {k: round(v, 2) for k, v in latency_by_cat.items()},
            "steps_by_category": {k: round(v, 2) for k, v in steps_by_cat.items()},
            "total_scenarios": n,
            "passed": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        }

    # ── Export ──────────────────────────────────────────────────────

    def export_csv(self, results: List[ScenarioResult], path: str | Path) -> None:
        """Write one row per scenario to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "scenario_id",
            "category",
            "description",
            "success",
            "pattern_correct",
            "agent_correct",
            "answer_keywords_found",
            "latency_ms",
            "agent_calls",
            "completeness_score",
            "solvability_score",
            "error",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "scenario_id": r.scenario_id,
                        "category": r.category,
                        "description": r.description,
                        "success": r.success,
                        "pattern_correct": r.pattern_correct,
                        "agent_correct": r.agent_correct,
                        "answer_keywords_found": r.answer_keywords_found,
                        "latency_ms": round(r.latency_ms, 2),
                        "agent_calls": r.agent_calls,
                        "completeness_score": r.completeness_score,
                        "solvability_score": (
                            round(r.solvability_score, 4) if r.solvability_score else None
                        ),
                        "error": r.error,
                    }
                )

    def export_json(self, results: List[ScenarioResult], path: str | Path) -> None:
        """Write full results (including per-turn data) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            d = asdict(r)
            # Remove raw_response from turns to keep file manageable
            for t in d.get("turns", []):
                t.pop("raw_response", None)
            data.append(d)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _load_scenarios(path: str | Path) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scenarios file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _detect_pattern(response: Dict[str, Any]) -> str:
        """Infer orchestration pattern from response keys."""
        if response.get("orchestration_pattern") == "hierarchical_delegation":
            return "hierarchical_delegation"
        if response.get("current_state") and response.get("workflow_id"):
            return "fsm_workflow"
        return "direct"

    @staticmethod
    def _extract_answer(response: Dict[str, Any]) -> str:
        """Extract human-readable answer text from a response dict.

        Prefer 'answer' over 'text' because voice rendering may overwrite
        resp['text'] with a short customer-facing message, while 'answer'
        preserves the full agent response.
        """
        for key in ("answer", "text", "message"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    @staticmethod
    def _spearman_rho(
        pairs: List[tuple],
    ) -> Optional[float]:
        """
        Compute Spearman rank correlation coefficient.

        Falls back to manual computation if scipy is not available.
        """
        if len(pairs) < 3:
            return None

        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]

        try:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(x, y)
            return float(rho) if rho == rho else None  # NaN check
        except ImportError:
            pass

        # Manual Spearman: rank both, compute Pearson on ranks
        def _rank(vals):
            n = len(vals)
            indexed = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        n = len(rx)
        mean_rx = sum(rx) / n
        mean_ry = sum(ry) / n
        cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
        std_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
        std_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5

        if std_x < 1e-12 or std_y < 1e-12:
            return None

        return cov / (std_x * std_y)
