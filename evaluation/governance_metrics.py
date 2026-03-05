# evaluation/governance_metrics.py
"""
RQ3 Governance Metrics — Safety/Compliance Trade-off Computation

Computes thesis-aligned metrics for comparing governance levels:
  - Task Completion Rate     (per level)
  - Autonomy Score           (agent-initiated / total actions)
  - Intervention Rate        (blocks + escalations / total requests)
  - False Positive Rate      (over-enforcement ratio)
  - Trade-off Deltas         (LOW → HIGH differences)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GovernanceScenarioResult:
    """Result for a single scenario under a specific governance level."""

    scenario_id: str
    governance_level: str
    category: str

    # Core metrics (from existing harness)
    success: bool = False
    pattern_correct: bool = False
    agent_correct: bool = False
    latency_ms: float = 0.0
    agent_calls: int = 0

    # RQ3-specific metrics
    governance_blocks: int = 0
    governance_escalations: int = 0
    governance_mutations: int = 0
    governance_skips: int = 0
    agent_initiated_actions: int = 0
    total_actions: int = 0

    # Derived (computed post-hoc)
    autonomy_score: float = 0.0
    intervention_rate: float = 0.0

    error: Optional[str] = None
    governance_events: List[Dict[str, Any]] = field(default_factory=list)


def compute_rq3_metrics(results: List[GovernanceScenarioResult]) -> Dict[str, Any]:
    """Compute aggregate RQ3 metrics for a single governance level."""
    if not results:
        return {}

    n = len(results)
    level = results[0].governance_level

    # Task Completion Rate
    task_completion_rate = sum(1 for r in results if r.success) / n

    # Intervention Rate: (blocks + escalations) / total requests
    total_blocks = sum(r.governance_blocks for r in results)
    total_escalations = sum(r.governance_escalations for r in results)
    intervention_rate = (total_blocks + total_escalations) / n

    # Autonomy Score: (agent-initiated actions / total actions)
    total_agent_actions = sum(r.agent_initiated_actions for r in results)
    total_all_actions = sum(r.total_actions for r in results)
    autonomy_score = total_agent_actions / total_all_actions if total_all_actions > 0 else 0.0

    # Governance Coverage: how many checks ran vs. skipped
    total_mutations = sum(r.governance_mutations for r in results)
    total_skips = sum(r.governance_skips for r in results)

    # Average latency
    avg_latency = sum(r.latency_ms for r in results) / n

    # False positive rate: governance blocks on scenarios that
    # succeed under the most permissive level (LOW). Approximated
    # here as blocks on scenarios where the expected outcome is success.
    blocked_results = [r for r in results if r.governance_blocks > 0]
    # Among blocked results, those that still succeeded had false-positive blocks
    false_positives = [r for r in blocked_results if r.success]
    false_positive_rate = len(false_positives) / len(blocked_results) if blocked_results else 0.0

    return {
        "governance_level": level,
        "total_scenarios": n,
        "task_completion_rate": round(task_completion_rate, 4),
        "intervention_rate": round(intervention_rate, 4),
        "autonomy_score": round(autonomy_score, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "total_governance_blocks": total_blocks,
        "total_governance_escalations": total_escalations,
        "total_governance_mutations": total_mutations,
        "total_governance_skips": total_skips,
        "false_positive_rate": round(false_positive_rate, 4),
        "passed": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
    }


def compute_comparison_table(
    all_results: Dict[str, List[GovernanceScenarioResult]],
) -> Dict[str, Any]:
    """
    Compute comparison across governance levels.

    Args:
        all_results: {"low": [...], "medium": [...], "high": [...]}

    Returns:
        Dict with per-level metrics and trade-off deltas.
    """
    per_level: Dict[str, Any] = {}
    for level_name, results in all_results.items():
        per_level[level_name] = compute_rq3_metrics(results)

    # Compute trade-off deltas (LOW → HIGH)
    tradeoffs: Dict[str, Any] = {}
    if "low" in per_level and "high" in per_level:
        low = per_level["low"]
        high = per_level["high"]
        tradeoffs = {
            "completion_delta": round(
                low["task_completion_rate"] - high["task_completion_rate"], 4
            ),
            "autonomy_delta": round(low["autonomy_score"] - high["autonomy_score"], 4),
            "intervention_delta": round(high["intervention_rate"] - low["intervention_rate"], 4),
            "latency_delta_ms": round(high["avg_latency_ms"] - low["avg_latency_ms"], 2),
        }

    return {
        "per_level": per_level,
        "tradeoffs": tradeoffs,
    }
