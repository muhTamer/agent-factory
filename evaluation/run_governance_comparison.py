# evaluation/run_governance_comparison.py
"""
RQ3 Governance Comparison Runner

Runs the same evaluation scenarios under LOW, MEDIUM, and HIGH governance,
collects RQ3-specific metrics, and exports comparison tables for thesis
figure generation.

Usage:
    python -m evaluation.run_governance_comparison
    python -m evaluation.run_governance_comparison --output results/rq3/
    python -m evaluation.run_governance_comparison --levels low,high

Outputs:
    governance_comparison.json  — per-level metrics + trade-off deltas
    governance_results.csv      — one row per (scenario, governance_level)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import unittest.mock as _mock
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestration.aop_coordinator import AOPCoordinator  # noqa: E402
from app.orchestration.performance_store import PerformanceStore  # noqa: E402
from app.runtime.governance_config import GovernanceConfig, GovernanceLevel  # noqa: E402
from app.runtime.governance_guardrails import GovernanceAwareGuardrails  # noqa: E402
from app.runtime.policy_pack import PolicyPack  # noqa: E402
from app.runtime.registry import AgentRegistry  # noqa: E402
from app.runtime.spine import RuntimeSpine  # noqa: E402

from evaluation.harness import EvaluationHarness  # noqa: E402
from evaluation.governance_metrics import (  # noqa: E402
    GovernanceScenarioResult,
    compute_comparison_table,
)
from evaluation.mock_factory import build_scenario_mock  # noqa: E402
from evaluation.run_evaluation import ScenarioRouter, StubAgent  # noqa: E402

# ── Scenarios paths ──────────────────────────────────────────────────
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "ground_truth.json"
RQ3_SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "governance_scenarios.json"


# ── Build spine with governance-aware guardrails ─────────────────────


def build_governed_spine(
    tmp_dir: Path,
    config: GovernanceConfig,
) -> tuple:
    """Create a RuntimeSpine with governance-aware guardrails.

    Returns:
        (spine, guardrails) — guardrails reference needed to drain events.
    """
    registry = AgentRegistry()

    # Same stub agents as run_evaluation.py
    faq = StubAgent(
        "faq_agent",
        {
            "answer": (
                "Based on our FAQ knowledge base: You can transfer your Current Account "
                "between branches. Outstation cheque clearing takes 7-14 working days. "
                "Premium CA offers free intercity clearing. SEZ units cannot open EEFC accounts. "
                "Flexi Account requires Rs. 75,000 initial deposit."
            ),
            "score": 0.82,
        },
    )
    faq._meta = {
        "type": "faq_rag",
        "description": "Answers customer FAQs about banking policies and account types",
        "capabilities": [
            "faq_answering",
            "policy_lookup",
            "knowledge_base_search",
            "account_information",
            "cheque_clearing",
            "deposit_requirements",
        ],
        "ready": True,
    }
    registry.register("faq_agent", faq, faq.metadata())

    refund = StubAgent(
        "refund_agent",
        {
            "answer": "Your refund request has been received and is being processed.",
            "text": "Your refund request has been received and is being processed.",
            "score": 0.88,
            "current_state": "eligibility_check",
            "workflow_id": "refunds_workflow_v1",
            "terminal": False,
            "slots": {"customer_id": "CUST-001", "amount": 200},
            "missing_slots": [],
        },
    )
    refund._meta = {
        "type": "workflow_runner",
        "description": "Processes refund and reversal requests against policy",
        "capabilities": [
            "refund_processing",
            "return_handling",
            "eligibility_check",
            "policy_evaluation",
            "transaction_reversal",
        ],
        "ready": True,
    }
    registry.register("refund_agent", refund, refund.metadata())

    lookup = StubAgent(
        "lookup_agent",
        {
            "answer": "Customer record retrieved: account active, KYC verified.",
            "score": 0.90,
        },
    )
    lookup._meta = {
        "type": "tool_operator",
        "description": "Fetches and validates customer records",
        "capabilities": ["customer_lookup", "account_status", "kyc_verification"],
        "ready": True,
    }
    registry.register("lookup_agent", lookup, lookup.metadata())

    # PolicyPack with blocked phrases for governance testing
    pack = PolicyPack(
        name="governance_eval",
        version="1",
        max_query_chars=config.max_query_chars,
        intent_rules={
            "refund_request": {"mode": "allow"},
            "refund_policy_query": {"mode": "allow"},
        },
        route_to_intent={
            "refunds_workflow": "refund_request",
            "faq_rag": "refund_policy_query",
        },
        blocked_phrases=["guaranteed refund", "we will definitely"],
        pii_redaction=False,
    )

    guardrails = GovernanceAwareGuardrails(pack=pack, config=config)

    perf_store = PerformanceStore(path=str(tmp_dir / f"eval_perf_{config.level.value}.json"))
    aop = AOPCoordinator(registry=registry, performance_store=perf_store)
    router = ScenarioRouter(registry)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=guardrails,
        aop_coordinator=aop,
    )

    return spine, guardrails


# ── Run single governance level ──────────────────────────────────────


def run_single_level(
    level: GovernanceLevel,
    scenarios: List[Dict[str, Any]],
    tmp_dir: Path,
) -> List[GovernanceScenarioResult]:
    """Run all scenarios under a single governance level."""
    config = GovernanceConfig.for_level(level)
    spine, guardrails = build_governed_spine(tmp_dir, config)

    results: List[GovernanceScenarioResult] = []

    for sc in scenarios:
        mock_responses = sc.get("mock_responses", {})
        mock_fn = build_scenario_mock(mock_responses)

        def voice_mock(**kw):
            return {"messages": ["OK"], "quick_replies": []}

        # Clear governance events before each scenario
        guardrails.get_events()

        with (
            _mock.patch("app.llm_client.chat_json", mock_fn),
            _mock.patch("app.orchestration.aop_coordinator.chat_json", mock_fn),
            _mock.patch("app.orchestration.completeness_detector.chat_json", mock_fn),
            _mock.patch("app.runtime.voice.chat_json", voice_mock),
        ):
            try:
                _mock.patch("app.shared.workflow.chat_json", mock_fn).start()
            except AttributeError:
                pass

            harness = EvaluationHarness(spine, SCENARIOS_PATH)
            sc_result = harness.run_scenario(sc)

            # Drain governance events
            gov_events = guardrails.drain_events()

            # Count governance-specific metrics from events
            blocks = sum(1 for e in gov_events if e.get("action") == "blocked")
            escalations = sum(1 for e in gov_events if e.get("action") == "escalated")
            mutations = sum(1 for e in gov_events if e.get("action") == "mutated")
            skips = sum(1 for e in gov_events if e.get("action") == "skipped")

            # Autonomy: for LOW (no user confirmation required), all actions
            # are agent-initiated. For MEDIUM/HIGH, actions require confirmation.
            total_actions = max(sc_result.agent_calls, 1)
            agent_initiated = total_actions if not config.require_user_confirmation else 0

            gov_result = GovernanceScenarioResult(
                scenario_id=sc["id"],
                governance_level=level.value,
                category=sc["category"],
                success=sc_result.success,
                pattern_correct=sc_result.pattern_correct,
                agent_correct=sc_result.agent_correct,
                latency_ms=sc_result.latency_ms,
                agent_calls=sc_result.agent_calls,
                governance_blocks=blocks,
                governance_escalations=escalations,
                governance_mutations=mutations,
                governance_skips=skips,
                agent_initiated_actions=agent_initiated,
                total_actions=total_actions,
                autonomy_score=(agent_initiated / total_actions if total_actions > 0 else 0.0),
                intervention_rate=(blocks + escalations) / max(total_actions, 1),
                error=sc_result.error,
                governance_events=gov_events,
            )
            results.append(gov_result)

            status = "PASS" if gov_result.success else "FAIL"
            print(
                f"  [{level.value:6s}] [{status}] {sc['id']:20s}  "
                f"blocks={blocks} skips={skips} "
                f"lat={gov_result.latency_ms:.0f}ms"
            )

    return results


# ── Export ────────────────────────────────────────────────────────────


def export_comparison(
    comparison: Dict[str, Any],
    all_results: Dict[str, List[GovernanceScenarioResult]],
    output_dir: Path,
) -> None:
    """Export comparison results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary_path = output_dir / "governance_comparison.json"
    summary_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")

    # CSV: one row per (scenario, level)
    csv_path = output_dir / "governance_results.csv"
    fieldnames = [
        "scenario_id",
        "governance_level",
        "category",
        "success",
        "pattern_correct",
        "agent_correct",
        "latency_ms",
        "agent_calls",
        "governance_blocks",
        "governance_escalations",
        "governance_mutations",
        "governance_skips",
        "agent_initiated_actions",
        "total_actions",
        "autonomy_score",
        "intervention_rate",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for level_name, results in all_results.items():
            for r in results:
                writer.writerow(
                    {
                        "scenario_id": r.scenario_id,
                        "governance_level": r.governance_level,
                        "category": r.category,
                        "success": r.success,
                        "pattern_correct": r.pattern_correct,
                        "agent_correct": r.agent_correct,
                        "latency_ms": round(r.latency_ms, 2),
                        "agent_calls": r.agent_calls,
                        "governance_blocks": r.governance_blocks,
                        "governance_escalations": r.governance_escalations,
                        "governance_mutations": r.governance_mutations,
                        "governance_skips": r.governance_skips,
                        "agent_initiated_actions": r.agent_initiated_actions,
                        "total_actions": r.total_actions,
                        "autonomy_score": round(r.autonomy_score, 4),
                        "intervention_rate": round(r.intervention_rate, 4),
                        "error": r.error,
                    }
                )

    print(f"\n  Results exported to: {output_dir.resolve()}")


# ── Main comparison runner ───────────────────────────────────────────


def run_governance_comparison(
    output_dir: Path,
    levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Main entry point: run comparison across governance levels."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="gov_eval_"))

    # Load scenarios (original + RQ3-specific)
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if RQ3_SCENARIOS_PATH.exists():
        rq3_scenarios = json.loads(RQ3_SCENARIOS_PATH.read_text(encoding="utf-8"))
        scenarios.extend(rq3_scenarios)

    # Determine which levels to run
    level_list = [GovernanceLevel.LOW, GovernanceLevel.MEDIUM, GovernanceLevel.HIGH]
    if levels:
        level_list = [GovernanceLevel(lv) for lv in levels]

    all_results: Dict[str, List[GovernanceScenarioResult]] = {}

    for level in level_list:
        print(f"\n{'=' * 60}")
        print(f"GOVERNANCE LEVEL: {level.value.upper()}")
        print(f"{'=' * 60}")
        results = run_single_level(level, scenarios, tmp_dir)
        all_results[level.value] = results

    # Compute comparison
    comparison = compute_comparison_table(all_results)

    # Print summary
    print(f"\n{'=' * 60}")
    print("RQ3 GOVERNANCE COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    for level_name, metrics in comparison["per_level"].items():
        print(f"\n  [{level_name.upper()}]")
        print(f"    Task Completion:    {metrics['task_completion_rate']:.1%}")
        print(f"    Autonomy Score:     {metrics['autonomy_score']:.1%}")
        print(f"    Intervention Rate:  {metrics['intervention_rate']:.1%}")
        print(f"    Avg Latency:        {metrics['avg_latency_ms']:.1f} ms")
        print(f"    False Positive:     {metrics['false_positive_rate']:.1%}")
        print(f"    Gov. Blocks:        {metrics['total_governance_blocks']}")

    if comparison.get("tradeoffs"):
        t = comparison["tradeoffs"]
        print("\n  TRADE-OFF DELTAS (LOW -> HIGH):")
        print(f"    Completion drop:    {t['completion_delta']:+.1%}")
        print(f"    Autonomy drop:      {t['autonomy_delta']:+.1%}")
        print(f"    Intervention rise:  {t['intervention_delta']:+.1%}")
        print(f"    Latency increase:   {t['latency_delta_ms']:+.1f} ms")

    export_comparison(comparison, all_results, output_dir)
    return comparison


# ── pytest integration ───────────────────────────────────────────────


def test_governance_comparison_runs():
    """pytest entry point: run all governance levels and assert no crashes."""
    output_dir = Path(tempfile.mkdtemp(prefix="gov_eval_out_"))
    comparison = run_governance_comparison(output_dir)

    assert "per_level" in comparison
    assert len(comparison["per_level"]) == 3
    for level_name in ["low", "medium", "high"]:
        assert level_name in comparison["per_level"]
        metrics = comparison["per_level"][level_name]
        assert metrics["total_scenarios"] > 0


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ3 Governance Comparison Runner")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rq3",
        help="Output directory for results (default: evaluation/results/rq3)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated levels to run (e.g. low,high)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RQ3: SAFETY/COMPLIANCE TRADE-OFF EVALUATION")
    print("=" * 60)

    levels = args.levels.split(",") if args.levels else None
    run_governance_comparison(Path(args.output), levels=levels)
