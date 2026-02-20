# evaluation/run_evaluation.py
"""
Week 3 Evaluation Runner — DSRM Stage 5

Usage:
    python -m evaluation.run_evaluation                         # default (mock mode)
    python -m evaluation.run_evaluation --output results/       # custom output dir
    python -m evaluation.run_evaluation --scenario deleg_01     # single scenario
    pytest evaluation/run_evaluation.py -v                      # as pytest

Outputs:
    evaluation_results.csv    — one row per scenario
    evaluation_summary.json   — aggregate metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest.mock as _mock
from pathlib import Path
from typing import Any, Dict, Optional

# ── Project root on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestration.aop_coordinator import AOPCoordinator  # noqa: E402
from app.orchestration.performance_store import PerformanceStore  # noqa: E402
from app.runtime.guardrails import NoOpGuardrails  # noqa: E402
from app.runtime.registry import AgentRegistry  # noqa: E402
from app.runtime.routing import Candidate, RoutePlan  # noqa: E402
from app.runtime.spine import RuntimeSpine  # noqa: E402

from evaluation.harness import EvaluationHarness  # noqa: E402
from evaluation.mock_factory import build_scenario_mock  # noqa: E402

# ── Scenarios path ──────────────────────────────────────────────────
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "ground_truth.json"


# ── Stub agent (same pattern as tests/test_spine_orchestration.py) ──


class StubAgent:
    """Minimal IAgent that returns a fixed response."""

    def __init__(self, agent_id: str, response: Dict[str, Any]):
        self._id = agent_id
        self._response = response
        self._meta: Dict[str, Any] = {}

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._response)

    def metadata(self) -> Dict[str, Any]:
        return {"id": self._id, **self._meta}


class ScenarioRouter:
    """Router that directs queries to the appropriate stub agent based on keywords."""

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    def route(self, query: str) -> RoutePlan:
        q = query.lower()

        # Refund/complaint/charge/unauthorized → refund agent
        refund_keywords = [
            "refund",
            "charge",
            "money back",
            "duplicate",
            "unauthorized",
            "cancel",
            "dispute",
            "stolen",
            "fees",
            "debited",
            "dispensed",
        ]
        if any(kw in q for kw in refund_keywords):
            return RoutePlan(
                primary="refund_agent",
                strategy="single",
                candidates=[Candidate(id="refund_agent", score=0.9, reason="refund keyword match")],
            )

        # Default → FAQ agent
        return RoutePlan(
            primary="faq_agent",
            strategy="single",
            candidates=[Candidate(id="faq_agent", score=0.85, reason="FAQ default")],
        )


# ── Build spine with stub agents ────────────────────────────────────


def build_eval_spine(tmp_dir: Path) -> RuntimeSpine:
    """Create a RuntimeSpine with stub agents for deterministic evaluation."""
    registry = AgentRegistry()

    # FAQ agent — returns answer with common banking keywords
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

    # Refund agent — returns workflow-like response
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

    # Lookup agent — for AOP delegation
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

    # Performance store in temp dir
    perf_store = PerformanceStore(path=str(tmp_dir / "eval_perf.json"))

    # AOP coordinator
    aop = AOPCoordinator(registry=registry, performance_store=perf_store)

    # Router
    router = ScenarioRouter(registry)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
    )

    return spine


# ── Run evaluation ──────────────────────────────────────────────────


def run_evaluation(
    output_dir: Path,
    scenario_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute all scenarios and write results."""
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_"))
    spine = build_eval_spine(tmp_dir)

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
        if not scenarios:
            print(f"[ERROR] No scenario with id={scenario_filter}")
            return {}

    # Run each scenario with its own mocks
    all_results = []
    for sc in scenarios:
        mock_responses = sc.get("mock_responses", {})
        mock_fn = build_scenario_mock(mock_responses)

        # Apply mocks for all LLM call sites (including voice rendering)
        def voice_mock(**kw):
            return {"messages": ["OK"], "quick_replies": []}

        with _mock.patch("app.llm_client.chat_json", mock_fn), _mock.patch(
            "app.orchestration.aop_coordinator.chat_json", mock_fn
        ), _mock.patch("app.orchestration.completeness_detector.chat_json", mock_fn), _mock.patch(
            "app.runtime.voice.chat_json", voice_mock
        ):
            try:
                _mock.patch("app.shared.workflow.chat_json", mock_fn).start()
            except AttributeError:
                pass

            harness = EvaluationHarness(spine, SCENARIOS_PATH)
            result = harness.run_scenario(sc)
            all_results.append(result)

            status = "PASS" if result.success else "FAIL"
            print(
                f"  [{status}] {sc['id']:20s}  "
                f"pattern={'OK' if result.pattern_correct else 'MISS':4s}  "
                f"agent={'OK' if result.agent_correct else 'MISS':4s}  "
                f"latency={result.latency_ms:.0f}ms"
            )

    # Compute metrics
    harness = EvaluationHarness(spine, SCENARIOS_PATH)
    metrics = harness.compute_metrics(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total scenarios:         {metrics.get('total_scenarios', 0)}")
    print(f"  Passed:                  {metrics.get('passed', 0)}")
    print(f"  Failed:                  {metrics.get('failed', 0)}")
    print(f"  Orchestration Accuracy:  {metrics.get('orchestration_accuracy', 0):.1%}")
    print(f"  Reasoning Accuracy:      {metrics.get('reasoning_accuracy', 0):.1%}")
    print(f"  Agent Accuracy:          {metrics.get('agent_accuracy', 0):.1%}")
    print(f"  Avg Latency:             {metrics.get('avg_latency_ms', 0):.1f} ms")

    solv = metrics.get("solvability_correlation")
    print(
        f"  Solvability Correlation: {solv:.4f}"
        if solv is not None
        else "  Solvability Correlation: N/A"
    )

    comp = metrics.get("completeness_rate")
    print(
        f"  Completeness Rate:       {comp:.1%}"
        if comp is not None
        else "  Completeness Rate:       N/A"
    )

    print("\n  Latency by category:")
    for cat, lat in metrics.get("latency_by_category", {}).items():
        steps = metrics.get("steps_by_category", {}).get(cat, 0)
        print(f"    {cat:30s}  {lat:8.1f} ms  ({steps:.1f} agent calls)")

    # Export results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness.export_csv(all_results, output_dir / "evaluation_results.csv")
    harness.export_json(all_results, output_dir / "evaluation_results.json")

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── pytest integration ──────────────────────────────────────────────


def test_all_scenarios_pass():
    """pytest entry point: run all scenarios and assert targets met."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="eval_out_"))
    metrics = run_evaluation(output_dir)

    assert (
        metrics["total_scenarios"] == 25
    ), f"Expected 25 scenarios, got {metrics['total_scenarios']}"
    assert (
        metrics["orchestration_accuracy"] >= 0.80
    ), f"Orchestration accuracy {metrics['orchestration_accuracy']:.1%} < 80% target"
    assert (
        metrics["reasoning_accuracy"] >= 0.75
    ), f"Reasoning accuracy {metrics['reasoning_accuracy']:.1%} < 75% target"


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="Output directory for results (default: evaluation/results)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by ID",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AGENT FACTORY ORCHESTRATION — EVALUATION HARNESS")
    print("DSRM Stage 5 | Week 3")
    print("=" * 60 + "\n")

    run_evaluation(Path(args.output), scenario_filter=args.scenario)
