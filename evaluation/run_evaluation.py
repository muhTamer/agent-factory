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
    """Router that directs queries to the appropriate stub agent based on keywords.

    Mirrors the real LLMRouter's intent-aware routing:
    - ACTIONABLE intents (refund, complaint) → domain agent with tools
    - INFORMATIONAL intents (FAQ, policy) → FAQ domain agent
    """

    def __init__(self, registry: AgentRegistry):
        self._registry = registry

    def route(self, query: str) -> RoutePlan:
        q = query.lower()

        # ── Detect INFORMATIONAL intent (asking about policies, not requesting action)
        info_patterns = [
            "what is your",
            "how do i",
            "how does",
            "tell me about",
            "explain",
            "policy",
            "procedure",
            "what are the",
            "can you describe",
        ]
        is_informational = any(p in q for p in info_patterns)

        # Informational queries about refunds/complaints → FAQ agent
        if is_informational:
            return RoutePlan(
                primary="agent_faq",
                strategy="single",
                candidates=[
                    Candidate(
                        id="agent_faq",
                        score=0.85,
                        reason="informational intent (FAQ)",
                    )
                ],
            )

        # ── ACTIONABLE: Complaint / dispute → complaints agent
        complaint_keywords = ["complaint", "formal dispute", "ombudsman", "escalate"]
        if any(kw in q for kw in complaint_keywords):
            return RoutePlan(
                primary="agent_complaints",
                strategy="single",
                candidates=[
                    Candidate(
                        id="agent_complaints",
                        score=0.92,
                        reason="complaint intent detected",
                    )
                ],
            )

        # ── ACTIONABLE: Refund / charge / unauthorized → refunds agent
        refund_keywords = [
            "refund",
            "charge",
            "money back",
            "duplicate",
            "unauthorized",
            "cancel",
            "stolen",
            "fees",
            "debited",
            "dispensed",
        ]
        if any(kw in q for kw in refund_keywords):
            return RoutePlan(
                primary="agent_refunds",
                strategy="single",
                candidates=[
                    Candidate(
                        id="agent_refunds",
                        score=0.90,
                        reason="refund intent detected",
                    )
                ],
            )

        # Default → FAQ agent
        return RoutePlan(
            primary="agent_faq",
            strategy="single",
            candidates=[
                Candidate(
                    id="agent_faq",
                    score=0.85,
                    reason="informational intent (FAQ)",
                )
            ],
        )


# ── Build spine with stub agents ────────────────────────────────────


def build_eval_spine(tmp_dir: Path) -> RuntimeSpine:
    """Create a RuntimeSpine with stub agents mirroring the real fleet.

    The real fleet has three domain_agent instances (all using ReAct engine):
      - agent_faq        — FAQ / informational queries
      - agent_refunds    — Refund processing with tools
      - agent_complaints — Complaint handling with tools
    """
    registry = AgentRegistry()

    # ── agent_faq (domain_agent) ──────────────────────────────────────
    faq = StubAgent(
        "agent_faq",
        {
            "answer": (
                "Based on our FAQ knowledge base: You can transfer your Current Account "
                "between branches. Outstation cheque clearing takes 7-14 working days. "
                "Premium CA offers free intercity clearing. SEZ units cannot open EEFC accounts. "
                "Flexi Account requires Rs. 75,000 initial deposit."
            ),
            "score": 0.82,
            "react_trace": [
                {
                    "step": 1,
                    "thought": "Looking up banking FAQ for the customer query.",
                    "action": "retrieve_knowledge",
                    "observation": "Found relevant FAQ entry in BankFAQs.csv.",
                },
                {
                    "step": 2,
                    "thought": "FAQ entry found, responding to customer.",
                    "action": "respond",
                    "observation": None,
                },
            ],
            "knowledge_sources": [
                {"query": "banking FAQ", "sources": ["BankFAQs.csv"]}
            ],
        },
    )
    faq._meta = {
        "type": "domain_agent",
        "description": "Answers customer FAQs about banking policies and account types",
        "capabilities": [
            "faq_answering",
            "policy_lookup",
            "knowledge_base_search",
            "account_information",
            "cheque_clearing",
            "deposit_requirements",
        ],
        "aop_eligible": True,
        "ready": True,
    }
    registry.register("agent_faq", faq, faq.metadata())

    # ── agent_refunds (domain_agent) ──────────────────────────────────
    refund = StubAgent(
        "agent_refunds",
        {
            "answer": (
                "Your refund request has been received. I've verified your identity "
                "and looked up the payment. The refund of EUR 200 has been initiated "
                "and you should see it within 3-5 business days."
            ),
            "score": 0.88,
            "react_trace": [
                {
                    "step": 1,
                    "thought": "Need to retrieve refund policy for eligibility check.",
                    "action": "retrieve_knowledge",
                    "observation": "Retrieved refunds_policy.yaml.",
                },
                {
                    "step": 2,
                    "thought": "Need to verify customer identity.",
                    "action": "call_tool",
                    "action_input": {"tool": "verify_identity"},
                    "observation": "Identity verified.",
                },
                {
                    "step": 3,
                    "thought": "Looking up payment details.",
                    "action": "call_tool",
                    "action_input": {"tool": "lookup_payment"},
                    "observation": "Payment found: EUR 200, age 5 days.",
                },
                {
                    "step": 4,
                    "thought": "Eligible per policy. Initiating refund.",
                    "action": "call_tool",
                    "action_input": {"tool": "initiate_refund"},
                    "observation": "Refund REF-001 initiated.",
                },
                {
                    "step": 5,
                    "thought": "Refund processed, responding to customer.",
                    "action": "respond",
                    "observation": None,
                },
            ],
            "tool_results": [
                {"step": 2, "tool": "verify_identity", "result": "verified"},
                {"step": 3, "tool": "lookup_payment", "result": "EUR 200"},
                {"step": 4, "tool": "initiate_refund", "result": "REF-001"},
            ],
            "knowledge_sources": [
                {"query": "refund policy", "sources": ["refunds_policy.yaml"]}
            ],
        },
    )
    refund._meta = {
        "type": "domain_agent",
        "description": "Processes refund and reversal requests against policy",
        "capabilities": [
            "refund_processing",
            "return_handling",
            "eligibility_check",
            "policy_evaluation",
            "transaction_reversal",
        ],
        "aop_eligible": True,
        "ready": True,
    }
    registry.register("agent_refunds", refund, refund.metadata())

    # ── agent_complaints (domain_agent) ───────────────────────────────
    complaints = StubAgent(
        "agent_complaints",
        {
            "answer": (
                "I've registered your complaint and created a support ticket. "
                "Your ticket ID is TKT-2024-001. Our team will investigate and "
                "respond within 48 hours."
            ),
            "score": 0.90,
            "react_trace": [
                {
                    "step": 1,
                    "thought": "Need complaints policy for handling procedure.",
                    "action": "retrieve_knowledge",
                    "observation": "Retrieved complaints_policy.yaml.",
                },
                {
                    "step": 2,
                    "thought": "Creating support ticket for the complaint.",
                    "action": "call_tool",
                    "action_input": {"tool": "create_ticket"},
                    "observation": "Ticket TKT-2024-001 created.",
                },
                {
                    "step": 3,
                    "thought": "Ticket created, responding to customer.",
                    "action": "respond",
                    "observation": None,
                },
            ],
            "tool_results": [
                {"step": 2, "tool": "create_ticket", "result": "TKT-2024-001"},
            ],
            "knowledge_sources": [
                {"query": "complaints policy", "sources": ["complaints_policy.yaml"]}
            ],
        },
    )
    complaints._meta = {
        "type": "domain_agent",
        "description": "Handles customer complaints, escalation, and ticket creation",
        "capabilities": [
            "complaint_handling",
            "ticket_creation",
            "escalation",
            "customer_support",
        ],
        "aop_eligible": True,
        "ready": True,
    }
    registry.register("agent_complaints", complaints, complaints.metadata())

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
        ), _mock.patch(
            "app.orchestration.completeness_detector.chat_json", mock_fn
        ), _mock.patch(
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
    summary_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── pytest integration ──────────────────────────────────────────────


def test_all_scenarios_pass():
    """pytest entry point: run all scenarios and assert targets met."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="eval_out_"))
    metrics = run_evaluation(output_dir)

    assert (
        metrics["total_scenarios"] >= 25
    ), f"Expected ≥25 scenarios, got {metrics['total_scenarios']}"
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
