# tests/test_multi_intent_workflows.py
"""
End-to-end workflow tests for multi-intent AOP orchestration through RuntimeSpine.

Tests verify behavioral properties for queries with multiple intents:
  - Correct decomposition into subtasks and menu presentation
  - Sequential task execution (one at a time, menu first)
  - Correct agent routing per subtask (via solvability estimator)
  - Remaining task tracking after each subtask completes
  - Continuation flow (user selects next task)
  - Domain agent multi-turn within AOP subtask execution
  - Slot propagation across subtasks
  - Task decline clears plan
  - Full lifecycle: decompose → menu → execute → remaining → complete
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from app.orchestration.aop_coordinator import AOPCoordinator
from app.orchestration.performance_store import PerformanceStore
from app.runtime.guardrails import NoOpGuardrails
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Candidate, RoutePlan
from app.runtime.spine import THREAD_CTX, RuntimeSpine

# ── Helpers ──────────────────────────────────────────────────────────


class TrackingAgent:
    """Mock agent that records calls and returns configurable responses.

    Supports:
    - Fixed response dict (same every time)
    - Response list (returns i-th response on i-th call)
    - Response function (dynamic based on request + call count)
    """

    def __init__(
        self,
        agent_id: str,
        responses=None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self._id = agent_id
        self._meta = meta or {}
        self.calls: List[Dict[str, Any]] = []
        self._response_fn: Any = None

        if responses is None:
            self._responses = [{"answer": "OK", "text": "OK", "score": 0.8}]
        elif callable(responses):
            self._response_fn = responses
            self._responses = []
        elif isinstance(responses, list):
            self._responses = responses
        else:
            self._responses = [responses]

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(request)
        if self._response_fn:
            return self._response_fn(request, len(self.calls))
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        return dict(self._responses[idx])

    def metadata(self) -> Dict[str, Any]:
        return {"id": self._id, **self._meta}


class FixedRouter:
    """Router that always returns a fixed primary (fallback for direct path)."""

    def __init__(self, primary: str):
        self._primary = primary

    def route(self, query: str) -> RoutePlan:
        return RoutePlan(
            primary=self._primary,
            strategy="single",
            candidates=[Candidate(id=self._primary, score=1.0, reason="fixed")],
        )


def make_mock_llm(subtasks: List[str]):
    """Create a mock LLM function that handles all spine/AOP LLM calls.

    Routes based on system prompt content:
    - Pattern classification → hierarchical_delegation
    - Decomposition → returns provided subtasks
    - Completeness → always complete
    - Voice rendering → simple messages with quick_replies
    """

    def mock(**kwargs):
        messages = kwargs.get("messages", [])
        system_msg = ""
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
                break
        lower = system_msg.lower()

        if "query classifier" in lower:
            return {"pattern": "hierarchical_delegation"}

        if "decomposition" in lower:
            return {"subtasks": list(subtasks)}

        if "completeness" in lower:
            return {
                "complete": True,
                "missing": [],
                "redundant": [],
                "coverage_ratio": 1.0,
                "reasoning": "All covered.",
            }

        if "customer-service" in lower or "chat voice" in lower:
            labels = []
            for i, s in enumerate(subtasks):
                clean = s
                for pfx in ("INFORMATIONAL: ", "ACTION: "):
                    if clean.startswith(pfx):
                        clean = clean[len(pfx) :]
                        break
                labels.append(f"{i + 1}. {clean[:60]}")
            labels.append("No thanks")
            return {
                "messages": [f"I can help with {len(subtasks)} tasks."],
                "quick_replies": labels,
            }

        return {"pattern": "hierarchical_delegation"}

    return mock


def build_multi_intent_spine(
    tmp_path,
    monkeypatch,
    agents: Dict[str, TrackingAgent],
    subtasks: List[str],
) -> tuple:
    """Build a RuntimeSpine with AOP coordinator and tracking agents."""
    registry = AgentRegistry()
    for aid, agent in agents.items():
        registry.register(aid, agent, agent.metadata())

    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    aop = AOPCoordinator(registry=registry, performance_store=store)

    first_agent = next(iter(agents.keys()))
    router = FixedRouter(first_agent)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
        governance_enabled=False,
    )

    mock_llm = make_mock_llm(subtasks)
    monkeypatch.setattr("app.llm_client.chat_json", mock_llm)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock_llm)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock_llm)

    return spine, registry, store


# ── Helpers for agent metadata ───────────────────────────────────────


def _domain_agent_meta(description: str, capabilities: List[str]) -> Dict[str, Any]:
    """Standard domain_agent metadata for testing."""
    return {
        "type": "domain_agent",
        "agent_kind": "domain_agent",
        "description": description,
        "capabilities": capabilities,
        "aop_eligible": True,
    }


# ══════════════════════════════════════════════════════════════════════
# Scenario 1: Decomposition & Menu Presentation
# ══════════════════════════════════════════════════════════════════════


class TestDecompositionAndMenu:
    """Multi-intent query is decomposed and presented as a menu (no execution)."""

    def test_multi_intent_presents_menu(self, monkeypatch, tmp_path):
        """Two-intent query → menu with 2 tasks, no agents called."""
        refund = TrackingAgent(
            "refund_agent",
            meta=_domain_agent_meta(
                "Handles refund requests for payments",
                ["refund_processing", "payment_refund"],
            ),
        )
        faq = TrackingAgent(
            "faq_agent",
            meta=_domain_agent_meta(
                "Answers questions about accounts and policies",
                ["account_faq", "policy_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund for payment",
            "INFORMATIONAL: account opening process",
        ]

        spine, _, store = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        result = spine.handle_chat(
            "I need a refund for my payment and how do I open an account?",
            context={"thread_id": "test_menu_1"},
        )

        assert result.get("orchestration_pattern") == "aop_task_menu"
        assert "task_menu" in result
        assert len(result["task_menu"]) == 2
        assert len(refund.calls) == 0
        assert len(faq.calls) == 0
        assert len(store.query()) == 0

    def test_menu_includes_agent_assignments(self, monkeypatch, tmp_path):
        """Task menu items should include which agent is assigned to each subtask."""
        refund = TrackingAgent(
            "refund_agent",
            meta=_domain_agent_meta(
                "Handles refund payment processing",
                ["refund_processing", "payment_refund"],
            ),
        )
        faq = TrackingAgent(
            "faq_agent",
            meta=_domain_agent_meta(
                "Answers account opening questions and policies",
                ["account_opening_faq", "policy_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund for payment",
            "INFORMATIONAL: account opening questions",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        result = spine.handle_chat(
            "Refund my payment and tell me about account opening",
            context={"thread_id": "test_menu_2"},
        )

        menu = result.get("task_menu", [])
        agent_ids = {item["agent_id"] for item in menu}
        # Both agents should be assigned (not both tasks to one agent)
        assert len(agent_ids) == 2

    def test_plan_stored_in_thread_context(self, monkeypatch, tmp_path):
        """After menu presentation, the plan should be stored in thread context."""
        agent_a = TrackingAgent(
            "agent_a",
            meta=_domain_agent_meta(
                "Handles refund processing tasks", ["refund_processing"]
            ),
        )
        agent_b = TrackingAgent(
            "agent_b",
            meta=_domain_agent_meta(
                "Handles complaint handling tasks", ["complaint_handling"]
            ),
        )

        subtasks = ["ACTION: refund processing", "ACTION: complaint handling"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat(
            "Refund and also file a complaint",
            context={"thread_id": "test_stored_1"},
        )

        ctx = THREAD_CTX.get("test_stored_1", {})
        assert "_pending_aop" in ctx
        plan = ctx["_pending_aop"]
        assert len(plan["subtasks"]) == 2
        assert all(s.get("result") is None for s in plan["subtasks"])


# ══════════════════════════════════════════════════════════════════════
# Scenario 2: Sequential Task Execution
# ══════════════════════════════════════════════════════════════════════


class TestSequentialExecution:
    """Tasks execute one at a time when selected by the user."""

    def test_select_task_executes_one(self, monkeypatch, tmp_path):
        """User selects '1' → only one task executes."""
        refund = TrackingAgent(
            "refund_agent",
            responses=[
                {
                    "answer": "Refund processed.",
                    "text": "Refund processed.",
                    "score": 0.9,
                }
            ],
            meta=_domain_agent_meta("Refund payment processing", ["refund_payment"]),
        )
        faq = TrackingAgent(
            "faq_agent",
            responses=[
                {
                    "answer": "Account opening is easy.",
                    "text": "Account opening is easy.",
                    "score": 0.85,
                }
            ],
            meta=_domain_agent_meta("Account opening questions", ["account_opening"]),
        )

        subtasks = ["ACTION: refund payment", "INFORMATIONAL: account opening"]

        spine, _, store = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Process my refund and tell me about account opening",
            context={"thread_id": "test_seq_1"},
        )
        result = spine.handle_chat("1", context={"thread_id": "test_seq_1"})

        assert result.get("orchestration_pattern") == "aop_task_result"
        assert "executed_subtask" in result
        total_calls = len(refund.calls) + len(faq.calls)
        assert total_calls == 1
        assert len(store.query()) == 1

    def test_remaining_tasks_offered_after_execution(self, monkeypatch, tmp_path):
        """After executing one task, remaining tasks should appear in response."""
        refund = TrackingAgent(
            "refund_agent",
            responses=[{"answer": "Done.", "text": "Done.", "score": 0.9}],
            meta=_domain_agent_meta("Refund processing", ["refund_processing"]),
        )
        faq = TrackingAgent(
            "faq_agent",
            responses=[{"answer": "FAQ.", "text": "FAQ.", "score": 0.85}],
            meta=_domain_agent_meta("Account questions", ["account_questions"]),
        )

        subtasks = ["ACTION: refund processing", "INFORMATIONAL: account questions"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Refund and account questions", context={"thread_id": "test_seq_2"}
        )
        result = spine.handle_chat("1", context={"thread_id": "test_seq_2"})

        remaining = result.get("remaining_subtasks")
        assert remaining is not None
        assert len(remaining) >= 1

    def test_execute_all_tasks_completes_plan(self, monkeypatch, tmp_path):
        """After all tasks execute, plan should be removed from context."""
        refund = TrackingAgent(
            "refund_agent",
            responses=[
                {"answer": "Refund done.", "text": "Refund done.", "score": 0.9}
            ],
            meta=_domain_agent_meta("Refund processing", ["refund_processing"]),
        )
        faq = TrackingAgent(
            "faq_agent",
            responses=[
                {"answer": "Account info.", "text": "Account info.", "score": 0.85}
            ],
            meta=_domain_agent_meta("Account questions answers", ["account_questions"]),
        )

        subtasks = ["ACTION: refund processing", "INFORMATIONAL: account questions"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Refund and account questions", context={"thread_id": "test_seq_3"}
        )
        spine.handle_chat("1", context={"thread_id": "test_seq_3"})
        spine.handle_chat("1", context={"thread_id": "test_seq_3"})

        ctx = THREAD_CTX.get("test_seq_3", {})
        assert "_pending_aop" not in ctx


# ══════════════════════════════════════════════════════════════════════
# Scenario 3: Correct Agent Routing
# ══════════════════════════════════════════════════════════════════════


class TestAgentRouting:
    """Each subtask should be routed to the most capable agent by solvability."""

    def test_solvability_assigns_both_agents(self, monkeypatch, tmp_path):
        """With distinct subtasks, both agents should be assigned (not all to one)."""
        refund = TrackingAgent(
            "refund_agent",
            meta=_domain_agent_meta(
                "Handles refund payment processing",
                ["refund_processing", "payment_refund"],
            ),
        )
        faq = TrackingAgent(
            "faq_agent",
            meta=_domain_agent_meta(
                "Answers account opening questions and policy",
                ["account_opening", "policy_faq"],
            ),
        )

        subtasks = [
            "ACTION: process refund for payment",
            "INFORMATIONAL: account opening questions and policy",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        result = spine.handle_chat(
            "Refund my payment and tell me about account opening",
            context={"thread_id": "test_route_1"},
        )

        solv = result.get("solvability", {})
        assignments = solv.get("assignments", {})
        assigned_agents = set(assignments.values())
        assert len(assigned_agents) == 2

    def test_agent_receives_query_without_prefix(self, monkeypatch, tmp_path):
        """Agent should receive the subtask description stripped of INFORMATIONAL/ACTION prefix."""
        faq = TrackingAgent(
            "faq_agent",
            responses=[{"answer": "Info.", "text": "Info.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Answers policy questions about returns", ["policy_questions"]
            ),
        )
        other = TrackingAgent(
            "other_agent",
            responses=[{"answer": "Done.", "text": "Done.", "score": 0.8}],
            meta=_domain_agent_meta(
                "Handles order tracking and shipment", ["order_tracking"]
            ),
        )

        subtasks = [
            "INFORMATIONAL: policy questions about returns",
            "ACTION: track order shipment",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"faq_agent": faq, "other_agent": other},
            subtasks,
        )

        spine.handle_chat(
            "Tell me about returns policy and track my order",
            context={"thread_id": "test_route_2"},
        )
        spine.handle_chat("1", context={"thread_id": "test_route_2"})

        # At least one agent should have been called
        called = faq if faq.calls else other
        query = called.calls[0].get("query", "")
        # Query should NOT contain INFORMATIONAL:/ACTION: prefix
        assert not query.startswith("INFORMATIONAL: ")
        assert not query.startswith("ACTION: ")


# ══════════════════════════════════════════════════════════════════════
# Scenario 4: Task Decline
# ══════════════════════════════════════════════════════════════════════


class TestTaskDecline:
    """User declining should clear the pending plan."""

    @pytest.mark.parametrize(
        "decline_phrase",
        ["no thanks", "no", "done", "skip", "that's all", "nope"],
    )
    def test_decline_clears_plan(self, monkeypatch, tmp_path, decline_phrase):
        """Various decline phrases should clear the pending plan."""
        agent_a = TrackingAgent(
            "agent_a",
            meta=_domain_agent_meta("Handles refund tasks", ["refund"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            meta=_domain_agent_meta("Handles complaint tasks", ["complaint"]),
        )

        subtasks = ["ACTION: refund task", "ACTION: complaint task"]
        tid = f"test_decline_{decline_phrase.replace(' ', '_')}"

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat("Refund and complaint", context={"thread_id": tid})
        result = spine.handle_chat(decline_phrase, context={"thread_id": tid})

        assert result.get("orchestration_pattern") == "aop_plan_declined"
        ctx = THREAD_CTX.get(tid, {})
        assert "_pending_aop" not in ctx

    def test_decline_after_partial_execution(self, monkeypatch, tmp_path):
        """Declining after completing one task should still clear the plan."""
        agent_a = TrackingAgent(
            "agent_a",
            responses=[{"answer": "A done.", "text": "A done.", "score": 0.9}],
            meta=_domain_agent_meta("Handles refund tasks", ["refund_tasks"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            meta=_domain_agent_meta("Handles complaint tasks", ["complaint_tasks"]),
        )

        subtasks = ["ACTION: refund tasks", "ACTION: complaint tasks"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat(
            "Refund and complaint", context={"thread_id": "test_decline_partial_1"}
        )
        spine.handle_chat("1", context={"thread_id": "test_decline_partial_1"})
        result = spine.handle_chat(
            "no thanks", context={"thread_id": "test_decline_partial_1"}
        )

        assert result.get("orchestration_pattern") == "aop_plan_declined"
        ctx = THREAD_CTX.get("test_decline_partial_1", {})
        assert "_pending_aop" not in ctx


# ══════════════════════════════════════════════════════════════════════
# Scenario 5: Selection Methods
# ══════════════════════════════════════════════════════════════════════


class TestSelectionMethods:
    """Various ways to select a task from the menu."""

    def _setup(self, monkeypatch, tmp_path, tid):
        agent_a = TrackingAgent(
            "agent_a",
            responses=[{"answer": "A done.", "text": "A done.", "score": 0.9}],
            meta=_domain_agent_meta("Handles refund payment tasks", ["refund_payment"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            responses=[{"answer": "B done.", "text": "B done.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Handles account questions tasks", ["account_questions"]
            ),
        )

        subtasks = [
            "ACTION: refund payment processing",
            "INFORMATIONAL: account questions and info",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat(
            "Refund payment and account questions",
            context={"thread_id": tid},
        )
        return spine, agent_a, agent_b

    def test_numeric_selection(self, monkeypatch, tmp_path):
        """'1' should select the first pending task."""
        spine, _, _ = self._setup(monkeypatch, tmp_path, "test_sel_num")
        result = spine.handle_chat("1", context={"thread_id": "test_sel_num"})
        assert result.get("orchestration_pattern") == "aop_task_result"

    def test_ordinal_selection(self, monkeypatch, tmp_path):
        """'second' should select the second pending task."""
        spine, _, _ = self._setup(monkeypatch, tmp_path, "test_sel_ord")
        result = spine.handle_chat("second", context={"thread_id": "test_sel_ord"})
        assert result.get("orchestration_pattern") == "aop_task_result"

    def test_continue_selects_first(self, monkeypatch, tmp_path):
        """'next' should select the first pending task."""
        spine, _, _ = self._setup(monkeypatch, tmp_path, "test_sel_next")
        result = spine.handle_chat("next", context={"thread_id": "test_sel_next"})
        assert result.get("orchestration_pattern") == "aop_task_result"

    def test_yes_selects_first(self, monkeypatch, tmp_path):
        """'yes' should select the first pending task."""
        spine, _, _ = self._setup(monkeypatch, tmp_path, "test_sel_yes")
        result = spine.handle_chat("yes", context={"thread_id": "test_sel_yes"})
        assert result.get("orchestration_pattern") == "aop_task_result"


# ══════════════════════════════════════════════════════════════════════
# Scenario 6: Domain Agent Multi-Turn within AOP
# ══════════════════════════════════════════════════════════════════════


class TestDomainAgentMultiTurnInAOP:
    """Domain agent asking clarification during AOP subtask execution.

    This is the key test scenario: when a domain agent returns
    needs_input=True during AOP subtask execution, the agent must be
    pinned so that follow-up input routes to it (not re-decomposed).
    """

    def _make_clarifying_agent(self, agent_id):
        """Create an agent that asks clarification on first call, responds on second."""
        call_count = {"n": 0}

        def response_fn(request, call_num):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "answer": "What is your email for verification?",
                    "text": "What is your email for verification?",
                    "score": 0.7,
                    "needs_input": True,
                    "domain_agent_clarification": True,
                    "domain": "refunds",
                }
            return {
                "answer": "Refund processed successfully.",
                "text": "Refund processed successfully.",
                "score": 0.9,
            }

        return TrackingAgent(
            agent_id,
            responses=response_fn,
            meta=_domain_agent_meta(
                "Handles refund payment processing",
                ["refund_payment_processing"],
            ),
        )

    def test_domain_agent_clarification_pins_agent(self, monkeypatch, tmp_path):
        """When domain agent asks clarification within AOP, agent should be pinned."""
        refund = self._make_clarifying_agent("refund_agent")
        faq = TrackingAgent(
            "faq_agent",
            responses=[{"answer": "FAQ.", "text": "FAQ.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Answers account questions",
                ["account_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund payment",
            "INFORMATIONAL: account questions",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        # Menu
        spine.handle_chat(
            "Process my refund and answer my account question",
            context={"thread_id": "test_pin_1"},
        )
        # Select task 1 → domain agent asks clarification
        spine.handle_chat("1", context={"thread_id": "test_pin_1"})

        # Agent should be pinned
        ctx = THREAD_CTX.get("test_pin_1", {})
        assert ctx.get("pinned_agent_id") == "refund_agent"
        assert ctx.get("pinned_agent_type") == "domain_agent"
        assert ctx.get("pinned_terminal") is False
        # Plan should still be stored
        assert "_pending_aop" in ctx

    def test_followup_routes_to_pinned_domain_agent(self, monkeypatch, tmp_path):
        """User's answer should route to the pinned domain agent, not re-decompose."""
        refund = self._make_clarifying_agent("refund_agent")
        faq = TrackingAgent(
            "faq_agent",
            responses=[{"answer": "FAQ.", "text": "FAQ.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Answers account questions",
                ["account_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund payment",
            "INFORMATIONAL: account questions",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Refund and account question",
            context={"thread_id": "test_followup_1"},
        )
        spine.handle_chat("1", context={"thread_id": "test_followup_1"})
        # Answer clarification
        r3 = spine.handle_chat(
            "john@email.com", context={"thread_id": "test_followup_1"}
        )

        # Should have been handled by refund_agent (sticky route)
        assert r3.get("agent_id") == "refund_agent"
        # Agent called twice (clarification + response)
        assert len(refund.calls) == 2
        # FAQ agent should NOT have been called
        assert len(faq.calls) == 0

    def test_remaining_tasks_after_clarification_resolved(self, monkeypatch, tmp_path):
        """After domain agent resolves (unpin), remaining AOP tasks should be offered."""
        refund = self._make_clarifying_agent("refund_agent")
        faq = TrackingAgent(
            "faq_agent",
            responses=[{"answer": "FAQ.", "text": "FAQ.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Answers account questions",
                ["account_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund payment",
            "INFORMATIONAL: account questions",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Refund and account question",
            context={"thread_id": "test_remain_1"},
        )
        spine.handle_chat("1", context={"thread_id": "test_remain_1"})
        r3 = spine.handle_chat("john@email.com", context={"thread_id": "test_remain_1"})

        # After unpin, remaining tasks should be offered
        remaining = r3.get("remaining_subtasks")
        assert remaining is not None
        assert len(remaining) >= 1
        # Plan should still exist (one task remaining)
        ctx = THREAD_CTX.get("test_remain_1", {})
        assert "_pending_aop" in ctx

    def test_plan_preserved_during_clarification(self, monkeypatch, tmp_path):
        """While domain agent is pinned for clarification, pending plan stays intact."""
        refund = self._make_clarifying_agent("refund_agent")
        faq = TrackingAgent(
            "faq_agent",
            meta=_domain_agent_meta("Account questions", ["account_questions"]),
        )

        subtasks = [
            "ACTION: process refund payment",
            "INFORMATIONAL: account questions",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        spine.handle_chat(
            "Refund and account",
            context={"thread_id": "test_plan_preserve_1"},
        )
        spine.handle_chat("1", context={"thread_id": "test_plan_preserve_1"})

        # While pinned, plan should still be in context
        ctx = THREAD_CTX.get("test_plan_preserve_1", {})
        assert "_pending_aop" in ctx
        plan = ctx["_pending_aop"]
        # One subtask should have a result (executed), one should still be pending
        executed = [s for s in plan["subtasks"] if s.get("result") is not None]
        pending = [s for s in plan["subtasks"] if s.get("result") is None]
        assert len(executed) == 1
        assert len(pending) == 1


# ══════════════════════════════════════════════════════════════════════
# Scenario 7: Slot Propagation
# ══════════════════════════════════════════════════════════════════════


class TestSlotPropagation:
    """Slots from one task should be available to subsequent tasks."""

    def test_slots_accumulated_across_tasks(self, monkeypatch, tmp_path):
        """Slots from task 1 should be in context when task 2 executes."""
        agent_a = TrackingAgent(
            "agent_a",
            responses=[
                {
                    "answer": "Task A done.",
                    "text": "Task A done.",
                    "score": 0.9,
                    "slots": {"customer_id": "CUST-123", "email": "test@test.com"},
                }
            ],
            meta=_domain_agent_meta("Refund with slot handling", ["refund_with_slots"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            responses=[
                {"answer": "Task B done.", "text": "Task B done.", "score": 0.85}
            ],
            meta=_domain_agent_meta(
                "Complaint with slot handling", ["complaint_with_slots"]
            ),
        )

        subtasks = [
            "ACTION: refund with slots handling",
            "ACTION: complaint with slots handling",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat("Refund and complaint", context={"thread_id": "test_slots_1"})
        spine.handle_chat("1", context={"thread_id": "test_slots_1"})
        spine.handle_chat("1", context={"thread_id": "test_slots_1"})

        # Check accumulated slots in context
        ctx = THREAD_CTX.get("test_slots_1", {})
        accumulated = ctx.get("_accumulated_slots", {})
        assert accumulated.get("customer_id") == "CUST-123"
        assert accumulated.get("email") == "test@test.com"


# ══════════════════════════════════════════════════════════════════════
# Scenario 8: Full Lifecycle
# ══════════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """Complete multi-intent lifecycle with and without multi-turn."""

    def test_lifecycle_with_clarification(self, monkeypatch, tmp_path):
        """Full flow: menu → task 1 (clarification) → answer → remaining → task 2 → done."""
        refund_call = {"n": 0}

        def refund_response(request, call_num):
            refund_call["n"] += 1
            if refund_call["n"] == 1:
                return {
                    "answer": "Please provide your order number.",
                    "text": "Please provide your order number.",
                    "score": 0.7,
                    "needs_input": True,
                    "domain_agent_clarification": True,
                }
            return {
                "answer": "Your refund has been initiated.",
                "text": "Your refund has been initiated.",
                "score": 0.9,
                "slots": {"order_id": "ORD-456"},
            }

        refund = TrackingAgent(
            "refund_agent",
            responses=refund_response,
            meta=_domain_agent_meta(
                "Handles refund payment requests",
                ["refund_payment"],
            ),
        )
        faq = TrackingAgent(
            "faq_agent",
            responses=[
                {
                    "answer": "To open an account, you need ID and proof of address.",
                    "text": "To open an account, you need ID and proof of address.",
                    "score": 0.85,
                }
            ],
            meta=_domain_agent_meta(
                "Answers account opening questions",
                ["account_opening_questions"],
            ),
        )

        subtasks = [
            "ACTION: process refund payment",
            "INFORMATIONAL: account opening questions",
        ]

        spine, _, store = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        tid = "test_full_lifecycle_1"

        # Turn 1: Multi-intent → menu
        r1 = spine.handle_chat(
            "I need a refund for my payment and how do I open an account?",
            context={"thread_id": tid},
        )
        assert r1.get("orchestration_pattern") == "aop_task_menu"
        assert len(r1["task_menu"]) == 2
        assert len(refund.calls) == 0
        assert len(faq.calls) == 0

        # Turn 2: Select task 1 → refund agent asks clarification
        r2 = spine.handle_chat("1", context={"thread_id": tid})
        assert r2.get("orchestration_pattern") == "aop_task_result"
        ctx = THREAD_CTX.get(tid, {})
        assert ctx.get("pinned_agent_id") == "refund_agent"
        assert len(refund.calls) == 1

        # Turn 3: Answer clarification → refund agent completes
        r3 = spine.handle_chat("ORD-456", context={"thread_id": tid})
        assert r3.get("agent_id") == "refund_agent"
        assert len(refund.calls) == 2
        # Agent should be unpinned
        ctx = THREAD_CTX.get(tid, {})
        assert ctx.get("pinned_agent_id") is None
        # Remaining tasks offered
        remaining = r3.get("remaining_subtasks")
        assert remaining is not None and len(remaining) >= 1
        assert "_pending_aop" in ctx

        # Turn 4: Select remaining task → faq agent answers
        r4 = spine.handle_chat("1", context={"thread_id": tid})
        assert r4.get("orchestration_pattern") == "aop_task_result"
        assert len(faq.calls) == 1
        # Plan should be complete
        ctx = THREAD_CTX.get(tid, {})
        assert "_pending_aop" not in ctx
        assert len(store.query()) == 2

    def test_lifecycle_without_clarification(self, monkeypatch, tmp_path):
        """Simple lifecycle: menu → task 1 → remaining → task 2 → done."""
        refund = TrackingAgent(
            "refund_agent",
            responses=[
                {
                    "answer": "Refund processed.",
                    "text": "Refund processed.",
                    "score": 0.9,
                }
            ],
            meta=_domain_agent_meta("Refund payment processing", ["refund_payment"]),
        )
        faq = TrackingAgent(
            "faq_agent",
            responses=[
                {
                    "answer": "Account opening info.",
                    "text": "Account opening info.",
                    "score": 0.85,
                }
            ],
            meta=_domain_agent_meta("Account opening questions", ["account_opening"]),
        )

        subtasks = ["ACTION: refund payment", "INFORMATIONAL: account opening"]

        spine, _, store = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"refund_agent": refund, "faq_agent": faq},
            subtasks,
        )

        tid = "test_simple_lifecycle_1"

        r1 = spine.handle_chat(
            "Refund payment and account opening",
            context={"thread_id": tid},
        )
        assert r1.get("orchestration_pattern") == "aop_task_menu"

        r2 = spine.handle_chat("1", context={"thread_id": tid})
        assert r2.get("orchestration_pattern") == "aop_task_result"
        assert r2.get("remaining_subtasks") is not None

        r3 = spine.handle_chat("1", context={"thread_id": tid})
        assert r3.get("orchestration_pattern") == "aop_task_result"

        ctx = THREAD_CTX.get(tid, {})
        assert "_pending_aop" not in ctx
        assert len(store.query()) == 2


# ══════════════════════════════════════════════════════════════════════
# Scenario 9: Three-Task Lifecycle
# ══════════════════════════════════════════════════════════════════════


class TestThreeTaskLifecycle:
    """Lifecycle with 3 subtasks — ensures proper indexing across executions."""

    def test_three_tasks_sequential(self, monkeypatch, tmp_path):
        """Execute 3 tasks sequentially, verifying remaining count after each."""
        agent_a = TrackingAgent(
            "agent_a",
            responses=[{"answer": "A done.", "text": "A done.", "score": 0.9}],
            meta=_domain_agent_meta(
                "Handles refund processing tasks", ["refund_processing"]
            ),
        )
        agent_b = TrackingAgent(
            "agent_b",
            responses=[{"answer": "B done.", "text": "B done.", "score": 0.85}],
            meta=_domain_agent_meta(
                "Handles complaint handling tasks", ["complaint_handling"]
            ),
        )
        agent_c = TrackingAgent(
            "agent_c",
            responses=[{"answer": "C done.", "text": "C done.", "score": 0.8}],
            meta=_domain_agent_meta(
                "Handles account inquiry tasks", ["account_inquiry"]
            ),
        )

        subtasks = [
            "ACTION: refund processing task",
            "ACTION: complaint handling task",
            "INFORMATIONAL: account inquiry task",
        ]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b, "agent_c": agent_c},
            subtasks,
        )

        tid = "test_three_1"

        r1 = spine.handle_chat(
            "Refund, complaint, and account inquiry",
            context={"thread_id": tid},
        )
        assert len(r1.get("task_menu", [])) == 3

        r2 = spine.handle_chat("1", context={"thread_id": tid})
        remaining2 = r2.get("remaining_subtasks", [])
        assert len(remaining2) == 2

        r3 = spine.handle_chat("1", context={"thread_id": tid})
        remaining3 = r3.get("remaining_subtasks", [])
        assert len(remaining3) == 1

        spine.handle_chat("1", context={"thread_id": tid})
        ctx = THREAD_CTX.get(tid, {})
        assert "_pending_aop" not in ctx


# ══════════════════════════════════════════════════════════════════════
# Scenario 10: Edge Cases
# ══════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases in multi-intent handling."""

    def test_single_subtask_executes_immediately(self, monkeypatch, tmp_path):
        """When decomposition returns only 1 subtask, execute immediately (no menu)."""
        agent = TrackingAgent(
            "agent_a",
            responses=[{"answer": "Done.", "text": "Done.", "score": 0.9}],
            meta=_domain_agent_meta("Handles refund tasks", ["refund"]),
        )

        subtasks = ["ACTION: single refund task"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent},
            subtasks,
        )

        result = spine.handle_chat(
            "Process a refund",
            context={"thread_id": "test_edge_1"},
        )

        # Should NOT show menu — should execute immediately
        assert result.get("orchestration_pattern") != "aop_task_menu"
        assert len(agent.calls) >= 1

    def test_performance_feedback_recorded(self, monkeypatch, tmp_path):
        """Each executed subtask should create a performance record."""
        agent_a = TrackingAgent(
            "agent_a",
            responses=[{"answer": "A.", "text": "A.", "score": 0.9}],
            meta=_domain_agent_meta("Refund processing", ["refund_processing"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            responses=[{"answer": "B.", "text": "B.", "score": 0.85}],
            meta=_domain_agent_meta("Complaint handling", ["complaint_handling"]),
        )

        subtasks = ["ACTION: refund processing", "ACTION: complaint handling"]

        spine, _, store = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat("Refund and complaint", context={"thread_id": "test_edge_2"})
        spine.handle_chat("1", context={"thread_id": "test_edge_2"})
        assert len(store.query()) == 1

        spine.handle_chat("1", context={"thread_id": "test_edge_2"})
        assert len(store.query()) == 2

    def test_quick_replies_include_no_thanks(self, monkeypatch, tmp_path):
        """Quick replies after execution should include a 'No thanks' option."""
        agent_a = TrackingAgent(
            "agent_a",
            responses=[{"answer": "Done.", "text": "Done.", "score": 0.9}],
            meta=_domain_agent_meta("Refund processing", ["refund_processing"]),
        )
        agent_b = TrackingAgent(
            "agent_b",
            meta=_domain_agent_meta("Complaint handling", ["complaint_handling"]),
        )

        subtasks = ["ACTION: refund processing", "ACTION: complaint handling"]

        spine, _, _ = build_multi_intent_spine(
            tmp_path,
            monkeypatch,
            {"agent_a": agent_a, "agent_b": agent_b},
            subtasks,
        )

        spine.handle_chat("Refund and complaint", context={"thread_id": "test_edge_3"})
        result = spine.handle_chat("1", context={"thread_id": "test_edge_3"})

        # Check quick_replies exist and contain "No thanks"
        chat = result.get("chat", {})
        qr = chat.get("quick_replies", [])
        has_no_thanks = any("no thanks" in r.lower() for r in qr)
        assert has_no_thanks, f"Expected 'No thanks' in quick_replies, got: {qr}"
