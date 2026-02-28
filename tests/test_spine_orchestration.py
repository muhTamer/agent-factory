# tests/test_spine_orchestration.py
"""
Integration tests for AOP orchestration path in RuntimeSpine.
"""
from __future__ import annotations

from typing import Any, Dict

from app.orchestration.aop_coordinator import AOPCoordinator
from app.orchestration.performance_store import PerformanceStore
from app.runtime.guardrails import NoOpGuardrails
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Candidate, RoutePlan
from app.runtime.spine import RuntimeSpine


# ── Helpers ──────────────────────────────────────────────────────────


class StubAgent:
    """Minimal IAgent for testing."""

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


class FixedRouter:
    """A test router that always returns a fixed primary agent."""

    def __init__(self, primary: str):
        self._primary = primary

    def route(self, query: str) -> RoutePlan:
        return RoutePlan(
            primary=self._primary,
            strategy="single",
            candidates=[Candidate(id=self._primary, score=1.0, reason="fixed")],
        )


def _build_spine_with_aop(tmp_path, monkeypatch, classify_result="direct"):
    """Build a RuntimeSpine with AOP coordinator and mocked LLM."""
    registry = AgentRegistry()

    refund = StubAgent("refund_agent", {"answer": "Refund processed.", "score": 0.9})
    refund._meta = {
        "type": "workflow_runner",
        "description": "Handles refund requests",
        "capabilities": ["refund_processing", "return_handling"],
        "ready": True,
    }
    registry.register("refund_agent", refund, refund.metadata())

    faq = StubAgent("faq_agent", {"answer": "Our return window is 30 days.", "score": 0.85})
    faq._meta = {
        "type": "faq_rag",
        "description": "Answers customer FAQs about policies",
        "capabilities": ["faq_answering", "policy_lookup"],
        "ready": True,
    }
    registry.register("faq_agent", faq, faq.metadata())

    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    aop = AOPCoordinator(registry=registry, performance_store=store)
    router = FixedRouter("refund_agent")

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
    )

    return spine, registry, store


# ── Tests ────────────────────────────────────────────────────────────


def test_classify_direct_pattern(monkeypatch, tmp_path):
    """Single-intent query should classify as 'direct'."""
    spine, _, _ = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "app.llm_client.chat_json",
        lambda **_kw: {"pattern": "direct"},
    )

    pattern = spine._classify_orchestration_pattern("What is your refund policy?")
    assert pattern == "direct"


def test_classify_hierarchical_pattern(monkeypatch, tmp_path):
    """Multi-intent query should classify as 'hierarchical_delegation'."""
    spine, _, _ = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "app.llm_client.chat_json",
        lambda **_kw: {"pattern": "hierarchical_delegation"},
    )

    pattern = spine._classify_orchestration_pattern(
        "I need a refund for order #123 AND what is the return window?"
    )
    assert pattern == "hierarchical_delegation"


def test_classify_defaults_to_direct_on_error(monkeypatch, tmp_path):
    """On LLM failure, default to 'direct'."""
    spine, _, _ = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr(
        "app.llm_client.chat_json",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("LLM down")),
    )

    pattern = spine._classify_orchestration_pattern("Any query")
    assert pattern == "direct"


def _mock_aop_chat_json(**kwargs):
    """Shared mock for AOP decomposition + completeness LLM calls."""
    messages = kwargs.get("messages", [])
    system_msg = ""
    for m in messages:
        if m.get("role") == "system":
            system_msg = m.get("content", "")
            break

    if "decomposition" in system_msg.lower():
        return {"subtasks": ["Process refund", "Answer FAQ"]}
    if "completeness" in system_msg.lower():
        return {
            "complete": True,
            "missing": [],
            "redundant": [],
            "coverage_ratio": 1.0,
            "reasoning": "All covered.",
        }
    # Voice rendering and pattern classification
    if "customer-service chat voice" in system_msg.lower():
        return {
            "messages": ["I can help with 2 tasks."],
            "quick_replies": ["1. Process refund", "2. Answer FAQ"],
        }
    return {"pattern": "hierarchical_delegation"}


def test_spine_aop_path_multi_subtask_presents_menu(monkeypatch, tmp_path):
    """Multi-subtask query should present a task menu, NOT execute immediately."""
    spine, _, store = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr("app.llm_client.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", _mock_aop_chat_json)

    result = spine.handle_chat(
        "I need a refund for order #123 AND what is the return window?",
        context={"thread_id": "test_aop_menu"},
    )

    assert "error" not in result
    assert result.get("orchestration_pattern") == "aop_task_menu"
    assert "task_menu" in result
    assert len(result["task_menu"]) == 2

    # No execution yet — performance store should be empty
    records = store.query()
    assert len(records) == 0


def test_spine_aop_task_selection(monkeypatch, tmp_path):
    """User selecting a task from the menu should execute only that task."""

    spine, _, store = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr("app.llm_client.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", _mock_aop_chat_json)

    # Step 1: Present menu
    result1 = spine.handle_chat(
        "I need a refund for order #123 AND what is the return window?",
        context={"thread_id": "test_aop_select"},
    )
    assert result1.get("orchestration_pattern") == "aop_task_menu"

    # Step 2: User selects task "1"
    result2 = spine.handle_chat(
        "1",
        context={"thread_id": "test_aop_select"},
    )
    assert "error" not in result2
    assert result2.get("orchestration_pattern") == "aop_task_result"
    assert "executed_subtask" in result2

    # One task executed → one feedback record
    records = store.query()
    assert len(records) == 1


def test_spine_aop_decline_clears_plan(monkeypatch, tmp_path):
    """User declining should clear pending plan."""
    from app.runtime.spine import THREAD_CTX

    spine, _, _ = _build_spine_with_aop(tmp_path, monkeypatch)

    monkeypatch.setattr("app.llm_client.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", _mock_aop_chat_json)

    # Present menu
    spine.handle_chat(
        "I need a refund for order #123 AND what is the return window?",
        context={"thread_id": "test_decline"},
    )

    # Decline
    result = spine.handle_chat(
        "no thanks",
        context={"thread_id": "test_decline"},
    )
    assert result.get("orchestration_pattern") == "aop_plan_declined"

    # Plan should be cleared from context
    ctx = THREAD_CTX.get("test_decline", {})
    assert "_pending_aop" not in ctx


def test_spine_aop_clarification_preserves_plan(monkeypatch, tmp_path):
    """When a subtask returns a clarification, the user's follow-up
    should NOT clear the pending AOP plan — it should route to the
    pinned agent via sticky routing, preserving remaining tasks."""
    from app.runtime.spine import THREAD_CTX

    spine, registry, store = _build_spine_with_aop(tmp_path, monkeypatch)

    # Override faq_agent to return a clarification on first call,
    # then a final answer on the second call.
    call_count = {"n": 0}
    _orig_faq = registry.get("faq_agent")

    class ClarifyingAgent:
        def __init__(self):
            self._id = "faq_agent"
            self._meta = _orig_faq._meta

        def load(self, spec):
            pass

        def handle(self, req):
            call_count["n"] += 1
            ctx = req.get("context", {})
            if call_count["n"] == 1:
                # First call: ask clarification, pin ourselves
                ctx["pinned_agent_id"] = "faq_agent"
                ctx["pinned_agent_type"] = "rag_fsm"
                ctx["pinned_terminal"] = False
                return {
                    "answer": "Which type of account?",
                    "score": 0.0,
                    "rag_state": "clarify",
                    "rag_clarification": True,
                }
            else:
                # Second call: final answer, signal unpin
                ctx.pop("pinned_agent_id", None)
                ctx.pop("pinned_agent_type", None)
                ctx.pop("pinned_terminal", None)
                return {
                    "answer": "You need: ID, proof of address.",
                    "score": 0.85,
                    "rag_answered": True,
                }

        def metadata(self):
            return {"id": self._id, **self._meta}

    clarifying = ClarifyingAgent()
    registry.register("faq_agent", clarifying, clarifying.metadata())

    monkeypatch.setattr("app.llm_client.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", _mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", _mock_aop_chat_json)

    # Step 1: Present task menu
    r1 = spine.handle_chat(
        "What docs for Current Account? Also refund order #4821",
        context={"thread_id": "test_clarify"},
    )
    assert r1.get("orchestration_pattern") == "aop_task_menu"

    # Step 2: User selects task "2" (Answer FAQ — maps to faq_agent)
    r2 = spine.handle_chat("2", context={"thread_id": "test_clarify"})
    assert r2.get("orchestration_pattern") == "aop_task_result"
    # Agent asked clarification → should be pinned now
    ctx = THREAD_CTX.get("test_clarify", {})
    assert ctx.get("pinned_agent_id") == "faq_agent"
    # Plan must still be in context
    assert "_pending_aop" in ctx

    # Step 3: User answers clarification
    r3 = spine.handle_chat("Individual account", context={"thread_id": "test_clarify"})
    # Should have routed to the pinned faq_agent (sticky route), NOT cleared the plan
    assert "error" not in r3
    assert r3.get("agent_id") == "faq_agent"
    # Plan should STILL exist (1 remaining task)
    ctx = THREAD_CTX.get("test_clarify", {})
    assert "_pending_aop" in ctx
    # After the agent provides a final answer, remaining tasks should be offered
    remaining = r3.get("remaining_subtasks")
    assert remaining is not None and len(remaining) >= 1


def test_spine_direct_path_unchanged(monkeypatch, tmp_path):
    """Single-intent query should follow existing direct routing path."""
    spine, _, _ = _build_spine_with_aop(tmp_path, monkeypatch)

    # Mock: classification returns direct
    monkeypatch.setattr(
        "app.llm_client.chat_json",
        lambda **_kw: {"pattern": "direct"},
    )

    result = spine.handle_chat(
        "What is your refund policy?",
        context={"thread_id": "test_direct"},
    )

    # Should use direct routing (fixed router -> refund_agent)
    assert "error" not in result
    assert result.get("agent_id") == "refund_agent"
    # Should NOT have AOP metadata
    assert (
        "orchestration_pattern" not in result
        or result.get("orchestration_pattern") != "hierarchical_delegation"
    )


def test_spine_no_aop_coordinator(tmp_path):
    """Spine without AOP coordinator should work normally (backward compat)."""
    registry = AgentRegistry()
    agent = StubAgent("agent_a", {"answer": "Hello", "score": 1.0})
    agent._meta = {"type": "faq", "description": "Test", "capabilities": [], "ready": True}
    registry.register("agent_a", agent, agent.metadata())

    router = FixedRouter("agent_a")
    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        # No aop_coordinator
    )

    result = spine.handle_chat("Hello", context={"thread_id": "test_noaop"})
    assert "error" not in result
    assert result.get("agent_id") == "agent_a"
