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


def test_spine_aop_path(monkeypatch, tmp_path):
    """Multi-intent query should go through AOP coordinator."""
    spine, _, store = _build_spine_with_aop(tmp_path, monkeypatch)

    # Mock: classification returns hierarchical_delegation
    monkeypatch.setattr(
        "app.llm_client.chat_json",
        lambda **_kw: {"pattern": "hierarchical_delegation"},
    )

    # Mock: AOP decomposition + completeness (both via chat_json in aop_coordinator)
    def mock_aop_chat_json(**kwargs):
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
        return {"pattern": "hierarchical_delegation"}

    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock_aop_chat_json)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock_aop_chat_json)

    result = spine.handle_chat(
        "I need a refund for order #123 AND what is the return window?",
        context={"thread_id": "test_aop"},
    )

    assert "error" not in result
    assert result.get("orchestration_pattern") == "hierarchical_delegation"
    assert "subtask_results" in result
    assert len(result["subtask_results"]) == 2

    # Feedback loop should have written records
    records = store.query()
    assert len(records) == 2


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
