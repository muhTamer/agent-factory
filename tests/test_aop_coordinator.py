# tests/test_aop_coordinator.py
from __future__ import annotations

from typing import Any, Dict

from app.orchestration.aop_coordinator import AOPCoordinator
from app.orchestration.performance_store import PerformanceStore
from app.runtime.registry import AgentRegistry


# ── Stub Agent ──────────────────────────────────────────────────────


class StubAgent:
    """Minimal IAgent for testing — returns canned responses."""

    def __init__(self, agent_id: str, response: Dict[str, Any]):
        self._id = agent_id
        self._response = response
        self._meta: Dict[str, Any] = {}

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._response)

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            **self._meta,
        }


def _build_registry() -> AgentRegistry:
    """Create a registry with two stub agents."""
    registry = AgentRegistry()

    refund = StubAgent("refund_agent", {"answer": "Refund processed for your order.", "score": 0.9})
    refund._meta = {
        "type": "workflow_runner",
        "description": "Handles refund requests and processes returns",
        "capabilities": ["refund_processing", "return_handling", "payment_reversal"],
        "ready": True,
    }
    registry.register("refund_agent", refund, refund.metadata())

    faq = StubAgent("faq_agent", {"answer": "Our return window is 30 days.", "score": 0.85})
    faq._meta = {
        "type": "faq_rag",
        "description": "Answers customer FAQs about policies and products",
        "capabilities": ["faq_answering", "policy_lookup", "knowledge_base_search"],
        "ready": True,
    }
    registry.register("faq_agent", faq, faq.metadata())

    return registry


def _mock_decompose_response(**_kw):
    """Mock LLM: returns two subtasks."""
    return {"subtasks": ["Process refund for order #123", "Answer FAQ about return window"]}


def _mock_completeness_complete(**_kw):
    """Mock LLM: plan is complete."""
    return {
        "complete": True,
        "missing": [],
        "redundant": [],
        "coverage_ratio": 1.0,
        "reasoning": "All aspects covered.",
    }


def _mock_completeness_incomplete(**_kw):
    """Mock LLM: plan is incomplete."""
    return {
        "complete": False,
        "missing": ["FAQ question not addressed"],
        "redundant": [],
        "coverage_ratio": 0.5,
        "reasoning": "Missing FAQ coverage.",
    }


def _build_mock_chat_json(decompose_resp, completeness_resp):
    """
    Build a mock chat_json that routes based on the system prompt content.
    Decompose calls contain 'task decomposition', completeness calls contain 'completeness'.
    """
    call_count = {"decompose": 0, "completeness": 0}

    def mock(**kwargs):
        messages = kwargs.get("messages", [])
        system_msg = ""
        for m in messages:
            if m.get("role") == "system":
                system_msg = m.get("content", "")
                break

        if "decomposition" in system_msg.lower():
            call_count["decompose"] += 1
            resp = decompose_resp
            if callable(resp):
                return resp(call_count["decompose"])
            return resp

        if "completeness" in system_msg.lower():
            call_count["completeness"] += 1
            resp = completeness_resp
            if callable(resp):
                return resp(call_count["completeness"])
            return resp

        # Default: return decompose response
        return decompose_resp if not callable(decompose_resp) else decompose_resp(1)

    return mock


def test_full_aop_cycle(monkeypatch, tmp_path):
    """Complete 5-step cycle with mocked LLM and stub agents."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        _mock_decompose_response(),
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    result = aop.orchestrate("I need a refund for order #123 and what is the return window?", {})

    assert "text" in result
    assert result["orchestration_pattern"] == "hierarchical_delegation"
    assert len(result["subtask_results"]) == 2
    assert result["completeness"]["complete"] is True


def test_feedback_loop_writes(monkeypatch, tmp_path):
    """After orchestrate(), performance store should have new records."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        _mock_decompose_response(),
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    aop.orchestrate("Refund and FAQ", {})

    records = store.query()
    assert len(records) == 2  # One per subtask


def test_single_subtask(monkeypatch, tmp_path):
    """Single subtask should still work (degenerate case)."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        {"subtasks": ["Process refund for order #123"]},
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    result = aop.orchestrate("I need a refund", {})

    assert len(result["subtask_results"]) == 1


def test_empty_catalog(monkeypatch, tmp_path):
    """No agents -> error response."""
    registry = AgentRegistry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    aop = AOPCoordinator(registry=registry, performance_store=store)

    result = aop.orchestrate("I need a refund", {})
    assert "error" in result


def test_decompose_failure(monkeypatch, tmp_path):
    """If decomposition fails, return error."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        {"subtasks": []},  # Empty decomposition
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    result = aop.orchestrate("I need a refund", {})

    assert "error" in result


def test_agent_not_found_graceful(monkeypatch, tmp_path):
    """If assigned agent is not in registry, handle gracefully."""
    registry = AgentRegistry()
    # Register only one agent, but decompose into subtask that maps to non-existent
    faq = StubAgent("faq_agent", {"answer": "FAQ answer", "score": 0.8})
    faq._meta = {
        "type": "faq_rag",
        "description": "FAQ agent",
        "capabilities": ["faq_answering"],
        "ready": True,
    }
    registry.register("faq_agent", faq, faq.metadata())

    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        {"subtasks": ["Process refund", "Answer FAQ"]},
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    result = aop.orchestrate("Refund and FAQ", {})

    # Should not crash — some subtasks may fail but others succeed
    assert "text" in result
    assert len(result["subtask_results"]) == 2


def test_solvability_scores_in_result(monkeypatch, tmp_path):
    """Result should include solvability metadata."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    mock = _build_mock_chat_json(
        _mock_decompose_response(),
        _mock_completeness_complete(),
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store)
    result = aop.orchestrate("Refund and FAQ", {})

    assert "solvability" in result
    assert "assignments" in result["solvability"]
    assert "assignment_scores" in result["solvability"]


def test_incomplete_triggers_redecompose(monkeypatch, tmp_path):
    """If completeness check fails, re-decomposition should be attempted."""
    registry = _build_registry()
    store = PerformanceStore(path=str(tmp_path / "perf.json"))

    def completeness_flip(call_count):
        """First call incomplete, second call complete."""
        if call_count <= 1:
            return {
                "complete": False,
                "missing": ["FAQ not addressed"],
                "redundant": [],
                "coverage_ratio": 0.5,
                "reasoning": "Missing FAQ.",
            }
        return _mock_completeness_complete()

    mock = _build_mock_chat_json(
        _mock_decompose_response(),
        completeness_flip,
    )
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock)

    aop = AOPCoordinator(registry=registry, performance_store=store, max_retries=1)
    result = aop.orchestrate("Refund and FAQ", {})

    # Should complete (after retry)
    assert "text" in result
    assert result["orchestration_pattern"] == "hierarchical_delegation"
