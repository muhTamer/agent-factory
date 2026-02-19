# tests/test_solvability_estimator.py
from __future__ import annotations

from app.orchestration.performance_store import ExecutionRecord, PerformanceStore
from app.orchestration.solvability_estimator import SolvabilityEstimator


def _make_store(tmp_path) -> PerformanceStore:
    return PerformanceStore(path=str(tmp_path / "perf.json"))


def _catalog():
    """Sample agent catalog (mimics registry.all_meta() output)."""
    return {
        "refund_agent": {
            "id": "refund_agent",
            "type": "workflow_runner",
            "description": "Handles refund requests and processes returns",
            "capabilities": ["refund_processing", "return_handling", "payment_reversal"],
        },
        "faq_agent": {
            "id": "faq_agent",
            "type": "faq_rag",
            "description": "Answers customer FAQs about policies and products",
            "capabilities": ["faq_answering", "policy_lookup", "knowledge_base_search"],
        },
        "account_agent": {
            "id": "account_agent",
            "type": "tool_operator",
            "description": "Manages customer account updates like email and address changes",
            "capabilities": ["account_update", "email_change", "address_change"],
        },
    }


def test_textual_similarity_high_overlap(tmp_path):
    """Subtask about refunds should score highest with refund agent."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund for order #123"], _catalog())

    assert result.assignments["Process refund for order #123"] == "refund_agent"
    assert result.assignment_scores["Process refund for order #123"] > 0.0


def test_textual_similarity_faq_match(tmp_path):
    """FAQ question should match FAQ agent."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(
        ["Search the knowledge base for frequently asked questions about products"], _catalog()
    )

    assert (
        result.assignments[
            "Search the knowledge base for frequently asked questions about products"
        ]
        == "faq_agent"
    )


def test_textual_similarity_account_match(tmp_path):
    """Account update should match account agent."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(["Update my email address"], _catalog())

    assert result.assignments["Update my email address"] == "account_agent"


def test_multiple_subtasks_independent(tmp_path):
    """Each subtask gets an independent assignment."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    subtasks = [
        "Process refund for order #123",
        "Search knowledge base for frequently asked questions about products",
    ]
    result = est.estimate(subtasks, _catalog())

    assert len(result.assignments) == 2
    assert result.assignments[subtasks[0]] == "refund_agent"
    assert result.assignments[subtasks[1]] == "faq_agent"


def test_combined_score_formula(tmp_path):
    """Verify: combined = alpha * textual + beta * historical."""
    store = _make_store(tmp_path)
    alpha, beta = 0.6, 0.4
    est = SolvabilityEstimator(store, alpha=alpha, beta=beta)
    result = est.estimate(["Process refund"], _catalog())

    for score in result.scores:
        expected = alpha * score.textual_similarity + beta * score.historical_performance
        assert (
            abs(score.combined_score - expected) < 1e-6
        ), f"Expected {expected}, got {score.combined_score}"


def test_historical_performance_no_history(tmp_path):
    """No history -> neutral prior 0.5 for historical component."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund"], _catalog())

    for score in result.scores:
        assert score.historical_performance == 0.5


def test_historical_performance_with_records(tmp_path):
    """Historical performance should reflect stored records."""
    store = _make_store(tmp_path)
    store.append(ExecutionRecord("refund_agent", "refund task", True, 0.9, 100))
    store.append(ExecutionRecord("refund_agent", "refund task 2", True, 0.7, 150))

    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund"], _catalog())

    refund_scores = [s for s in result.scores if s.agent_id == "refund_agent"]
    assert len(refund_scores) == 1
    assert refund_scores[0].historical_performance == 0.8  # (0.9 + 0.7) / 2


def test_history_boosts_assignment(tmp_path):
    """Good history should boost an agent's combined score."""
    store = _make_store(tmp_path)
    # Give faq_agent excellent history on a "refund" task
    for _ in range(5):
        store.append(ExecutionRecord("faq_agent", "refund related", True, 1.0, 50))
    # Give refund_agent poor history
    for _ in range(5):
        store.append(ExecutionRecord("refund_agent", "refund related", False, 0.1, 500))

    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund"], _catalog())

    # Despite textual similarity favoring refund_agent, history should push faq_agent up
    faq_scores = [s for s in result.scores if s.agent_id == "faq_agent"]
    refund_scores = [s for s in result.scores if s.agent_id == "refund_agent"]
    assert len(faq_scores) == 1
    assert len(refund_scores) == 1
    # faq_agent history = 1.0, refund_agent history = 0.1
    assert faq_scores[0].historical_performance > refund_scores[0].historical_performance


def test_empty_subtasks(tmp_path):
    """Empty subtasks -> empty result."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate([], _catalog())

    assert result.assignments == {}
    assert result.scores == []


def test_empty_catalog(tmp_path):
    """Empty catalog -> empty result."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund"], {})

    assert result.assignments == {}
    assert result.scores == []


def test_scores_include_all_pairs(tmp_path):
    """All (subtask, agent) pairs should be scored."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    catalog = _catalog()
    subtasks = ["task1", "task2"]
    result = est.estimate(subtasks, catalog)

    assert len(result.scores) == len(subtasks) * len(catalog)


def test_reasoning_string_populated(tmp_path):
    """Each score should have a non-empty reasoning string."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store)
    result = est.estimate(["Process refund"], _catalog())

    for score in result.scores:
        assert score.reasoning
        assert "textual=" in score.reasoning
        assert "historical=" in score.reasoning
        assert "combined=" in score.reasoning


def test_custom_alpha_beta(tmp_path):
    """Custom weights should be applied."""
    store = _make_store(tmp_path)
    est = SolvabilityEstimator(store, alpha=1.0, beta=0.0)
    result = est.estimate(["Process refund"], _catalog())

    for score in result.scores:
        # With beta=0, historical_performance shouldn't affect combined
        assert abs(score.combined_score - score.textual_similarity) < 1e-6
