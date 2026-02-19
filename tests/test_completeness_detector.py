# tests/test_completeness_detector.py
from __future__ import annotations

from app.orchestration.completeness_detector import CompletenessDetector


def _mock_chat_json_complete(**_kw):
    """LLM returns a complete plan."""
    return _kw  # ignored; we monkeypatch to return the value below


def test_complete_plan(monkeypatch):
    """Plan covering all query aspects should return complete=True."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: {
            "complete": True,
            "missing": [],
            "redundant": [],
            "coverage_ratio": 1.0,
            "reasoning": "All aspects covered.",
        },
    )
    det = CompletenessDetector()
    result = det.check(
        query="I want a refund and update my email",
        subtasks=["Process refund", "Update email"],
        assignments={"Process refund": "refund_agent", "Update email": "account_agent"},
    )

    assert result.complete is True
    assert result.missing == []
    assert result.redundant == []
    assert result.coverage_ratio == 1.0


def test_incomplete_plan(monkeypatch):
    """Plan missing an aspect should return complete=False with missing list."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: {
            "complete": False,
            "missing": ["Email update not addressed"],
            "redundant": [],
            "coverage_ratio": 0.5,
            "reasoning": "Only refund is covered, email update missing.",
        },
    )
    det = CompletenessDetector()
    result = det.check(
        query="I want a refund and update my email",
        subtasks=["Process refund"],
        assignments={"Process refund": "refund_agent"},
    )

    assert result.complete is False
    assert len(result.missing) == 1
    assert "Email" in result.missing[0]
    assert result.coverage_ratio == 0.5


def test_redundant_subtasks(monkeypatch):
    """Overlapping subtasks should appear in redundant list."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: {
            "complete": True,
            "missing": [],
            "redundant": [["Process refund for order", "Handle return for order"]],
            "coverage_ratio": 1.0,
            "reasoning": "Refund and return subtasks overlap.",
        },
    )
    det = CompletenessDetector()
    result = det.check(
        query="Process my return",
        subtasks=["Process refund for order", "Handle return for order"],
        assignments={
            "Process refund for order": "refund_agent",
            "Handle return for order": "refund_agent",
        },
    )

    assert result.complete is True
    assert len(result.redundant) == 1
    assert result.redundant[0] == ("Process refund for order", "Handle return for order")


def test_empty_subtasks():
    """Empty subtask list should return complete=False."""
    det = CompletenessDetector()
    result = det.check(
        query="I want a refund",
        subtasks=[],
        assignments={},
    )

    assert result.complete is False
    assert len(result.missing) > 0
    assert result.coverage_ratio == 0.0


def test_malformed_llm_response(monkeypatch):
    """Graceful handling of unexpected LLM response format."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: {"raw": "something unexpected"},
    )
    det = CompletenessDetector()
    result = det.check(
        query="I want a refund",
        subtasks=["Process refund"],
        assignments={"Process refund": "refund_agent"},
    )

    # Should not crash — defaults to complete with empty missing
    assert isinstance(result.complete, bool)
    assert isinstance(result.missing, list)
    assert isinstance(result.redundant, list)


def test_llm_exception_fail_open(monkeypatch):
    """If LLM call raises, fail-open as complete."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("LLM down")),
    )
    det = CompletenessDetector()
    result = det.check(
        query="I want a refund",
        subtasks=["Process refund"],
        assignments={"Process refund": "refund_agent"},
    )

    assert result.complete is True
    assert "LLM unavailable" in result.reasoning


def test_coverage_ratio_clamped(monkeypatch):
    """Coverage ratio should be clamped to [0.0, 1.0]."""
    monkeypatch.setattr(
        "app.orchestration.completeness_detector.chat_json",
        lambda **_kw: {
            "complete": True,
            "missing": [],
            "redundant": [],
            "coverage_ratio": 1.5,  # Invalid — should be clamped
            "reasoning": "Over-reported coverage.",
        },
    )
    det = CompletenessDetector()
    result = det.check(
        query="Test",
        subtasks=["subtask"],
        assignments={"subtask": "agent"},
    )

    assert result.coverage_ratio == 1.0
