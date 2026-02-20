# evaluation/mock_factory.py
"""
Deterministic LLM mock factory for evaluation scenarios.

Dispatches by system-prompt keywords (same pattern as test_spine_orchestration.py).
Each scenario carries its own mock_responses dict so evaluation is fully reproducible
without API calls.
"""
from __future__ import annotations

from typing import Any, Dict


def build_scenario_mock(mock_responses: Dict[str, Any]):
    """
    Return a chat_json replacement that dispatches by system-prompt content.

    mock_responses keys:
      - "classify"     → spine classification (direct / hierarchical_delegation)
      - "decompose"    → AOP decomposition {"subtasks": [...]}
      - "completeness" → completeness audit {"complete": ..., "missing": ...}
      - "workflow"     → workflow mapper {"event": ..., "slots": ...}

    Unmatched calls return a safe fallback.
    """
    classify = mock_responses.get("classify", {"pattern": "direct"})
    decompose = mock_responses.get("decompose", {"subtasks": []})
    completeness = mock_responses.get(
        "completeness",
        {
            "complete": True,
            "missing": [],
            "redundant": [],
            "coverage_ratio": 1.0,
            "reasoning": "mock: all covered",
        },
    )
    workflow = mock_responses.get("workflow")

    # Track call index for multi-turn workflow scenarios
    workflow_turns: list = []
    if isinstance(workflow, list):
        workflow_turns.extend(workflow)
    elif workflow is not None:
        workflow_turns.append(workflow)
    _wf_idx = [0]

    def mock_chat_json(**kwargs) -> Dict[str, Any]:
        messages = kwargs.get("messages", [])
        system_msg = _extract_system(messages)

        # 1) Spine classification
        if "query classifier" in system_msg.lower() or "classify" in system_msg.lower():
            return dict(classify)

        # 2) AOP decomposition
        if "decomposition" in system_msg.lower():
            return dict(decompose)

        # 3) Completeness audit
        if "completeness" in system_msg.lower():
            return dict(completeness)

        # 4) Workflow mapper / event extraction
        if workflow_turns and (
            "event" in system_msg.lower()
            or "workflow" in system_msg.lower()
            or "slot" in system_msg.lower()
        ):
            idx = min(_wf_idx[0], len(workflow_turns) - 1)
            _wf_idx[0] += 1
            return dict(workflow_turns[idx])

        # 5) Fallback — return classification as safe default
        return dict(classify)

    return mock_chat_json


def build_voice_mock():
    """Return a no-op voice chat mock (returns empty string)."""

    def mock_voice_chat(**kwargs) -> str:
        return ""

    return mock_voice_chat


def apply_mocks(monkeypatch, mock_responses: Dict[str, Any]) -> None:
    """
    Apply all LLM mocks for a scenario using pytest monkeypatch.

    Patches:
      - app.llm_client.chat_json       (spine classification)
      - app.orchestration.aop_coordinator.chat_json  (decompose)
      - app.orchestration.completeness_detector.chat_json (completeness)
      - app.shared.workflow.chat_json   (workflow FSM mapper)
    """
    mock_fn = build_scenario_mock(mock_responses)

    monkeypatch.setattr("app.llm_client.chat_json", mock_fn)
    monkeypatch.setattr("app.orchestration.aop_coordinator.chat_json", mock_fn)
    monkeypatch.setattr("app.orchestration.completeness_detector.chat_json", mock_fn)

    try:
        monkeypatch.setattr("app.shared.workflow.chat_json", mock_fn)
    except AttributeError:
        pass  # workflow module may not use chat_json directly


def _extract_system(messages: list) -> str:
    """Extract the system message content from a messages list."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return m.get("content", "")
    return ""
