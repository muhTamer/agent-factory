# tests/test_memory_workflow_integration.py
"""Tests for memory integration in workflow agents (PMPA audit trail)."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch


# ── Minimal workflow spec for testing ────────────────────────────────

SIMPLE_WF_SPEC: Dict[str, Any] = {
    "id": "test_wf",
    "initial_state": "start",
    "slots": {
        "amount": {"type": "number", "required": True, "description": "Amount"},
    },
    "states": {
        "start": {
            "on": {"begin": "collect_info"},
        },
        "collect_info": {
            "entry_prompt": "Please provide amount.",
            "on": {"info_provided": "done"},
        },
        "done": {
            "terminal": True,
            "entry_prompt": "Complete.",
        },
    },
}


def _build_workflow_agent(tmp_path: Path, agent_id: str = "test_agent") -> Any:
    """Build a real workflow agent from the generator and import it."""
    from app.shared.workflow import build_agent

    gen_dir = tmp_path / "generated" / agent_id
    build_agent(agent_id, {"workflow_spec": SIMPLE_WF_SPEC}, gen_dir)

    # Import the generated agent module
    agent_py = gen_dir / "agent.py"
    spec = importlib.util.spec_from_file_location(f"gen_{agent_id}", str(agent_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Agent()


class TestWorkflowMemoryIntegration:
    def test_workflow_agent_has_memory_after_load(self, tmp_path):
        """After load(), the workflow agent should have a ConversationMemory."""
        agent = _build_workflow_agent(tmp_path)
        agent.load({})
        assert agent._memory is not None

    @patch("app.runtime.workflow_mapper.chat_json")
    def test_workflow_stores_turn_in_memory(self, mock_chat, tmp_path):
        """After handling a request, the turn should be recorded in memory."""
        mock_chat.return_value = {
            "event": "begin",
            "slots": {},
            "confidence": 0.9,
            "rationale": "User wants to start",
        }

        agent = _build_workflow_agent(tmp_path)
        agent.load({})

        agent.handle({"query": "I want a refund", "context": {"thread_id": "t1"}})

        turns = agent._memory.get_turns("t1")
        assert len(turns) == 1
        assert turns[0].query == "I want a refund"
        assert turns[0].agent_id == "test_agent"

    @patch("app.runtime.workflow_mapper.chat_json")
    def test_workflow_stores_state_snapshot(self, mock_chat, tmp_path):
        """After handling a request, an FSM state snapshot should be recorded."""
        mock_chat.return_value = {
            "event": "begin",
            "slots": {},
            "confidence": 0.9,
            "rationale": "User wants to start",
        }

        agent = _build_workflow_agent(tmp_path)
        agent.load({})

        agent.handle({"query": "Start refund", "context": {"thread_id": "t2"}})

        snaps = agent._memory.get_snapshots("t2")
        assert len(snaps) >= 1
        assert snaps[0].agent_id == "test_agent"

    @patch("app.runtime.workflow_mapper.chat_json")
    def test_workflow_multiple_turns_recorded(self, mock_chat, tmp_path):
        """Multiple handle() calls should each record a turn."""
        mock_chat.return_value = {
            "event": "begin",
            "slots": {},
            "confidence": 0.9,
            "rationale": "begin",
        }

        agent = _build_workflow_agent(tmp_path)
        agent.load({})

        agent.handle({"query": "Q1", "context": {"thread_id": "t3"}})

        # Second call: provide info
        mock_chat.return_value = {
            "event": "info_provided",
            "slots": {"amount": 100},
            "confidence": 0.9,
            "rationale": "info",
        }
        agent.handle({"query": "Amount is 100", "context": {"thread_id": "t3"}})

        turns = agent._memory.get_turns("t3")
        assert len(turns) == 2
        assert turns[0].query == "Q1"
        assert turns[1].query == "Amount is 100"

    @patch("app.runtime.workflow_mapper.chat_json")
    def test_workflow_memory_context_available_for_mapper(self, mock_chat, tmp_path):
        """Conversation context from memory should be passed to the mapper."""
        call_args_list = []

        def capture_chat(messages, **kwargs):
            # Capture user message content for inspection
            for m in messages:
                if m["role"] == "user":
                    call_args_list.append(m["content"])
            return {
                "event": "begin",
                "slots": {},
                "confidence": 0.9,
                "rationale": "ok",
            }

        mock_chat.side_effect = capture_chat

        agent = _build_workflow_agent(tmp_path)
        agent.load({})

        # First call — no prior context
        agent.handle({"query": "Q1", "context": {"thread_id": "t4"}})

        # Second call — should now have conversation_history from prior turn
        mock_chat.side_effect = capture_chat
        agent.handle({"query": "Q2", "context": {"thread_id": "t4"}})

        # The second call's user content should contain conversation_history
        assert len(call_args_list) >= 2
        second_call = call_args_list[1]
        assert "conversation_history" in second_call
