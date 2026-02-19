# tests/test_spine_rag_delegation.py
"""Tests for RAG delegation, clarification pinning, and memory in RuntimeSpine."""
from __future__ import annotations

from typing import Any, Dict

from app.runtime.guardrails import NoOpGuardrails
from app.runtime.memory import ConversationMemory
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Candidate, RoutePlan
from app.runtime.spine import THREAD_CTX, RuntimeSpine


# ── Stub Agents ──────────────────────────────────────────────────────


class StubRAGAgent:
    """RAG agent that returns configurable responses."""

    def __init__(self, response: Dict[str, Any]):
        self._response = response

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._response)

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "faq_agent",
            "type": "faq_rag",
            "ready": True,
            "description": "FAQ agent",
            "capabilities": ["faq_answering"],
        }


class StubWorkflowAgent:
    """Workflow agent that delegation targets."""

    def __init__(self, response: Dict[str, Any]):
        self._response = response

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._response)

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "refund_agent",
            "type": "workflow_runner",
            "ready": True,
            "description": "Refund workflow",
            "capabilities": ["refund_processing"],
        }


class FixedRouter:
    def __init__(self, agent_id: str):
        self._id = agent_id

    def route(self, query: str) -> RoutePlan:
        return RoutePlan(
            primary=self._id,
            strategy="single",
            candidates=[Candidate(id=self._id, score=1.0, reason="fixed")],
        )


def _clear_thread(thread_id: str):
    THREAD_CTX.pop(thread_id, None)


# ── RAG Pinning Tests ───────────────────────────────────────────────


class TestRAGPinning:
    def test_rag_clarification_pins_agent(self):
        """When RAG returns rag_clarification=True, agent gets pinned."""
        _clear_thread("pin_test")
        registry = AgentRegistry()
        rag = StubRAGAgent(
            {"intent": "faq", "answer": "What topic?", "score": 0.5, "rag_clarification": True}
        )
        registry.register("faq_agent", rag, rag.metadata())

        memory = ConversationMemory()
        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
            memory=memory,
        )

        result = spine.handle_chat("um", context={"thread_id": "pin_test"})
        assert result.get("rag_clarification") is True

        # Check context was pinned
        ctx = THREAD_CTX.get("pin_test", {})
        assert ctx.get("pinned_agent_id") == "faq_agent"
        assert ctx.get("pinned_agent_type") == "rag_fsm"
        _clear_thread("pin_test")

    def test_rag_pinned_agent_receives_followup(self):
        """Pinned RAG agent should receive the next message via sticky routing."""
        _clear_thread("followup_test")
        registry = AgentRegistry()

        # First call: clarification
        rag = StubRAGAgent(
            {"intent": "faq", "answer": "What topic?", "score": 0.5, "rag_clarification": True}
        )
        registry.register("faq_agent", rag, rag.metadata())

        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
        )

        spine.handle_chat("um", context={"thread_id": "followup_test"})

        # Now change the agent's response to a final answer
        rag._response = {
            "intent": "faq",
            "answer": "30-day refund window.",
            "score": 0.9,
            "rag_answered": True,
        }

        result2 = spine.handle_chat(
            "What is the refund policy?", context={"thread_id": "followup_test"}
        )
        assert result2.get("answer") == "30-day refund window."
        # Agent should now be unpinned
        ctx = THREAD_CTX.get("followup_test", {})
        assert "pinned_agent_id" not in ctx
        _clear_thread("followup_test")

    def test_rag_answer_unpins_agent(self):
        """When RAG returns rag_answered=True, agent gets unpinned."""
        _clear_thread("unpin_test")

        # Pre-pin a RAG agent
        THREAD_CTX["unpin_test"] = {
            "thread_id": "unpin_test",
            "pinned_agent_id": "faq_agent",
            "pinned_agent_type": "rag_fsm",
            "pinned_terminal": False,
        }

        registry = AgentRegistry()
        rag = StubRAGAgent(
            {"intent": "faq", "answer": "Here is the answer.", "score": 0.9, "rag_answered": True}
        )
        registry.register("faq_agent", rag, rag.metadata())

        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
        )

        spine.handle_chat("refund policy", context={"thread_id": "unpin_test"})
        ctx = THREAD_CTX.get("unpin_test", {})
        assert "pinned_agent_id" not in ctx
        _clear_thread("unpin_test")


# ── Delegation Tests ─────────────────────────────────────────────────


class TestDelegationSignals:
    def test_delegation_signal_triggers_reroute(self):
        """When RAG returns delegation_target, spine re-routes to that agent."""
        _clear_thread("delegate_test")
        registry = AgentRegistry()

        rag = StubRAGAgent(
            {
                "intent": "faq",
                "answer": "",
                "score": 0.1,
                "delegation_target": "refund_agent",
                "delegation_reason": "Out of scope for FAQ",
            }
        )
        registry.register("faq_agent", rag, rag.metadata())

        workflow = StubWorkflowAgent({"answer": "Refund processed.", "score": 0.9})
        registry.register("refund_agent", workflow, workflow.metadata())

        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
        )

        result = spine.handle_chat("I want a refund", context={"thread_id": "delegate_test"})
        # Should have been re-routed to refund_agent
        assert result.get("agent_id") == "refund_agent"
        assert result.get("answer") == "Refund processed."
        _clear_thread("delegate_test")

    def test_delegation_to_unknown_agent_keeps_original(self):
        """If delegation target doesn't exist, keep original response."""
        _clear_thread("unknown_delegate")
        registry = AgentRegistry()

        rag = StubRAGAgent(
            {
                "intent": "faq",
                "answer": "I can't help with that.",
                "score": 0.1,
                "delegation_target": "nonexistent_agent",
                "delegation_reason": "Out of scope",
            }
        )
        registry.register("faq_agent", rag, rag.metadata())

        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
        )

        result = spine.handle_chat("quantum physics", context={"thread_id": "unknown_delegate"})
        # Original agent's response should still be there (delegation failed gracefully)
        assert result.get("agent_id") == "faq_agent"
        _clear_thread("unknown_delegate")


# ── Memory Recording Tests ───────────────────────────────────────────


class TestMemoryRecording:
    def test_spine_records_turns_in_memory(self):
        """Spine should record each turn in ConversationMemory."""
        _clear_thread("mem_test")
        registry = AgentRegistry()
        rag = StubRAGAgent(
            {"intent": "faq", "answer": "30 days.", "score": 0.9, "rag_answered": True}
        )
        registry.register("faq_agent", rag, rag.metadata())

        memory = ConversationMemory()
        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
            memory=memory,
        )

        spine.handle_chat("What is the refund policy?", context={"thread_id": "mem_test"})

        turns = memory.get_turns("mem_test")
        assert len(turns) == 1
        assert turns[0].query == "What is the refund policy?"
        assert turns[0].agent_id == "faq_agent"
        _clear_thread("mem_test")

    def test_spine_memory_multiple_turns(self):
        """Multiple turns recorded in order."""
        _clear_thread("multi_mem")
        registry = AgentRegistry()
        rag = StubRAGAgent({"intent": "faq", "answer": "Answer", "score": 0.8})
        registry.register("faq_agent", rag, rag.metadata())

        memory = ConversationMemory()
        spine = RuntimeSpine(
            registry=registry,
            router=FixedRouter("faq_agent"),
            guardrails=NoOpGuardrails(),
            memory=memory,
        )

        spine.handle_chat("Q1", context={"thread_id": "multi_mem"})
        spine.handle_chat("Q2", context={"thread_id": "multi_mem"})

        turns = memory.get_turns("multi_mem")
        assert len(turns) == 2
        assert turns[0].query == "Q1"
        assert turns[1].query == "Q2"
        _clear_thread("multi_mem")
