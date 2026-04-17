# tests/test_auto_chain.py
"""
Tests for ReACT multi-step reasoning (replaces old auto-chain FSM tests).

In the old architecture, the workflow runner used a policy bridge to
auto-advance through system states. In the ReACT architecture, the agent
autonomously reasons through retrieve → tool-call → respond steps.

These tests verify that the ReACT engine correctly chains multiple
actions in a single turn without user intervention.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.runtime.domain_agent_engine import (
    DomainAgentConfig,
    DomainAgentEngine,
)
from app.shared.rag import CorpusItem, build_index


def _build_refund_corpus():
    return [
        CorpusItem(
            text=(
                "Refund eligibility: customer must have verified identity "
                "and active account. Frozen or closed accounts are not eligible."
            ),
            source="refunds_policy.yaml",
            kind="policy",
            meta={"visibility": "internal"},
        ),
        CorpusItem(
            text=(
                "Refunds up to EUR 5000 are auto-approved. "
                "Refunds above EUR 5000 require manager approval."
            ),
            source="refunds_policy.yaml",
            kind="policy",
            meta={"visibility": "internal"},
        ),
        CorpusItem(
            text=(
                "Step 1: Collect transaction reference. "
                "Step 2: Verify eligibility. "
                "Step 3: Execute refund via initiate_refund tool."
            ),
            source="refunds_policy.yaml",
            kind="policy",
            meta={"visibility": "internal"},
        ),
    ]


def _mock_tool(name, result):
    tool = MagicMock()
    tool.execute.return_value = result
    tool.describe.return_value = {"description": f"{name} tool"}
    return tool


# ---------------------------------------------------------------------------
# Multi-step reasoning: retrieve → tool → respond
# ---------------------------------------------------------------------------
class TestReActMultiStepChain:
    """ReACT agent chains multiple actions in a single turn."""

    def test_retrieve_then_tool_then_respond(self):
        """Agent retrieves policy, calls tool, then responds — all in one turn."""
        refund_tool = _mock_tool(
            "initiate_refund",
            {"refund_id": "REF-001", "status": "initiated", "amount": 100.0},
        )

        call_count = {"n": 0}

        def mock_llm(messages, model=None, temperature=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "thought": "Retrieve refund policy first.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "refund eligibility"},
                }
            if call_count["n"] == 2:
                return {
                    "thought": "Customer eligible. Initiate refund.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "initiate_refund",
                        "args": {"amount": 100.0, "order_id": "ORD-123"},
                    },
                }
            return {
                "thought": "Refund initiated. Confirm to user.",
                "action": "respond",
                "action_input": {
                    "answer": "Your refund of EUR 100 has been initiated (REF-001)."
                },
            }

        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="refunds_agent",
            domain="refunds",
            goal="Process refund requests",
            max_steps=5,
        )
        engine = DomainAgentEngine(
            config=config,
            index=index,
            tools={"initiate_refund": refund_tool},
            llm_fn=mock_llm,
        )

        result = engine.handle("Refund EUR 100 for order ORD-123")

        assert result["step_count"] == 3
        assert result["knowledge_retrieved"] is True
        assert "initiate_refund" in result["tools_used"]
        assert "REF-001" in result["answer"]
        refund_tool.execute.assert_called_once()

    def test_eligible_small_amount_auto_completes(self):
        """Small refund (< 5000) should complete without asking for approval."""
        refund_tool = _mock_tool(
            "initiate_refund",
            {"refund_id": "REF-002", "status": "initiated", "amount": 500.0},
        )
        call_count = {"n": 0}

        def mock_llm(messages, model=None, temperature=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "thought": "Retrieve policy to check eligibility.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "refund eligibility auto approve"},
                }
            if call_count["n"] == 2:
                return {
                    "thought": "Amount 500 EUR is under 5000, auto-approved.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "initiate_refund",
                        "args": {"amount": 500.0},
                    },
                }
            return {
                "thought": "Confirm refund.",
                "action": "respond",
                "action_input": {"answer": "Refund of EUR 500 initiated."},
            }

        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="refunds_agent", domain="refunds", goal="Refunds", max_steps=5
        )
        engine = DomainAgentEngine(
            config=config,
            index=index,
            tools={"initiate_refund": refund_tool},
            llm_fn=mock_llm,
        )

        result = engine.handle("Refund EUR 500")
        assert "initiate_refund" in result["tools_used"]
        assert result.get("escalation") is not True

    def test_ineligible_does_not_call_refund_tool(self):
        """Frozen account should NOT call initiate_refund."""
        refund_tool = _mock_tool("initiate_refund", {})

        def mock_llm(messages, model=None, temperature=None):
            return {
                "thought": "Account is frozen per policy. Cannot refund.",
                "action": "respond",
                "action_input": {
                    "answer": "I'm sorry, refunds cannot be processed for frozen accounts."
                },
            }

        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="refunds_agent", domain="refunds", goal="Refunds", max_steps=5
        )
        engine = DomainAgentEngine(
            config=config,
            index=index,
            tools={"initiate_refund": refund_tool},
            llm_fn=mock_llm,
        )

        result = engine.handle(
            "Refund for frozen account",
            context={"_accumulated_slots": {"account_status": "frozen"}},
        )
        refund_tool.execute.assert_not_called()
        assert "frozen" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Slot accumulation across chained steps
# ---------------------------------------------------------------------------
class TestReActSlotChain:
    """Slots accumulate across tool calls within a single turn."""

    def test_slots_from_multiple_tools(self):
        """Slots from lookup + refund tools both appear in final slots."""
        lookup_tool = _mock_tool(
            "lookup_payment",
            {"order_id": "ORD-1", "amount": 99.0, "status": "paid"},
        )
        refund_tool = _mock_tool(
            "initiate_refund",
            {"refund_id": "REF-99", "status": "initiated"},
        )
        call_count = {"n": 0}

        def mock_llm(messages, model=None, temperature=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "thought": "Look up the payment first.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "lookup_payment",
                        "args": {"order_id": "ORD-1"},
                    },
                }
            if call_count["n"] == 2:
                return {
                    "thought": "Payment found. Initiate refund.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "initiate_refund",
                        "args": {"amount": 99.0},
                    },
                }
            return {
                "thought": "Done.",
                "action": "respond",
                "action_input": {"answer": "Refund initiated."},
            }

        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="test", domain="refunds", goal="Refunds", max_steps=5
        )
        engine = DomainAgentEngine(
            config=config,
            index=index,
            tools={"lookup_payment": lookup_tool, "initiate_refund": refund_tool},
            llm_fn=mock_llm,
        )

        result = engine.handle("Refund order ORD-1")
        slots = result["slots"]
        assert slots["order_id"] == "ORD-1"
        assert slots["refund_id"] == "REF-99"
        assert "lookup_payment" in result["tools_used"]
        assert "initiate_refund" in result["tools_used"]
