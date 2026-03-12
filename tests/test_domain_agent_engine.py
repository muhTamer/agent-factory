# tests/test_domain_agent_engine.py
"""Tests for the Domain Agent ReAct engine."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock


from app.runtime.domain_agent_engine import (
    DomainAgentConfig,
    DomainAgentEngine,
)
from app.shared.rag import CorpusItem, build_index

# ── Helpers ──────────────────────────────────────────────────────────


def _build_corpus() -> list:
    items = [
        CorpusItem(
            text="Refunds are available within 30 days of purchase for eligible items.",
            source="refund_policy.csv",
            kind="faq",
            meta={},
        ),
        CorpusItem(
            text="To reset your password, go to Settings > Security > Reset Password.",
            source="account_help.csv",
            kind="faq",
            meta={},
        ),
    ]
    return items


def _make_engine(
    llm_responses: List[Dict[str, Any]] | None = None,
    tools: Dict[str, Any] | None = None,
    policies: List[str] | None = None,
    max_steps: int = 5,
) -> DomainAgentEngine:
    corpus = _build_corpus()
    index = build_index(corpus)

    config = DomainAgentConfig(
        agent_id="test_agent",
        domain="refunds",
        goal="Help customers with refund requests",
        policies=policies or [],
        max_steps=max_steps,
    )

    # Build mock LLM
    call_count = {"n": 0}
    responses = llm_responses or []

    def mock_llm(messages, model=None, temperature=None):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return (
            responses[idx]
            if responses
            else {
                "thought": "Default response",
                "action": "respond",
                "action_input": {"answer": "Default answer"},
            }
        )

    return DomainAgentEngine(
        config=config,
        index=index,
        tools=tools or {},
        llm_fn=mock_llm,
        memory=None,
    )


def _mock_tool(name: str, result: Dict[str, Any]) -> MagicMock:
    tool = MagicMock()
    tool.execute.return_value = result
    tool.describe.return_value = {"description": f"Mock {name} tool"}
    return tool


# ── Tests ────────────────────────────────────────────────────────────


class TestReActLoop:
    """Test the core ReAct reasoning loop."""

    def test_single_step_respond(self):
        """Agent responds directly in one step."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "User wants to know the refund policy, I can answer directly.",
                    "action": "respond",
                    "action_input": {"answer": "Refunds are available within 30 days."},
                }
            ]
        )

        result = engine.handle("What is the refund policy?")
        assert result["answer"] == "Refunds are available within 30 days."
        assert result["step_count"] == 1
        assert result["domain"] == "refunds"
        assert result["agent_id"] == "test_agent"

    def test_retrieve_then_respond(self):
        """Agent retrieves knowledge, then responds."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "I need to look up refund policy information.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "refund policy"},
                },
                {
                    "thought": "Found the policy. Refunds within 30 days.",
                    "action": "respond",
                    "action_input": {
                        "answer": "Based on our policy, refunds are available within 30 days of purchase."
                    },
                },
            ]
        )

        result = engine.handle("What is the refund policy?")
        assert "30 days" in result["answer"]
        assert result["step_count"] == 2
        assert result["knowledge_retrieved"] is True

    def test_tool_call_then_respond(self):
        """Agent calls a tool, then responds."""
        lookup_tool = _mock_tool(
            "lookup_payment", {"order_id": "123", "amount": 49.99, "status": "paid"}
        )

        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Need to look up the payment for order 123.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "lookup_payment",
                        "args": {"order_id": "123"},
                    },
                },
                {
                    "thought": "Payment found. Amount was $49.99.",
                    "action": "respond",
                    "action_input": {
                        "answer": "Your order #123 was $49.99 and has been paid."
                    },
                },
            ],
            tools={"lookup_payment": lookup_tool},
        )

        result = engine.handle("Check my order 123")
        assert result["step_count"] == 2
        assert "lookup_payment" in result["tools_used"]
        lookup_tool.execute.assert_called_once()

    def test_ask_user(self):
        """Agent asks for more information."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "I need the order ID to proceed.",
                    "action": "ask_user",
                    "action_input": {"question": "What is your order number?"},
                },
            ]
        )

        result = engine.handle("I want a refund")
        assert result["needs_input"] is True
        assert result["domain_agent_clarification"] is True
        assert "order number" in result["answer"]

    def test_escalate(self):
        """Agent escalates when it cannot handle the request."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "This requires manual review.",
                    "action": "escalate",
                    "action_input": {
                        "reason": "Request exceeds automated processing limits"
                    },
                },
            ]
        )

        result = engine.handle("I want to escalate my complaint")
        assert result["escalation"] is True
        assert "escalation_reason" in result

    def test_max_steps_reached(self):
        """Agent hits max_steps without terminal action → escalation."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Looking up info...",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "something"},
                },
            ]
            * 3,
            max_steps=3,
        )

        result = engine.handle("Some complex query")
        assert result["step_count"] == 3
        assert result["escalation"] is True
        assert "Max reasoning steps" in result["escalation_reason"]


class TestReActTrace:
    """Test explainability / RQ2: ReAct trace in responses."""

    def test_trace_present(self):
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Simple question, respond directly.",
                    "action": "respond",
                    "action_input": {"answer": "Hello!"},
                },
            ]
        )

        result = engine.handle("Hi")
        assert "react_trace" in result
        assert len(result["react_trace"]) == 1
        trace = result["react_trace"][0]
        assert trace["step"] == 1
        assert trace["thought"] == "Simple question, respond directly."
        assert trace["action"] == "respond"

    def test_multi_step_trace(self):
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Retrieve first.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "refund"},
                },
                {
                    "thought": "Now respond.",
                    "action": "respond",
                    "action_input": {"answer": "Done."},
                },
            ]
        )

        result = engine.handle("Refund question")
        assert len(result["react_trace"]) == 2
        assert result["react_trace"][0]["action"] == "retrieve_knowledge"
        assert result["react_trace"][1]["action"] == "respond"


class TestMultiTurn:
    """Test multi-turn conversations (ask_user → resume)."""

    def test_ask_then_resume(self):
        """Simulate ask_user on turn 1, then user responds on turn 2."""
        call_count = {"n": 0}

        def mock_llm(messages, model=None, temperature=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {
                    "thought": "Need email to verify identity.",
                    "action": "ask_user",
                    "action_input": {"question": "What is your email address?"},
                }
            return {
                "thought": "User provided email. Can now respond.",
                "action": "respond",
                "action_input": {
                    "answer": "Identity verified. Processing your refund."
                },
            }

        corpus = _build_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="test_agent",
            domain="refunds",
            goal="Help with refunds",
            max_steps=5,
        )
        engine = DomainAgentEngine(
            config=config, index=index, tools={}, llm_fn=mock_llm
        )

        # Turn 1: ask_user
        r1 = engine.handle("I want a refund", thread_id="t1")
        assert r1["needs_input"] is True

        # Turn 2: user responds with email
        r2 = engine.handle("john@example.com", thread_id="t1")
        assert "Identity verified" in r2["answer"]
        assert r2.get("needs_input") is not True


class TestSlotAccumulation:
    """Test slot accumulation across tool calls."""

    def test_slots_accumulate(self):
        tool = _mock_tool("lookup_payment", {"order_id": "123", "amount": 49.99})

        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Look up payment.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "lookup_payment",
                        "args": {"order_id": "123"},
                    },
                },
                {
                    "thought": "Got payment info.",
                    "action": "respond",
                    "action_input": {"answer": "Found it."},
                },
            ],
            tools={"lookup_payment": tool},
        )

        result = engine.handle("Check order 123")
        assert result["slots"]["order_id"] == "123"
        assert result["slots"]["amount"] == 49.99


class TestPolicies:
    """Test RQ3: policy enforcement."""

    def test_policies_in_response(self):
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Respond.",
                    "action": "respond",
                    "action_input": {"answer": "OK"},
                },
            ],
            policies=["Refunds within 30 days only", "Verify identity first"],
        )

        result = engine.handle("Refund please")
        assert result["policies_applied"] == [
            "Refunds within 30 days only",
            "Verify identity first",
        ]


class TestErrorHandling:
    """Test graceful degradation."""

    def test_unknown_tool(self):
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Call a tool that doesn't exist.",
                    "action": "call_tool",
                    "action_input": {"tool": "nonexistent_tool", "args": {}},
                },
                {
                    "thought": "Tool not found, escalate.",
                    "action": "escalate",
                    "action_input": {"reason": "Required tool unavailable"},
                },
            ]
        )

        result = engine.handle("Do something")
        assert result["escalation"] is True

    def test_no_llm_fallback(self):
        """Without LLM, engine escalates immediately."""
        corpus = _build_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="test",
            domain="general",
            goal="Help",
            max_steps=5,
        )
        engine = DomainAgentEngine(config=config, index=index, tools={}, llm_fn=None)

        result = engine.handle("Hello")
        assert result["escalation"] is True

    def test_unrecognized_action_graceful(self):
        """LLM returns unrecognized action → graceful respond."""
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "I'll use a custom action.",
                    "action": "custom_action",
                    "action_input": {"answer": "Here's my response anyway."},
                },
            ]
        )

        result = engine.handle("Test")
        assert result["answer"] == "Here's my response anyway."
        assert result["step_count"] == 1


class TestExternalSlots:
    """Test cross-agent slot handoff via context._accumulated_slots."""

    def test_external_slots_injected(self):
        engine = _make_engine(
            llm_responses=[
                {
                    "thought": "Respond with available info.",
                    "action": "respond",
                    "action_input": {"answer": "Got it."},
                },
            ]
        )

        result = engine.handle(
            "Process my refund",
            context={"_accumulated_slots": {"customer_id": "C-999", "verified": True}},
        )
        assert result["slots"]["customer_id"] == "C-999"
        assert result["slots"]["verified"] is True
