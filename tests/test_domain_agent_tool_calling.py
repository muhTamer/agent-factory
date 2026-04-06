# tests/test_domain_agent_tool_calling.py
"""
Tests that domain agents actually call tools when they should.

These tests target the exact failure patterns from soft-pass evaluation
scenarios: agents responding textually instead of calling tools like
initiate_refund and create_ticket.

The fix: rich tool descriptions + bias-toward-action prompt + better
multi-turn context. These tests validate the LLM receives the right
signals to call tools.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from app.runtime.domain_agent_engine import DomainAgentConfig, DomainAgentEngine
from app.shared.rag import CorpusItem, build_index


# ── Helpers ──────────────────────────────────────────────────────────


def _build_refund_corpus():
    return [
        CorpusItem(
            text="Refund Policy: Refunds under EUR 5000 are auto-approved. "
            "Step 1: Collect transaction reference. Step 2: Verify identity. "
            "Step 3: Initiate refund via initiate_refund tool.",
            source="refunds_policy.yaml",
            kind="other",
            meta={"visibility": "internal"},
        ),
    ]


def _build_complaint_corpus():
    return [
        CorpusItem(
            text="Complaints Policy: All complaints must be logged via "
            "create_ticket tool. Step 1: Collect complaint details. "
            "Step 2: Create ticket. Step 3: Acknowledge to customer.",
            source="complaints_policy.yaml",
            kind="other",
            meta={"visibility": "internal"},
        ),
    ]


def _make_tool(name: str, result: Dict[str, Any]) -> MagicMock:
    """Create a mock tool with a rich description (as the fix provides)."""
    from app.runtime.tools.stub_tools import TOOL_DESCRIPTIONS

    tool = MagicMock()
    tool.execute.return_value = result
    tool.describe.return_value = {
        "name": name,
        "description": TOOL_DESCRIPTIONS.get(name, f"{name} tool"),
    }
    return tool


def _make_refund_engine(
    llm_responses: List[Dict[str, Any]],
    max_steps: int = 5,
) -> DomainAgentEngine:
    corpus = _build_refund_corpus()
    index = build_index(corpus)

    config = DomainAgentConfig(
        agent_id="refunds_agent",
        domain="refunds",
        goal="Help customers with refund requests",
        policies=["Auto-approve refunds under EUR 5000"],
        max_steps=max_steps,
    )

    call_count = {"n": 0}

    def mock_llm(messages, model=None, temperature=None):
        idx = min(call_count["n"], len(llm_responses) - 1)
        call_count["n"] += 1
        return llm_responses[idx]

    tools = {
        "initiate_refund": _make_tool(
            "initiate_refund",
            {"refund_id": "REF-001", "refund_status": "success"},
        ),
        "lookup_payment": _make_tool(
            "lookup_payment",
            {"payment_found": True, "amount": 75.0, "status": "settled"},
        ),
        "verify_identity": _make_tool(
            "verify_identity",
            {"kyc_status": "verified", "identity_verified": True},
        ),
    }

    return DomainAgentEngine(
        config=config, index=index, tools=tools, llm_fn=mock_llm
    )


def _make_complaint_engine(
    llm_responses: List[Dict[str, Any]],
    max_steps: int = 5,
) -> DomainAgentEngine:
    corpus = _build_complaint_corpus()
    index = build_index(corpus)

    config = DomainAgentConfig(
        agent_id="complaints_agent",
        domain="complaints",
        goal="Handle customer complaints",
        policies=["All complaints must be logged"],
        max_steps=max_steps,
    )

    call_count = {"n": 0}

    def mock_llm(messages, model=None, temperature=None):
        idx = min(call_count["n"], len(llm_responses) - 1)
        call_count["n"] += 1
        return llm_responses[idx]

    tools = {
        "create_ticket": _make_tool(
            "create_ticket",
            {"ticket_id": "TKT-001", "ticket_status": "created"},
        ),
        "handoff_to_human": _make_tool(
            "handoff_to_human",
            {"handed_off": True},
        ),
    }

    return DomainAgentEngine(
        config=config, index=index, tools=tools, llm_fn=mock_llm
    )


# ── Tests ────────────────────────────────────────────────────────────


class TestToolDescriptionsInPrompt:
    """Verify that tool descriptions are rich and informative in the prompt."""

    def test_tool_descriptions_include_usage_guidance(self):
        """Tool descriptions should tell the LLM WHEN to call each tool."""
        from app.runtime.tools.stub_tools import TOOL_DESCRIPTIONS

        # All stub tools should have descriptions
        from app.runtime.tools.stub_tools import STUB_TOOLS

        for name in STUB_TOOLS:
            assert name in TOOL_DESCRIPTIONS, (
                f"Tool '{name}' missing from TOOL_DESCRIPTIONS"
            )
            desc = TOOL_DESCRIPTIONS[name]
            assert len(desc) > 20, (
                f"Tool '{name}' description too short: {desc}"
            )

    def test_stub_tool_carries_description(self):
        """StubTool.describe() should return the rich description."""
        from app.runtime.tools.adapters.stub import StubTool
        from app.runtime.tools.stub_tools import STUB_TOOLS, TOOL_DESCRIPTIONS

        for name, fn in STUB_TOOLS.items():
            desc = TOOL_DESCRIPTIONS.get(name, "")
            tool = StubTool(name, fn, description=desc)
            info = tool.describe()
            assert info["description"] == desc
            assert "Stub implementation" not in info["description"]

    def test_default_registry_has_descriptions(self):
        """The default registry tools should have rich descriptions."""
        from app.runtime.tools import _build_default_registry

        registry = _build_default_registry()
        for name in registry.all_names():
            tool = registry.get(name)
            info = tool.describe()
            assert "Stub implementation" not in info["description"], (
                f"Tool '{name}' still has generic description"
            )


class TestRefundToolCalling:
    """
    Test that refund agent calls initiate_refund when it should.
    Mirrors soft-pass scenarios: refund_05, deleg_12, hitl_03, b77_refund_03.
    """

    def test_refund_agent_calls_tool_after_details(self):
        """
        Scenario: User asks for refund → agent retrieves policy → asks for
        details → user provides details → agent calls initiate_refund.
        """
        engine = _make_refund_engine(
            llm_responses=[
                # Turn 1: retrieve policy
                {
                    "thought": "Need to look up refund policy.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "refund duplicate charge"},
                },
                # Turn 1: ask for account details
                {
                    "thought": "Policy retrieved. Need account info.",
                    "action": "ask_user",
                    "action_input": {
                        "question": "Could you provide your account number?"
                    },
                },
                # Turn 2: call initiate_refund (after user provides details)
                {
                    "thought": "User provided account. Proceeding with refund.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "initiate_refund",
                        "args": {
                            "amount": 75,
                            "account_id": "ACC-2024-5501",
                            "reason": "duplicate charge",
                        },
                    },
                },
                # Turn 2: respond with confirmation
                {
                    "thought": "Refund initiated successfully.",
                    "action": "respond",
                    "action_input": {
                        "answer": "Your refund of EUR 75 has been initiated."
                    },
                },
            ]
        )

        # Turn 1
        r1 = engine.handle(
            "I was charged twice for EUR 75, please refund one",
            thread_id="t1",
        )
        assert r1.get("needs_input") or r1.get("domain_agent_clarification")

        # Turn 2
        r2 = engine.handle(
            "Account ACC-2024-5501, charges on March 10th",
            thread_id="t1",
        )
        assert "initiate_refund" in r2.get("tools_used", [])
        assert "75" in r2.get("answer", "")

    def test_refund_agent_calls_tool_in_single_turn(self):
        """
        When user provides all details upfront, agent should call the
        tool without asking for more info.
        """
        engine = _make_refund_engine(
            llm_responses=[
                {
                    "thought": "User provided amount and account. Call refund tool.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "initiate_refund",
                        "args": {
                            "amount": 200,
                            "account_id": "ACC-123",
                            "reason": "unauthorized charge",
                        },
                    },
                },
                {
                    "thought": "Refund done.",
                    "action": "respond",
                    "action_input": {
                        "answer": "Your refund of EUR 200 has been processed."
                    },
                },
            ]
        )

        result = engine.handle(
            "Refund EUR 200 unauthorized charge on ACC-123"
        )
        assert "initiate_refund" in result.get("tools_used", [])
        assert "200" in result.get("answer", "")


class TestComplaintToolCalling:
    """
    Test that complaint agent calls create_ticket.
    Mirrors soft-pass scenario: complaint_01.
    """

    def test_complaint_agent_creates_ticket(self):
        """Agent should call create_ticket when customer files a complaint."""
        engine = _make_complaint_engine(
            llm_responses=[
                {
                    "thought": "Customer filing a complaint. Retrieve policy.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "complaint procedure"},
                },
                {
                    "thought": "Need complaint details.",
                    "action": "ask_user",
                    "action_input": {
                        "question": "Could you describe the issue?"
                    },
                },
                {
                    "thought": "Have details. Creating ticket.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "create_ticket",
                        "args": {
                            "subject": "Poor service at branch",
                            "description": "Waited 2 hours at downtown branch",
                        },
                    },
                },
                {
                    "thought": "Ticket created. Confirming.",
                    "action": "respond",
                    "action_input": {
                        "answer": "Your complaint has been logged. "
                        "A reference number will be sent to you."
                    },
                },
            ]
        )

        # Turn 1
        r1 = engine.handle(
            "I want to file a formal complaint about the poor service",
            thread_id="c1",
        )
        assert r1.get("needs_input") or r1.get("domain_agent_clarification")

        # Turn 2
        r2 = engine.handle(
            "Downtown branch, High Street, waited 2 hours on March 18th",
            thread_id="c1",
        )
        assert "create_ticket" in r2.get("tools_used", [])


class TestBiasTowardActionPrompt:
    """Verify the ReAct prompt includes action-bias instructions."""

    def test_prompt_contains_action_bias(self):
        """The system prompt should include bias-toward-action guidance."""
        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="test",
            domain="refunds",
            goal="Help with refunds",
        )
        engine = DomainAgentEngine(
            config=config, index=index, tools={}, llm_fn=lambda **kw: {}
        )

        from app.runtime.domain_agent_engine import ThreadState

        state = ThreadState()
        state.turn_count = 1
        msgs = engine._build_react_prompt("refund request", state, [], {})
        system = msgs[0]["content"]

        assert "Bias toward completing actions" in system
        assert "initiate_refund" in system
        assert "create_ticket" in system

    def test_multi_turn_resume_prompt_encourages_action(self):
        """After ask_user, resume prompt should push toward tool calls."""
        corpus = _build_refund_corpus()
        index = build_index(corpus)
        config = DomainAgentConfig(
            agent_id="test",
            domain="refunds",
            goal="Help with refunds",
        )
        engine = DomainAgentEngine(
            config=config, index=index, tools={}, llm_fn=lambda **kw: {}
        )

        from app.runtime.domain_agent_engine import ThreadState

        state = ThreadState()
        state.turn_count = 2
        state.pending_question = "What is your account number?"
        msgs = engine._build_react_prompt("ACC-12345", state, [], {})
        user_content = msgs[1]["content"]

        assert "Do NOT ask another question" in user_content
        assert "proceed to call the appropriate tool" in user_content


class TestRetrievalThresholds:
    """Verify the updated retrieval thresholds."""

    def test_domain_agent_config_defaults(self):
        """top_k should be 8 and threshold should be 0.10."""
        config = DomainAgentConfig(
            agent_id="test", domain="faq", goal="answer FAQ"
        )
        assert config.top_k == 8
        assert config.retrieval_threshold == 0.10

    def test_rag_fsm_config_defaults(self):
        """RAGFSMConfig should have updated defaults."""
        from app.runtime.rag_fsm import RAGFSMConfig

        cfg = RAGFSMConfig()
        assert cfg.top_k == 8
        assert cfg.relevance_gate == 0.10
        assert cfg.solvability_threshold == 0.20
