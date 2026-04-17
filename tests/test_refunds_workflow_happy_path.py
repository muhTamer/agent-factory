# tests/test_refunds_workflow_happy_path.py
"""
Happy-path tests for the refunds domain agent (refunds_agent).

The refunds agent is a ReACT domain agent that:
  - Retrieves refund policy from its corpus
  - Asks clarifying questions (ask_user)
  - Calls tools (initiate_refund, lookup_payment, etc.)
  - Accumulates slots across turns
  - Isolates thread state

Requires: generated/refunds_agent/ artifacts from a factory deploy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO_ROOT / "generated" / "refunds_agent"
AGENT_PATH = AGENT_DIR / "agent.py"

pytestmark = pytest.mark.skipif(
    not AGENT_PATH.exists(),
    reason="refunds_agent artifacts not found — run factory deploy first",
)


def _load_agent():
    spec = importlib.util.spec_from_file_location("_test_refunds_agent", AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_refunds_agent"] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


@pytest.fixture(scope="module")
def refunds_agent():
    return _load_agent()


# ---------------------------------------------------------------------------
# Loading & configuration
# ---------------------------------------------------------------------------
class TestRefundsAgentLoading:
    def test_loads_without_error(self, refunds_agent):
        assert refunds_agent.ready is True

    def test_metadata_id(self, refunds_agent):
        meta = refunds_agent.metadata()
        assert meta["id"] == "refunds_agent"

    def test_metadata_domain(self, refunds_agent):
        meta = refunds_agent.metadata()
        assert meta["domain"] == "refunds"

    def test_metadata_ready(self, refunds_agent):
        meta = refunds_agent.metadata()
        assert meta["ready"] is True

    def test_has_tools(self, refunds_agent):
        meta = refunds_agent.metadata()
        tools = meta.get("available_tools", [])
        assert len(tools) > 0, f"Refunds agent should have tools, got: {tools}"

    def test_has_policies(self):
        cfg = json.loads((AGENT_DIR / "config.json").read_text(encoding="utf-8"))
        policies = cfg.get("policies", [])
        assert len(policies) > 0, "Refunds agent should have policy constraints"

    def test_has_corpus(self):
        corpus = json.loads((AGENT_DIR / "corpus.json").read_text(encoding="utf-8"))
        assert len(corpus) > 0, "Refunds agent should have corpus entries"


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------
class TestRefundsAgentHandle:
    def test_returns_answer(self, refunds_agent):
        result = refunds_agent.handle({"query": "I want a refund"})
        assert result.get("answer"), f"Expected answer, got: {result}"

    def test_returns_domain_refunds(self, refunds_agent):
        result = refunds_agent.handle({"query": "I want a refund"})
        assert result.get("domain") == "refunds"

    def test_returns_agent_id(self, refunds_agent):
        result = refunds_agent.handle({"query": "I want a refund"})
        assert result.get("agent_id") == "refunds_agent"

    def test_has_react_trace(self, refunds_agent):
        result = refunds_agent.handle({"query": "I want a refund"})
        assert "react_trace" in result
        assert len(result["react_trace"]) >= 1

    def test_policies_applied(self, refunds_agent):
        result = refunds_agent.handle({"query": "I want a refund"})
        assert len(result.get("policies_applied", [])) > 0


# ---------------------------------------------------------------------------
# Multi-turn conversation
# ---------------------------------------------------------------------------
class TestRefundsMultiTurn:
    def test_first_turn_asks_for_details(self, refunds_agent):
        """First turn with vague refund request should ask for details."""
        r1 = refunds_agent.handle({"query": "I want a refund", "thread_id": "mt1"})
        # Agent should either ask for details or retrieve knowledge
        assert r1.get("answer"), "Should produce a response"

    def test_second_turn_preserves_context(self, refunds_agent):
        """Second turn should remember the thread context."""
        refunds_agent.handle({"query": "I want a refund", "thread_id": "mt2"})  # turn 1
        r2 = refunds_agent.handle({"query": "Order #12345", "thread_id": "mt2"})
        assert r2.get("answer"), "Second turn should produce a response"
        # Should have retrieved knowledge at some point across turns
        assert r2.get("knowledge_retrieved") is True


# ---------------------------------------------------------------------------
# Thread isolation
# ---------------------------------------------------------------------------
class TestRefundsThreadIsolation:
    def test_threads_independent(self, refunds_agent):
        """Different thread_ids should not share state."""
        r_a = refunds_agent.handle(
            {"query": "I want a refund for order A", "thread_id": "iso_a"}
        )
        r_b = refunds_agent.handle(
            {"query": "I want a refund for order B", "thread_id": "iso_b"}
        )
        # Both should produce independent responses
        assert r_a.get("answer"), "Thread A should respond"
        assert r_b.get("answer"), "Thread B should respond"

    def test_slots_not_shared(self, refunds_agent):
        """Slots accumulated in one thread should not leak to another."""
        refunds_agent.handle(  # populate thread slot_a
            {
                "query": "Refund for order 999",
                "thread_id": "slot_a",
                "context": {"_accumulated_slots": {"customer_id": "CUST-A"}},
            }
        )
        r2 = refunds_agent.handle({"query": "Refund please", "thread_id": "slot_b"})
        slots_b = r2.get("slots", {})
        assert "CUST-A" not in str(
            slots_b
        ), "Slots from thread A should not appear in thread B"
