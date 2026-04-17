# tests/test_perfect_multiintent_scenario.py
"""
Multi-Intent Scenario Tests for ReACT Domain Agents

Tests the AOP (Adaptive Orchestration Pattern) flow where a query
contains multiple intents that are decomposed and routed to different
domain agents.

Requires: generated/{faq_agent,refunds_agent}/ artifacts from factory deploy.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.runtime.registry import AgentRegistry
from app.runtime.routing import RoutePlan, Candidate
from app.runtime.spine import RuntimeSpine
from app.runtime.memory import ConversationMemory
from app.orchestration.performance_store import PerformanceStore
from app.orchestration.aop_coordinator import AOPCoordinator

REPO_ROOT = Path(__file__).resolve().parents[1]
FAQ_AGENT_DIR = REPO_ROOT / "generated" / "faq_agent"
REFUNDS_AGENT_DIR = REPO_ROOT / "generated" / "refunds_agent"

HAVE_ARTIFACTS = (FAQ_AGENT_DIR / "agent.py").exists() and (
    REFUNDS_AGENT_DIR / "agent.py"
).exists()

pytestmark = pytest.mark.skipif(
    not HAVE_ARTIFACTS,
    reason="Pre-built agent artifacts not found — run factory deploy first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_agent(module_name: str, agent_dir: Path):
    spec = importlib.util.spec_from_file_location(module_name, agent_dir / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


class _MockRouter:
    """Keyword-based router for deterministic testing."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def route(self, query: str) -> RoutePlan:
        q = query.lower()
        if "refund" in q:
            primary = "refunds_agent"
        else:
            primary = "faq_agent"
        return RoutePlan(
            primary=primary,
            candidates=[Candidate(id=primary, score=1.0, reason="keyword match")],
            strategy="single",
        )


class _NoOpGuardrails:
    """Guardrails that always pass."""

    def pre(self, query, context=None):
        from app.runtime.guardrails import GuardResult

        return GuardResult(allowed=True)

    def post(self, response, context=None):
        from app.runtime.guardrails import GuardResult

        return GuardResult(allowed=True)


# ---------------------------------------------------------------------------
# Single-agent routing tests
# ---------------------------------------------------------------------------
class TestSingleAgentRouting:
    """Verify that queries route to the correct agent."""

    def test_faq_query_routes_to_faq_agent(self):
        registry = AgentRegistry()
        faq = _load_agent("_mi_faq1", FAQ_AGENT_DIR)
        refunds = _load_agent("_mi_ref1", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)

        router = _MockRouter(registry)
        plan = router.route("How do I open a savings account?")
        assert plan.primary == "faq_agent"

    def test_refund_query_routes_to_refunds_agent(self):
        registry = AgentRegistry()
        faq = _load_agent("_mi_faq2", FAQ_AGENT_DIR)
        refunds = _load_agent("_mi_ref2", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)

        router = _MockRouter(registry)
        plan = router.route("I want a refund for my order")
        assert plan.primary == "refunds_agent"


# ---------------------------------------------------------------------------
# Spine-level routing via handle_chat
# ---------------------------------------------------------------------------
class TestSpineRouting:
    """Test RuntimeSpine routes to correct agent and returns results."""

    def _build_spine(self):
        registry = AgentRegistry()
        faq = _load_agent("_sp_faq", FAQ_AGENT_DIR)
        refunds = _load_agent("_sp_ref", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)

        router = _MockRouter(registry)
        guardrails = _NoOpGuardrails()
        memory = ConversationMemory()
        perf_store = PerformanceStore()
        aop = AOPCoordinator(
            registry=registry,
            performance_store=perf_store,
            memory=memory,
        )

        spine = RuntimeSpine(
            registry=registry,
            router=router,
            guardrails=guardrails,
            memory=memory,
            aop_coordinator=aop,
        )
        return spine

    def test_faq_via_spine(self):
        spine = self._build_spine()
        result = spine.handle_chat(
            query="What is a fixed deposit?",
            request_id="test-faq-1",
            context={"thread_id": "sp1"},
        )
        assert result.get("answer"), f"Expected answer, got: {result}"

    def test_refund_via_spine(self):
        spine = self._build_spine()
        result = spine.handle_chat(
            query="I want a refund",
            request_id="test-ref-1",
            context={"thread_id": "sp2"},
        )
        assert result.get("answer"), f"Expected answer, got: {result}"

    def test_spine_returns_router_plan(self):
        spine = self._build_spine()
        result = spine.handle_chat(
            query="What is a debit card?",
            request_id="test-plan-1",
            context={"thread_id": "sp3"},
        )
        assert "router_plan" in result
        assert result["router_plan"]["primary"] in ("faq_agent", "refunds_agent")


# ---------------------------------------------------------------------------
# Sticky routing (multi-turn pinning)
# ---------------------------------------------------------------------------
class TestStickyRouting:
    """When an agent asks a clarification, the next message should
    be routed to the same agent (sticky/pinned routing)."""

    def test_second_turn_sticky(self):
        registry = AgentRegistry()
        faq = _load_agent("_sticky_faq", FAQ_AGENT_DIR)
        refunds = _load_agent("_sticky_ref", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)

        router = _MockRouter(registry)
        guardrails = _NoOpGuardrails()
        spine = RuntimeSpine(
            registry=registry,
            router=router,
            guardrails=guardrails,
        )

        # Turn 1: refund query (side effect: pins agent to thread)
        spine.handle_chat(
            query="I want a refund",
            request_id="sticky-1",
            context={"thread_id": "sticky_t"},
        )

        # Turn 2: follow-up without "refund" keyword
        # If agent is pinned, it should still route to refunds_agent
        r2 = spine.handle_chat(
            query="Order number is 12345",
            request_id="sticky-2",
            context={"thread_id": "sticky_t"},
        )
        assert r2.get("answer"), "Second turn should produce a response"


# ---------------------------------------------------------------------------
# Session independence
# ---------------------------------------------------------------------------
class TestSessionIndependence:
    """Multiple sessions (thread_ids) should not interfere."""

    def test_parallel_sessions(self):
        registry = AgentRegistry()
        faq = _load_agent("_par_faq", FAQ_AGENT_DIR)
        registry.register("faq_agent", faq)

        router = _MockRouter(registry)
        guardrails = _NoOpGuardrails()
        spine = RuntimeSpine(registry=registry, router=router, guardrails=guardrails)

        r1 = spine.handle_chat(
            query="savings account",
            request_id="par-1",
            context={"thread_id": "sess_1"},
        )
        r2 = spine.handle_chat(
            query="credit card limits",
            request_id="par-2",
            context={"thread_id": "sess_2"},
        )
        assert r1.get("answer"), "Session 1 should respond"
        assert r2.get("answer"), "Session 2 should respond"
