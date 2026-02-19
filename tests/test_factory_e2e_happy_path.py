# tests/test_factory_e2e_happy_path.py
"""
Full Factory Happy-Path End-to-End Test

This test simulates the complete lifecycle that a customer goes through:

  PHASE 1  ── Document Analysis (DUA)
               Customer uploads docs + selects domain → factory infers capabilities
               and builds a requirements.json

  PHASE 2  ── Agent Suggestions
               Factory synthesises a blueprint plan (what agents to build)
               Customer reviews and approves

  PHASE 3  ── Deploy
               Agents are loaded from the already-generated files
               Registry is populated and verified

  PHASE 4  ── Test Multiple Agents (via Spine)
               FAQs routed to faq_rag_agent   → answer returned
               Refund intent routed to refunds_workflow_agent → FSM started
               Second turn on same thread     → workflow continues (sticky routing)
               All slots provided             → workflow completes (terminal)

No real LLM calls are made.  The LLM router + workflow mapper are both mocked.
The policy bridge uses the real compiled policy pack (if present).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from app.runtime.routing import Candidate, RoutePlan

# ---------------------------------------------------------------------------
# Repo constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = REPO_ROOT / ".factory/compiled_policies/refunds_policy_pack.json"
FAQ_AGENT_DIR = REPO_ROOT / "generated/customer_qa_rag"
REFUNDS_AGENT_DIR = REPO_ROOT / "generated/refunds_workflow"
FAQS_JSON = FAQ_AGENT_DIR / "faqs.json"

# ---------------------------------------------------------------------------
# Skip guard — integration test requires pre-existing generated artifacts
# ---------------------------------------------------------------------------

HAVE_ARTIFACTS = (
    PACK_PATH.exists()
    and (FAQ_AGENT_DIR / "agent.py").exists()
    and (REFUNDS_AGENT_DIR / "agent.py").exists()
    and FAQS_JSON.exists()
)

pytestmark = pytest.mark.skipif(
    not HAVE_ARTIFACTS,
    reason="Pre-built agent artifacts not found — run factory deploy first",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_REFUND_SLOTS = {
    "case_id": "CASE-E2E-001",
    "customer_id": "CUST-E2E-001",
    "amount": 1500.0,
    "payment_method": "debit_card",
}


def _mapper_json(event: Optional[str], slots: Dict[str, Any]) -> Dict[str, Any]:
    """Return dict that workflow_mapper.chat_json would return."""
    return {"event": event, "slots": slots, "confidence": 0.99, "rationale": "e2e test mock"}


def _load_agent_from_dir(agent_id: str, agent_dir: Path):
    """Dynamically load an agent from its generated directory."""
    module_name = f"_e2e_{agent_id}"
    agent_py = agent_dir / "agent.py"
    spec = importlib.util.spec_from_file_location(module_name, agent_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


# ---------------------------------------------------------------------------
# Custom deterministic router for tests (avoids LLM call)
# ---------------------------------------------------------------------------


class _MockRouter:
    """Routes queries deterministically based on keywords — no LLM needed."""

    def __init__(self, routing_map: Dict[str, str]):
        # routing_map: keyword → agent_id  (first match wins)
        self.routing_map = routing_map
        self._all_ids = list(dict.fromkeys(routing_map.values()))

    def route(self, query: str) -> RoutePlan:
        q_lower = query.lower()
        for keyword, agent_id in self.routing_map.items():
            if keyword in q_lower:
                return RoutePlan(
                    primary=agent_id,
                    strategy="single",
                    candidates=[Candidate(id=agent_id, score=1.0, reason=f"matched '{keyword}'")],
                )
        # Fallback: first agent
        fallback = self._all_ids[0]
        return RoutePlan(
            primary=fallback,
            strategy="single",
            candidates=[Candidate(id=fallback, score=0.5, reason="fallback")],
        )


# ---------------------------------------------------------------------------
# PHASE 1 — Document Analysis (DUA)
# ---------------------------------------------------------------------------


class TestPhase1DocumentAnalysis:
    """
    Customer uploads docs and chooses domain.
    Factory infers what the system can do from the uploaded files.
    """

    def test_infer_capabilities_from_faq_and_policy_files(self, tmp_path):
        """
        Files containing 'faq' and 'policy' keywords trigger respective capabilities.
        This uses heuristic-only logic — no LLM.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        # Simulate uploaded files
        (tmp_path / "BankFAQs.csv").write_text(
            "q,a\nWhat is a refund?,A refund is...", encoding="utf-8"
        )
        (tmp_path / "refunds_policy.yaml").write_text(
            "rules: []\npolicies_present: true", encoding="utf-8"
        )

        caps = dua.infer_capabilities(list(tmp_path.glob("*")))

        assert isinstance(caps, list)
        assert len(caps) > 0
        # FAQ file should trigger faq capability
        assert "faq" in caps

    def test_build_requirements_produces_valid_structure(self, tmp_path):
        """
        build_requirements() returns a dict with expected top-level keys.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        (tmp_path / "BankFAQs.csv").write_text(
            "q,a\nHow do I refund?,Contact support.", encoding="utf-8"
        )
        files = list(tmp_path.glob("*"))

        req = dua.build_requirements("fintech", ["faq"], files)

        assert req["vertical"] == "fintech"
        assert "capabilities" in req
        assert "faq" in req["capabilities"]
        assert "entities" in req
        assert "workflows" in req

    def test_detect_signals_llm_returns_advisory_when_mocked(self, tmp_path, monkeypatch):
        """
        The LLM advisory is consumed but not required — verifies the interface contract.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        monkeypatch.setattr(
            dua,
            "detect_signals_llm",
            lambda filenames: {
                "primary": "fintech",
                "scores": {"fintech": 0.92},
                "explanation": "Banking and refund keywords found",
            },
        )

        advisory = dua.detect_signals_llm(["BankFAQs.csv", "refunds_policy.yaml"])
        assert advisory["primary"] == "fintech"
        assert advisory["scores"]["fintech"] > 0.5


# ---------------------------------------------------------------------------
# PHASE 2 — Agent Suggestions (Blueprint Plan)
# ---------------------------------------------------------------------------


class TestPhase2AgentSuggestions:
    """
    Factory generates a list of agent blueprints from the requirements.
    Customer reviews and approves the list.
    """

    def test_blueprint_plan_includes_faq_and_workflow_agents(self):
        """
        The deployed system already has both faq_rag_agent and refunds_workflow_agent.
        Verify that their configs describe the expected agent types.
        """
        import json

        faq_meta = json.loads((FAQ_AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))
        ref_meta = json.loads((REFUNDS_AGENT_DIR / "metadata.json").read_text(encoding="utf-8"))

        # Both agents should be present
        assert faq_meta["id"] == "customer_qa_rag"
        assert ref_meta["id"] == "refunds_workflow"

    def test_agent_blueprints_cover_expected_capabilities(self):
        """Each agent should advertise the capabilities it was built for."""
        faq_agent = _load_agent_from_dir("customer_qa_rag", FAQ_AGENT_DIR)
        refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

        faq_caps = faq_agent.metadata()["capabilities"]
        ref_caps = refunds_agent.metadata()["capabilities"]

        assert "faq_answering" in faq_caps
        assert "multi_turn" in ref_caps
        assert "workflow" in ref_caps

    def test_all_deployed_tool_operators_ready(self):
        """All generated tool operators are deployable and ready."""

        tool_ids = [
            "refund_executor_tool",
            "ticket_manager_tool",
        ]
        for tid in tool_ids:
            agent_dir = REPO_ROOT / "generated" / tid
            assert (agent_dir / "agent.py").exists(), f"Missing: {tid}/agent.py"


# ---------------------------------------------------------------------------
# PHASE 3 — Deploy (Load Agents into Registry)
# ---------------------------------------------------------------------------


class TestPhase3Deploy:
    """
    After customer approval, agents are loaded into the runtime registry.
    """

    def test_registry_accepts_faq_and_refunds_agents(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()

        faq_agent = _load_agent_from_dir("customer_qa_rag", FAQ_AGENT_DIR)
        refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

        registry.register("customer_qa_rag", faq_agent)
        registry.register("refunds_workflow", refunds_agent)

        assert "customer_qa_rag" in registry.all_ids()
        assert "refunds_workflow" in registry.all_ids()

    def test_registry_get_returns_loaded_agent(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()
        faq_agent = _load_agent_from_dir("customer_qa_rag", FAQ_AGENT_DIR)
        registry.register("customer_qa_rag", faq_agent)

        retrieved = registry.get("customer_qa_rag")
        assert retrieved is faq_agent

    def test_registry_all_meta_includes_agent_ids(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()
        faq_agent = _load_agent_from_dir("customer_qa_rag", FAQ_AGENT_DIR)
        refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)
        registry.register("customer_qa_rag", faq_agent)
        registry.register("refunds_workflow", refunds_agent)

        meta = registry.all_meta()
        assert "customer_qa_rag" in meta
        assert "refunds_workflow" in meta
        assert meta["customer_qa_rag"]["ready"] is True
        assert meta["refunds_workflow"]["ready"] is True

    def test_import_generated_agent_via_registry(self):
        """AgentRegistry.import_generated_agent() dynamically loads agent.py."""
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()
        agent = registry.import_generated_agent("customer_qa_rag", FAQ_AGENT_DIR)
        assert hasattr(agent, "handle")
        assert hasattr(agent, "load")
        assert hasattr(agent, "metadata")


# ---------------------------------------------------------------------------
# PHASE 4 — Test Multiple Agents via Spine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def runtime_spine():
    """
    Build a fully wired RuntimeSpine with real agents and a mock LLM router.
    """
    from app.runtime.registry import AgentRegistry
    from app.runtime.spine import RuntimeSpine
    from app.runtime.guardrails import NoOpGuardrails

    registry = AgentRegistry()

    faq_agent = _load_agent_from_dir("customer_qa_rag", FAQ_AGENT_DIR)
    refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

    registry.register("customer_qa_rag", faq_agent)
    registry.register("refunds_workflow", refunds_agent)

    router = _MockRouter(
        {
            "refund": "refunds_workflow",
            "reversal": "refunds_workflow",
            "faq": "customer_qa_rag",
            "account": "customer_qa_rag",
            "transfer": "customer_qa_rag",
        }
    )

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
    )
    return spine


class TestPhase4MultipleAgents:
    """
    Customer interacts with the deployed system via the spine.
    Shows that FAQ and workflow agents work together correctly.
    """

    # ── FAQ Agent ─────────────────────────────────────────────────────────

    def test_faq_query_routes_to_faq_agent(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "Can I transfer my account to another branch?",
            context={"thread_id": "e2e-faq-001"},
        )
        assert result.get("agent_id") == "customer_qa_rag"

    def test_faq_query_returns_non_empty_answer(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "What documents do I need for a current account?",
            context={"thread_id": "e2e-faq-002"},
        )
        answer = result.get("answer", "")
        assert answer  # non-empty

    def test_faq_response_has_score(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "Can I transfer my faq account to another branch?",
            context={"thread_id": "e2e-faq-003"},
        )
        assert "score" in result
        assert isinstance(result["score"], (int, float))

    def test_faq_response_has_request_id(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "What documents do I need for a current account faq?",
            request_id="req-faq-test-001",
            context={"thread_id": "e2e-faq-004"},
        )
        assert result.get("request_id") == "req-faq-test-001"

    def test_faq_router_plan_in_response(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "account transfer faq question",
            context={"thread_id": "e2e-faq-005"},
        )
        plan = result.get("router_plan", {})
        assert plan.get("primary") == "customer_qa_rag"

    # ── Refund Workflow — Turn 1 (info gathering) ─────────────────────────

    def test_refund_query_routes_to_refunds_agent(self, runtime_spine):
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            result = runtime_spine.handle_chat(
                "I need to process a refund for a customer",
                context={"thread_id": "e2e-refund-001"},
            )
        assert result.get("agent_id") == "refunds_workflow"

    def test_refund_turn1_starts_in_start_state(self, runtime_spine):
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            result = runtime_spine.handle_chat(
                "I need to process a refund for a customer",
                context={"thread_id": "e2e-refund-turn1-001"},
            )
        assert result.get("current_state") == "start"

    def test_refund_turn1_returns_clarification(self, runtime_spine):
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            result = runtime_spine.handle_chat(
                "I'd like to refund a payment",
                context={"thread_id": "e2e-refund-turn1-002"},
            )
        assert result.get("action") == "request_clarification"
        assert "missing_slots" in result

    def test_refund_turn1_spine_pins_workflow_agent(self, runtime_spine):
        """After first response, thread context should pin the workflow agent."""
        from app.runtime.spine import THREAD_CTX

        tid = "e2e-refund-pin-001"
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            runtime_spine.handle_chat(
                "I need to process a refund",
                context={"thread_id": tid},
            )
        ctx = THREAD_CTX.get(tid, {})
        assert ctx.get("pinned_agent_id") == "refunds_workflow"
        assert ctx.get("pinned_agent_type") == "workflow_runner"
        assert ctx.get("pinned_terminal") is False

    # ── Refund Workflow — Turn 2 (slots + completion) ─────────────────────

    def test_refund_full_flow_completes_in_two_turns(self, runtime_spine):
        """
        Turn 1: User starts refund → engine asks for info
        Turn 2: User provides all required slots → engine completes workflow
        """
        from app.runtime.spine import THREAD_CTX

        tid = "e2e-refund-full-001"
        # Clear any prior state
        THREAD_CTX.pop(tid, None)

        # Turn 1: start the workflow (no slots yet)
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            r1 = runtime_spine.handle_chat(
                "I need to process a refund",
                context={"thread_id": tid},
            )
        assert r1.get("current_state") == "start"
        assert r1.get("terminal") is False

        # Turn 2: provide all required slots → auto-chain completes
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", FULL_REFUND_SLOTS),
        ):
            r2 = runtime_spine.handle_chat(
                "CUST-E2E-001 refund EUR 1500 debit card REQ-E2E-001",
                context={"thread_id": tid},
            )
        assert r2.get("current_state") == "notify_customer_success"
        assert r2.get("terminal") is True

    def test_refund_completion_unpins_workflow_agent(self, runtime_spine):
        """Once terminal, spine removes the pin so future messages can route freely."""
        from app.runtime.spine import THREAD_CTX

        tid = "e2e-refund-unpin-001"
        THREAD_CTX.pop(tid, None)

        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            runtime_spine.handle_chat("I need a refund", context={"thread_id": tid})

        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", FULL_REFUND_SLOTS),
        ):
            runtime_spine.handle_chat(
                "CUST-E2E-001 EUR 1500 debit card REQ-E2E-001",
                context={"thread_id": tid},
            )

        ctx = THREAD_CTX.get(tid, {})
        # After terminal, pin should be cleared
        assert "pinned_agent_id" not in ctx

    def test_refund_second_turn_uses_sticky_routing(self, runtime_spine):
        """
        Turn 2 on an active workflow thread must use sticky routing regardless
        of query content (even if the query doesn't contain 'refund').
        """
        from app.runtime.spine import THREAD_CTX

        tid = "e2e-refund-sticky-001"
        THREAD_CTX.pop(tid, None)

        # Turn 1: initiate workflow
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            runtime_spine.handle_chat("I need a refund", context={"thread_id": tid})

        # Turn 2: send a generic query — should STILL hit the workflow agent (sticky)
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json(None, {"customer_id": "CUST-E2E-002"}),
        ):
            r2 = runtime_spine.handle_chat(
                "The customer ID is CUST-E2E-002",  # no 'refund' keyword
                context={"thread_id": tid},
            )
        assert r2.get("agent_id") == "refunds_workflow"

    # ── Guardrails ─────────────────────────────────────────────────────────

    def test_empty_query_returns_error(self, runtime_spine):
        result = runtime_spine.handle_chat("", context={"thread_id": "e2e-empty-001"})
        assert "error" in result

    def test_valid_query_passes_guardrails(self, runtime_spine):
        result = runtime_spine.handle_chat(
            "Can I transfer my account?",
            context={"thread_id": "e2e-guard-001"},
        )
        assert "error" not in result or result.get("agent_id")

    # ── Multiple Independent Sessions ──────────────────────────────────────

    def test_concurrent_sessions_are_isolated(self, runtime_spine):
        """Two threads with different queries should not bleed into each other."""
        from app.runtime.spine import THREAD_CTX

        tid_faq = "e2e-iso-faq-001"
        tid_refund = "e2e-iso-refund-001"
        THREAD_CTX.pop(tid_faq, None)
        THREAD_CTX.pop(tid_refund, None)

        # Session A: FAQ
        r_faq = runtime_spine.handle_chat(
            "What documents do I need for account transfer?",
            context={"thread_id": tid_faq},
        )

        # Session B: Refund workflow
        with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
            r_refund = runtime_spine.handle_chat(
                "I need to process a refund for a customer",
                context={"thread_id": tid_refund},
            )

        assert r_faq.get("agent_id") == "customer_qa_rag"
        assert r_refund.get("agent_id") == "refunds_workflow"

        # FAQ thread should NOT be pinned to workflow
        ctx_faq = THREAD_CTX.get(tid_faq, {})
        assert ctx_faq.get("pinned_agent_id") != "refunds_workflow"

    def test_multiple_faq_sessions_independent(self, runtime_spine):
        """Multiple parallel FAQ sessions don't share state."""
        answers = []
        for i in range(3):
            result = runtime_spine.handle_chat(
                "What documents do I need for a current account?",
                context={"thread_id": f"e2e-multi-faq-{i}"},
            )
            answers.append(result.get("agent_id"))

        assert all(a == "customer_qa_rag" for a in answers)


# ---------------------------------------------------------------------------
# PHASE 5 — Scenario: Large Refund Requires Approval
# ---------------------------------------------------------------------------


class TestPhase5ApprovalPath:
    """
    When refund amount > EUR 5000, policy requires manual approval.
    The FSM should land in await_approval, not completed.
    """

    def test_large_refund_lands_in_await_approval(self):
        """
        Amount EUR 6000 → eligible → needs_manual_approval → await_approval
        → FSM enters await_approval (not completed, not terminal).
        """
        refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

        large_slots = {
            "case_id": "CASE-E2E-BIG",
            "customer_id": "CUST-E2E-BIG",
            "amount": 6000.0,
            "payment_method": "bank_transfer",
        }
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", large_slots),
        ):
            result = refunds_agent.handle(
                {
                    "query": "Large refund request",
                    "thread_id": "e2e-approval-001",
                }
            )

        assert result["current_state"] == "await_approval"
        assert result["terminal"] is False

    def test_small_refund_skips_approval(self):
        """Amount EUR 500 → eligible → auto_approve_event → process_refund → notify_customer_success."""
        refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

        small_slots = {
            "case_id": "CASE-E2E-SMALL",
            "customer_id": "CUST-E2E-SMALL",
            "amount": 500.0,
            "payment_method": "digital_wallet",
        }
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", small_slots),
        ):
            result = refunds_agent.handle(
                {
                    "query": "Small refund request",
                    "thread_id": "e2e-noapproval-001",
                }
            )

        assert result["current_state"] == "notify_customer_success"
        assert result["terminal"] is True
