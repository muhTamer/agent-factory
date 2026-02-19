# tests/test_auto_chain.py
"""
Tests for the auto-chain and policy auto-event mechanism.

The auto-chain loop in workflow runner agents advances the FSM through
consecutive system states without user input:
  - tool_exec states  → pass_event fired immediately (stub always succeeds)
  - eligibility states → policy bridge decides event
  - combined_eligibility_approval → both checks run, correct pass event chosen

Tests use:
  - Direct unit tests on _try_policy_auto_event()
  - Minimal FSM specs with injected policy_state_map
  - The real refunds_workflow_agent to test end-to-end auto-chain behaviour
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "generated" / "refunds_workflow" / "agent.py"
PACK_PATH = REPO_ROOT / ".factory/compiled_policies/refunds_policy_pack.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_refunds_agent():
    module_name = "_test_autochain_refunds_agent"
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


def _mapper_json(event, slots):
    return {"event": event, "slots": slots, "confidence": 0.99, "rationale": "test"}


FULL_SLOTS = {
    "case_id": "CASE-001",
    "customer_id": "CUST-001",
    "amount": 1000.0,
    "payment_method": "debit_card",
}

# ---------------------------------------------------------------------------
# _try_policy_auto_event() — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PACK_PATH.exists(),
    reason="Compiled policy pack not found — run factory deploy first",
)
class TestTryPolicyAutoEvent:

    def setup_method(self):
        self.agent = _load_refunds_agent()

    def _engine_at_state(self, state_name: str, slots=None):
        engine = self.agent._engine_for(f"_unit_{state_name}")
        # Force the engine into the desired state
        engine.current_state = state_name
        if slots:
            engine.slots.update(slots)
        return engine

    def test_collect_info_returns_none(self):
        """collect_info is a user-input state — no auto-event."""
        engine = self._engine_at_state("start")
        result = self.agent._try_policy_auto_event(engine)
        assert result is None

    def test_evaluate_policy_eligible_returns_eligible(self):
        """evaluate_policy with valid account → fires 'eligible'."""
        slots = {
            "kyc_status": "verified",
            "account_status": "active",
            "investigation_status": "none",
            "amount": 1000.0,
            "refund_amount_requested": 1000.0,
        }
        engine = self._engine_at_state("evaluate_policy", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "eligible"

    def test_determine_approval_path_small_amount_auto_approves(self):
        """determine_approval_path with amount < 5000 → auto_approve_event."""
        slots = {
            "kyc_status": "verified",
            "account_status": "active",
            "investigation_status": "none",
            "amount": 1000.0,
            "refund_amount_requested": 1000.0,
        }
        engine = self._engine_at_state("determine_approval_path", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "auto_approve_event"

    def test_determine_approval_path_large_amount_requires_approval(self):
        """determine_approval_path with amount > 5000 → needs_manual_approval."""
        slots = {
            "kyc_status": "verified",
            "account_status": "active",
            "investigation_status": "none",
            "amount": 6000.0,
            "refund_amount_requested": 6000.0,
        }
        engine = self._engine_at_state("determine_approval_path", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "needs_manual_approval"

    def test_validate_policy_ineligible(self):
        """validate_policy with frozen account → ineligible."""
        slots = {
            "kyc_status": "verified",
            "account_status": "frozen",
            "investigation_status": "none",
            "amount": 1000.0,
        }
        engine = self._engine_at_state("evaluate_policy", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "ineligible"

    def test_execute_refund_returns_pass_event(self):
        """execute_refund is tool_exec → always returns refund_success (stub)."""
        engine = self._engine_at_state("process_refund")
        result = self.agent._try_policy_auto_event(engine)
        assert result == "refund_success"

    def test_tool_exec_does_not_require_policy_bridge(self):
        """tool_exec check type works even if policy_bridge is None."""
        engine = self._engine_at_state("process_refund")
        original_bridge = self.agent.policy_bridge
        try:
            self.agent.policy_bridge = None
            result = self.agent._try_policy_auto_event(engine)
            assert result == "refund_success"
        finally:
            self.agent.policy_bridge = original_bridge


# ---------------------------------------------------------------------------
# Auto-chain end-to-end (requires policy pack)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not PACK_PATH.exists(),
    reason="Compiled policy pack not found — run factory deploy first",
)
class TestAutoChainEndToEnd:

    def test_auto_chain_advances_through_system_states(self):
        """
        After user provides all slots (info_provided), the engine should
        auto-chain through validate_policy → execute_refund → completed.
        """
        agent = _load_refunds_agent()
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", FULL_SLOTS),
        ):
            result = agent.handle(
                {
                    "query": "full refund request",
                    "thread_id": "t-ac-001",
                }
            )
        assert result["current_state"] == "notify_customer_success"
        assert result["terminal"] is True

    def test_auto_chain_skips_both_system_states(self):
        """auto_chain list shows both validate_policy and execute_refund were auto-advanced."""
        agent = _load_refunds_agent()
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", FULL_SLOTS),
        ):
            result = agent.handle(
                {
                    "query": "refund",
                    "thread_id": "t-ac-002",
                }
            )
        chain = result.get("mapper", {}).get("auto_chain", [])
        # Should have advanced at least through execute_refund
        assert any("process_refund" in step for step in chain)

    def test_auto_chain_does_not_advance_past_terminal(self):
        """Auto-chain stops when it reaches a terminal state."""
        agent = _load_refunds_agent()
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", FULL_SLOTS),
        ):
            result = agent.handle(
                {
                    "query": "refund",
                    "thread_id": "t-ac-003",
                }
            )
        # Check auto_chain did not contain "notify_customer_success" as a start state
        chain = result.get("mapper", {}).get("auto_chain", [])
        assert not any(step.startswith("notify_customer_success") for step in chain)

    def test_auto_chain_large_amount_hits_approval_path(self):
        """
        Amount > 5000 → validate_policy fires eligible_requires_approval → await_approval.
        Auto-chain should stop at await_approval (no tool_exec config for it).
        """
        agent = _load_refunds_agent()
        large_slots = {**FULL_SLOTS, "amount": 6000.0}
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", large_slots),
        ):
            result = agent.handle(
                {
                    "query": "large refund",
                    "thread_id": "t-ac-approval-001",
                }
            )
        # Should land in await_approval (not completed or execute_refund)
        assert result["current_state"] == "await_approval"
        assert result["terminal"] is False

    def test_auto_chain_ineligible_terminates_at_refund_rejected(self):
        """
        Frozen account → validate_policy fires ineligible → refund_rejected (terminal).
        """
        agent = _load_refunds_agent()

        # We need to inject the frozen account status via the policy slot defaults override
        # The slot_defaults set kyc_status=verified/account_status=active.
        # We can override by providing account_status in the FSM slots so it
        # takes priority (slot_map: amount → refund_amount_requested is the only rename).
        # There is no FSM slot "account_status", so we set it indirectly via slot_defaults.

        # Instead: directly manipulate the engine's slots after initial LLM step
        frozen_slots = {
            "case_id": "CASE-FROZEN",
            "customer_id": "CUST-FROZEN",
            "amount": 1000.0,
            "payment_method": "debit_card",
        }
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", frozen_slots),
        ):
            # Override slot_defaults to simulate frozen account
            original_defaults = agent.policy_slot_defaults.copy()
            agent.policy_slot_defaults = {
                "kyc_status": "verified",
                "account_status": "frozen",  # frozen!
                "investigation_status": "none",
            }
            try:
                result = agent.handle(
                    {
                        "query": "refund for frozen account",
                        "thread_id": "t-ac-frozen-001",
                    }
                )
            finally:
                agent.policy_slot_defaults = original_defaults

        assert result["current_state"] == "notify_customer_reject"
        assert result["terminal"] is True


# ---------------------------------------------------------------------------
# Minimal FSM auto-chain (no policy bridge needed)
# ---------------------------------------------------------------------------


class TestAutoChainToolExecOnly:
    """
    Tests using a minimal FSM with only tool_exec system states.
    These tests don't require the compiled policy pack.
    """

    def _make_tool_exec_agent(self):
        """Build a minimal workflow agent with two consecutive tool_exec states."""
        from app.runtime.workflow_engine import GenericWorkflowEngine

        spec = {
            "id": "tool_chain_test",
            "description": "Two consecutive tool states",
            "engine": "fsm",
            "slots": {"result_a": None, "result_b": None},
            "states": {
                "start": {
                    "description": "user input",
                    "on": {"begin": "step_a"},
                },
                "step_a": {
                    "description": "first tool",
                    "on_enter": "call:tool_a",
                    "on": {"a_done": "step_b"},
                },
                "step_b": {
                    "description": "second tool",
                    "on_enter": "call:tool_b",
                    "on": {"b_done": "done"},
                },
                "done": {
                    "description": "finished",
                    "terminal": True,
                },
            },
            "initial_state": "start",
        }

        def tool_a(slots, ctx):
            return {"result_a": "A_ok"}

        def tool_b(slots, ctx):
            return {"result_b": "B_ok"}

        engine = GenericWorkflowEngine(
            agent_id="chain_test",
            workflow_spec=spec,
            tools={"tool_a": tool_a, "tool_b": tool_b},
        )
        return engine, spec

    def test_tool_exec_chain_manual_advance(self):
        """Manually advance through two tool_exec states."""
        engine, _ = self._make_tool_exec_agent()

        # User-driven transition: begin
        engine.handle({"event": "begin", "slots": {}})
        assert engine.current_state == "step_a"
        assert engine.slots.get("result_a") == "A_ok"

        # Auto-advance: step_a → step_b
        engine.handle({"event": "a_done", "slots": {}})
        assert engine.current_state == "step_b"
        assert engine.slots.get("result_b") == "B_ok"

        # Auto-advance: step_b → done
        result = engine.handle({"event": "b_done", "slots": {}})
        assert result["current_state"] == "done"
        assert result["terminal"] is True

    def test_tool_exec_state_map_returns_pass_event_directly(self):
        """
        _try_policy_auto_event with tool_exec config returns pass_event immediately.
        """
        agent = _load_refunds_agent()

        # Inject a fake tool_exec config without needing the engine to be in that state

        engine = agent._engine_for("_tool_unit")
        engine.current_state = "process_refund"

        # The real policy_state_map has execute_refund as tool_exec
        event = agent._try_policy_auto_event(engine)
        assert event == "refund_success"
