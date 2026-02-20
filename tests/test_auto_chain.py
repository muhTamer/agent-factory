# tests/test_auto_chain.py
"""
Tests for the auto-chain and policy auto-event mechanism.

The auto-chain loop in workflow runner agents advances the FSM through
consecutive system states without user input:
  - tool_exec states  → pass_event fired immediately (stub always succeeds)
  - eligibility states → policy bridge decides event
  - approval_needed   → policy bridge checks amount threshold

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
    "customer_id": "CUST-001",
    "payment_id": "PAY-001",
    "amount": 1000.0,
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

    def test_start_returns_none(self):
        """start is a user-input state — no auto-event."""
        engine = self._engine_at_state("start")
        result = self.agent._try_policy_auto_event(engine)
        assert result is None

    def test_eligibility_check_eligible_returns_eligible(self):
        """eligibility_check with valid account → fires 'eligible'."""
        slots = {
            "kyc_status": "verified",
            "account_status": "active",
            "investigation_status": "none",
            "amount": 1000.0,
            "refund_amount_requested": 1000.0,
        }
        engine = self._engine_at_state("eligibility_check", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "eligible"

    def test_eligibility_check_ineligible_frozen_account(self):
        """eligibility_check with frozen account → ineligible."""
        slots = {
            "kyc_status": "verified",
            "account_status": "frozen",
            "investigation_status": "none",
            "amount": 1000.0,
        }
        engine = self._engine_at_state("eligibility_check", slots)
        result = self.agent._try_policy_auto_event(engine)
        assert result == "ineligible"

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

    def test_execute_refund_returns_pass_event(self):
        """execute_refund is tool_exec → always returns success (stub)."""
        engine = self._engine_at_state("execute_refund")
        result = self.agent._try_policy_auto_event(engine)
        assert result == "success"

    def test_tool_exec_does_not_require_policy_bridge(self):
        """tool_exec check type works even if policy_bridge is None."""
        engine = self._engine_at_state("execute_refund")
        original_bridge = self.agent.policy_bridge
        try:
            self.agent.policy_bridge = None
            result = self.agent._try_policy_auto_event(engine)
            assert result == "success"
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
        After user provides all slots, the engine should auto-chain through
        eligibility_check → determine_approval_path → execute_refund → completed.
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
        assert result["current_state"] == "completed"
        assert result["terminal"] is True

    def test_auto_chain_skips_both_system_states(self):
        """auto_chain list shows eligibility_check and execute_refund were auto-advanced."""
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
        assert any("execute_refund" in step for step in chain)

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
        chain = result.get("mapper", {}).get("auto_chain", [])
        assert not any(step.startswith("completed") for step in chain)

    def test_auto_chain_large_amount_hits_approval_path(self):
        """
        Amount > 5000 → determine_approval_path fires needs_manual_approval → await_approval.
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
        assert result["current_state"] == "await_approval"
        assert result["terminal"] is False

    def test_auto_chain_ineligible_terminates_at_deny_refund(self):
        """
        Frozen account → eligibility_check fires ineligible → deny_refund (terminal).
        """
        agent = _load_refunds_agent()
        frozen_slots = {
            "customer_id": "CUST-FROZEN",
            "payment_id": "PAY-FROZEN",
            "amount": 1000.0,
        }
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value=_mapper_json("validated", frozen_slots),
        ):
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

        assert result["current_state"] == "deny_refund"
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

        engine.handle({"event": "begin", "slots": {}})
        assert engine.current_state == "step_a"
        assert engine.slots.get("result_a") == "A_ok"

        engine.handle({"event": "a_done", "slots": {}})
        assert engine.current_state == "step_b"
        assert engine.slots.get("result_b") == "B_ok"

        result = engine.handle({"event": "b_done", "slots": {}})
        assert result["current_state"] == "done"
        assert result["terminal"] is True

    def test_tool_exec_state_map_returns_pass_event_directly(self):
        """
        _try_policy_auto_event with tool_exec config returns pass_event immediately.
        """
        agent = _load_refunds_agent()
        engine = agent._engine_for("_tool_unit")
        engine.current_state = "execute_refund"

        event = agent._try_policy_auto_event(engine)
        assert event == "success"
