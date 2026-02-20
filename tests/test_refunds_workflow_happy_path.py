# tests/test_refunds_workflow_happy_path.py
"""
Happy-path end-to-end tests for refunds_workflow_agent.

Refunds workflow state machine:
  start  →  eligibility_check  →  determine_approval_path  →  execute_refund  →  completed
  (user)       (policy auto)          (policy auto)            (tool auto)      (terminal)

Strategy:
  - Mock `app.runtime.workflow_mapper.chat_json` so the LLM mapper
    returns a predictable dict without a real API call.
  - The real map_query_to_event_and_slots() parses that dict into a MapResult.
  - The agent's hard guard enforces required slots before transitioning.
  - After the user-driven transition (start → eligibility_check),
    the auto-chain loop advances through the system states automatically.

All required slots: customer_id, payment_id, amount
"""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_PATH = REPO_ROOT / "generated" / "refunds_workflow" / "agent.py"
PACK_PATH = REPO_ROOT / ".factory/compiled_policies/refunds_policy_pack.json"

pytestmark = pytest.mark.skipif(
    not PACK_PATH.exists(),
    reason="Compiled policy pack not found — run factory deploy first",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FULL_SLOTS = {
    "customer_id": "CUST-001",
    "payment_id": "PAY-001",
    "amount": 1000.0,
}


def _load_refunds_agent():
    module_name = "_test_refunds_workflow"
    spec = importlib.util.spec_from_file_location(module_name, AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


def _mapper_json(event, slots):
    """Build the dict that chat_json returns (workflow_mapper parses this)."""
    return {
        "event": event,
        "slots": slots,
        "confidence": 0.99,
        "rationale": "test mock",
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_refunds_agent_loads_without_error():
    agent = _load_refunds_agent()
    assert agent.ready is True


def test_refunds_agent_has_policy_bridge():
    agent = _load_refunds_agent()
    assert agent.policy_bridge is not None


def test_refunds_agent_policy_state_map_configured():
    agent = _load_refunds_agent()
    assert "eligibility_check" in agent.policy_state_map
    assert "determine_approval_path" in agent.policy_state_map
    assert "execute_refund" in agent.policy_state_map


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_fresh_engine_starts_in_start():
    agent = _load_refunds_agent()
    engine = agent._engine_for("test-init")
    assert engine.current_state == "start"


def test_first_handle_without_event_returns_clarification():
    agent = _load_refunds_agent()
    with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
        result = agent.handle({"query": "I need a refund", "thread_id": "t-init"})
    assert result["action"] == "request_clarification"
    assert result["current_state"] == "start"


def test_clarification_lists_required_slots():
    agent = _load_refunds_agent()
    with patch("app.runtime.workflow_mapper.chat_json", return_value=_mapper_json(None, {})):
        result = agent.handle({"query": "I need a refund", "thread_id": "t-missing"})
    missing = result.get("missing_slots", [])
    for slot in ("customer_id", "payment_id", "amount"):
        assert slot in missing, f"Expected {slot!r} in missing_slots"


# ---------------------------------------------------------------------------
# Happy path — full flow in one turn
# ---------------------------------------------------------------------------


def test_full_flow_reaches_completed():
    """All slots provided in one turn → auto-chain through system states → completed."""
    agent = _load_refunds_agent()
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle(
            {
                "query": "Refund EUR 1000 for CUST-001 payment PAY-001",
                "thread_id": "t-happy-001",
            }
        )
    assert result["current_state"] == "completed"
    assert result["terminal"] is True


def test_full_flow_slots_contain_user_provided_data():
    """After completion, FSM slots should still hold user-provided values."""
    agent = _load_refunds_agent()
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle(
            {
                "query": "refund request",
                "thread_id": "t-slots-001",
            }
        )
    assert result["slots"].get("customer_id") == "CUST-001"
    assert result["slots"].get("amount") == 1000.0


def test_full_flow_auto_chain_reported_in_mapper():
    """auto_chain key in mapper shows which system states were advanced."""
    agent = _load_refunds_agent()
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle(
            {
                "query": "refund request",
                "thread_id": "t-chain-001",
            }
        )
    mapper = result.get("mapper", {})
    assert "auto_chain" in mapper
    chain = mapper["auto_chain"]
    assert len(chain) > 0


# ---------------------------------------------------------------------------
# Hard guard: partial slots prevent transition
# ---------------------------------------------------------------------------


def test_partial_slots_stay_in_start():
    """If required slots are missing, event is forced to None — stays in start."""
    agent = _load_refunds_agent()
    partial = {"customer_id": "C1"}  # missing payment_id and amount
    with patch(
        "app.runtime.workflow_mapper.chat_json", return_value=_mapper_json("validated", partial)
    ):
        result = agent.handle(
            {
                "query": "I need a refund",
                "thread_id": "t-partial-001",
            }
        )
    assert result["current_state"] == "start"


def test_partial_slots_action_is_request_clarification():
    agent = _load_refunds_agent()
    partial = {"customer_id": "C1"}
    with patch(
        "app.runtime.workflow_mapper.chat_json", return_value=_mapper_json("validated", partial)
    ):
        result = agent.handle({"query": "refund", "thread_id": "t-partial-002"})
    assert result["action"] == "request_clarification"


# ---------------------------------------------------------------------------
# Full required slots — advance and complete
# ---------------------------------------------------------------------------


def test_all_required_slots_advance_from_start():
    """All required slots provided → workflow advances past start."""
    agent = _load_refunds_agent()
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle(
            {
                "query": "I need a refund for payment PAY-001",
                "thread_id": "t-full-001",
            }
        )
    assert result["current_state"] != "start"


def test_all_required_slots_reach_completed():
    agent = _load_refunds_agent()
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle(
            {
                "query": "refund payment PAY-001 amount 1000",
                "thread_id": "t-full-002",
            }
        )
    assert result["current_state"] == "completed"
    assert result["terminal"] is True


# ---------------------------------------------------------------------------
# Multi-turn slot accumulation
# ---------------------------------------------------------------------------


def test_multi_turn_slots_accumulate():
    """Slots provided across turns are retained by the engine."""
    agent = _load_refunds_agent()
    tid = "t-multi-001"

    # Turn 1: partial info — no event
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json(None, {"customer_id": "CUST-001"}),
    ):
        r1 = agent.handle({"query": "I need a refund", "thread_id": tid})

    assert r1["current_state"] == "start"
    assert agent.engines[tid].slots["customer_id"] == "CUST-001"

    # Turn 2: remaining slots + event
    remaining = {
        "payment_id": "PAY-001",
        "amount": 500.0,
    }
    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", remaining),
    ):
        r2 = agent.handle(
            {
                "query": "payment PAY-001 amount 500",
                "thread_id": tid,
            }
        )

    assert r2["current_state"] == "completed"
    assert r2["terminal"] is True


def test_multi_turn_customer_id_persists():
    agent = _load_refunds_agent()
    tid = "t-multi-002"

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json(None, {"customer_id": "CUST-999"}),
    ):
        agent.handle({"query": "hi", "thread_id": tid})

    engine = agent.engines[tid]
    assert engine.slots["customer_id"] == "CUST-999"


# ---------------------------------------------------------------------------
# Thread isolation
# ---------------------------------------------------------------------------


def test_thread_isolation_separate_states():
    """Two thread IDs should maintain independent FSM states."""
    agent = _load_refunds_agent()

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        r_a = agent.handle({"query": "full request", "thread_id": "t-iso-A"})

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json(None, {"customer_id": "C2"}),
    ):
        r_b = agent.handle({"query": "partial request", "thread_id": "t-iso-B"})

    assert r_a["current_state"] == "completed"
    assert r_b["current_state"] == "start"


def test_thread_isolation_slots_not_shared():
    agent = _load_refunds_agent()

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json(None, {"customer_id": "CUST-A"}),
    ):
        agent.handle({"query": "a", "thread_id": "t-iso-slots-A"})

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json(None, {"customer_id": "CUST-B"}),
    ):
        agent.handle({"query": "b", "thread_id": "t-iso-slots-B"})

    assert agent.engines["t-iso-slots-A"].slots["customer_id"] == "CUST-A"
    assert agent.engines["t-iso-slots-B"].slots["customer_id"] == "CUST-B"


# ---------------------------------------------------------------------------
# Terminal state — no further transitions
# ---------------------------------------------------------------------------


def test_completed_state_returns_terminal_on_second_call():
    agent = _load_refunds_agent()
    tid = "t-terminal-001"

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        agent.handle({"query": "refund", "thread_id": tid})

    with patch(
        "app.runtime.workflow_mapper.chat_json",
        return_value=_mapper_json("validated", FULL_SLOTS),
    ):
        result = agent.handle({"query": "another request", "thread_id": tid})

    assert result["current_state"] == "completed"
    assert result["terminal"] is True


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------


def test_refunds_agent_metadata():
    agent = _load_refunds_agent()
    meta = agent.metadata()
    assert meta["id"] == "refunds_workflow"
    assert meta["type"] == "workflow_runner"
    assert meta["ready"] is True
    assert "multi_turn" in meta["capabilities"]
