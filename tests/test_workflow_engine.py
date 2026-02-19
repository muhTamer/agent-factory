# tests/test_workflow_engine.py
"""
Happy-path unit tests for GenericWorkflowEngine.

Covers:
- FSM initialization and initial state entry
- Event-driven state transitions
- Slot updates across multiple calls
- Terminal state detection
- Missing required slots → request_clarification action
- Tool invocation via call:<tool_name> actions
- Unknown/invalid events are silently ignored
"""
from app.runtime.workflow_engine import GenericWorkflowEngine

# ---------------------------------------------------------------------------
# Minimal test fixtures
# ---------------------------------------------------------------------------

SIMPLE_SPEC = {
    "id": "test_simple",
    "description": "Simple test workflow",
    "engine": "fsm",
    "slots": {
        "customer_id": {"type": "string", "required": True, "description": "Customer ID"},
        "amount": {"type": "number", "required": True, "description": "Amount"},
        "reason": {"type": "string", "required": False, "description": "Optional reason"},
    },
    "states": {
        "collect_info": {
            "description": "Collect customer info",
            "on": {"info_provided": "process"},
        },
        "process": {
            "description": "Process the request",
            "on": {"success": "completed", "failure": "failed"},
        },
        "completed": {
            "description": "Done successfully",
            "terminal": True,
        },
        "failed": {
            "description": "Request failed",
            "terminal": True,
        },
    },
    "initial_state": "collect_info",
}


def _make_engine(spec=None, tools=None):
    return GenericWorkflowEngine(
        agent_id="test_agent",
        workflow_spec=spec or SIMPLE_SPEC,
        tools=tools or {},
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_engine_starts_at_initial_state():
    eng = _make_engine()
    assert eng.current_state == "collect_info"


def test_initial_state_is_not_terminal():
    eng = _make_engine()
    assert eng.states["collect_info"]["terminal"] is False


def test_history_records_initial_state_on_enter():
    eng = _make_engine()
    assert len(eng.history) == 1
    assert eng.history[0]["state"] == "collect_info"


def test_slots_initialized_to_none():
    eng = _make_engine()
    assert eng.slots["customer_id"] is None
    assert eng.slots["amount"] is None
    assert eng.slots["reason"] is None


# ---------------------------------------------------------------------------
# Event-driven transitions
# ---------------------------------------------------------------------------


def test_valid_event_transitions_to_next_state():
    eng = _make_engine()
    eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 100.0}})
    assert eng.current_state == "process"


def test_transition_appends_to_history():
    eng = _make_engine()
    eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 100.0}})
    assert len(eng.history) == 2
    assert eng.history[1]["state"] == "process"


def test_success_event_reaches_completed_terminal():
    eng = _make_engine()
    eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 100.0}})
    res = eng.handle({"event": "success", "slots": {}})
    assert res["current_state"] == "completed"
    assert res["terminal"] is True


def test_failure_event_reaches_failed_terminal():
    eng = _make_engine()
    eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 100.0}})
    res = eng.handle({"event": "failure", "slots": {}})
    assert res["current_state"] == "failed"
    assert res["terminal"] is True


def test_unknown_event_is_silently_ignored():
    eng = _make_engine()
    eng.handle({"event": "nonexistent_event", "slots": {}})
    assert eng.current_state == "collect_info"


def test_event_in_terminal_state_returns_terminal_action():
    eng = _make_engine()
    eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 100.0}})
    eng.handle({"event": "success", "slots": {}})
    # Extra event after terminal → engine stays terminal
    res = eng.handle({"event": "info_provided", "slots": {}})
    assert res["current_state"] == "completed"
    assert res["action"] == "terminal"


# ---------------------------------------------------------------------------
# Slot updates
# ---------------------------------------------------------------------------


def test_slot_updates_applied_on_each_call():
    eng = _make_engine()
    eng.handle({"event": None, "slots": {"customer_id": "C1"}})
    assert eng.slots["customer_id"] == "C1"


def test_slots_accumulate_across_multiple_calls():
    eng = _make_engine()
    eng.handle({"event": None, "slots": {"customer_id": "C1"}})
    eng.handle({"event": None, "slots": {"amount": 50.0}})
    assert eng.slots["customer_id"] == "C1"
    assert eng.slots["amount"] == 50.0


def test_slot_update_with_event_applies_before_transition():
    eng = _make_engine()
    res = eng.handle({"event": "info_provided", "slots": {"customer_id": "C1", "amount": 75.0}})
    assert res["slots"]["customer_id"] == "C1"
    assert res["slots"]["amount"] == 75.0
    assert res["current_state"] == "process"


# ---------------------------------------------------------------------------
# request_clarification (no event)
# ---------------------------------------------------------------------------


def test_no_event_returns_request_clarification():
    eng = _make_engine()
    res = eng.handle({"event": None, "slots": {}})
    assert res["action"] == "request_clarification"


def test_missing_required_slots_listed_in_response():
    eng = _make_engine()
    res = eng.handle({"event": None, "slots": {}})
    assert "customer_id" in res["missing_slots"]
    assert "amount" in res["missing_slots"]


def test_optional_slot_not_in_missing_slots():
    eng = _make_engine()
    res = eng.handle({"event": None, "slots": {}})
    assert "reason" not in res["missing_slots"]


def test_all_required_slots_provided_no_missing():
    eng = _make_engine()
    res = eng.handle({"event": None, "slots": {"customer_id": "C1", "amount": 100.0}})
    assert res["missing_slots"] == []


# ---------------------------------------------------------------------------
# Tool invocation via call:<tool_name>
# ---------------------------------------------------------------------------


def _make_tool_spec(action: str):
    return {
        "id": "tool_test",
        "description": "test",
        "engine": "fsm",
        "slots": {},
        "states": {
            "start": {
                "description": "start",
                "on_enter": action,
                "on": {"done": "end"},
            },
            "end": {"description": "end", "terminal": True},
        },
        "initial_state": "start",
    }


def test_tool_called_on_state_enter():
    called = []

    def my_tool(slots, context):
        called.append(True)
        return {}

    spec = _make_tool_spec("call:my_tool")
    GenericWorkflowEngine(
        agent_id="t",
        workflow_spec=spec,
        tools={"my_tool": my_tool},
    )
    assert called == [True]


def test_tool_return_dict_updates_slots():
    def my_tool(slots, context):
        return {"tool_result": "ok", "extra": 42}

    spec = _make_tool_spec("call:my_tool")
    eng = GenericWorkflowEngine(
        agent_id="t",
        workflow_spec=spec,
        tools={"my_tool": my_tool},
    )
    assert eng.slots.get("tool_result") == "ok"
    assert eng.slots.get("extra") == 42


def test_missing_tool_logs_warning_without_raising():
    spec = _make_tool_spec("call:nonexistent_tool")
    # Should not raise — unknown tools emit a warning and are skipped
    eng = GenericWorkflowEngine(agent_id="t", workflow_spec=spec, tools={})
    assert eng.current_state == "start"


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_contains_required_keys():
    eng = _make_engine()
    res = eng.handle({"event": None, "slots": {}})
    for key in ("current_state", "terminal", "slots", "history"):
        assert key in res, f"Missing key: {key}"


def test_metadata_reflects_current_state():
    eng = _make_engine()
    meta = eng.metadata()
    assert meta["state"] == "collect_info"
    assert meta["ready"] is True
    assert meta["type"] == "workflow_runner"
