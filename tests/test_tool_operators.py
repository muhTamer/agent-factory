# tests/test_tool_operators.py
"""
Happy-path tests for the five generated tool operator agents.

Each tool operator agent:
  - reads its config.json to find its tool name
  - loads the STUB_RESPONSE for that tool
  - returns the stub dict on any handle() call

Tests verify:
  - Agent loads without error
  - metadata() returns correct id, type, ready
  - handle() returns the stub response with correct fields
"""
import importlib.util
import sys
from pathlib import Path

import pytest

from app.runtime.tools.stub_tools import STUB_RESPONSES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool_agent(agent_id: str):
    """Dynamically load a generated tool operator agent and call load()."""
    agent_path = REPO_ROOT / "generated" / agent_id / "agent.py"
    module_name = f"_test_tool_{agent_id}"

    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    agent = mod.Agent()
    agent.load({})
    return agent


TOOL_AGENTS = [
    ("tool_initiate_refund", "initiate_refund"),
    ("tool_lookup_customer", "lookup_customer"),
    ("tool_verify_identity", "verify_identity"),
    ("tool_create_ticket", "create_ticket"),
    ("tool_handoff_to_human", "handoff_to_human"),
]

AGENT_IDS = [agent_id for agent_id, _ in TOOL_AGENTS]

# ---------------------------------------------------------------------------
# Generic tests (parametrized over all tool operators)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_loads_without_error(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    assert agent is not None


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_metadata_id(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    meta = agent.metadata()
    assert meta["id"] == agent_id


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_metadata_type(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    meta = agent.metadata()
    assert meta["type"] == "tool_operator"


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_metadata_ready(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    meta = agent.metadata()
    assert meta["ready"] is True


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_handle_includes_agent_id(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    result = agent.handle({"query": "execute"})
    assert result.get("agent_id") == agent_id


@pytest.mark.parametrize("agent_id,tool_name", TOOL_AGENTS)
def test_tool_operator_handle_includes_tool_name(agent_id, tool_name):
    agent = _load_tool_agent(agent_id)
    result = agent.handle({"query": "execute"})
    assert "tool" in result


# ---------------------------------------------------------------------------
# Specific stub-response content assertions
# ---------------------------------------------------------------------------


def test_initiate_refund_returns_refund_id():
    agent = _load_tool_agent("tool_initiate_refund")
    result = agent.handle({})
    stub = STUB_RESPONSES["initiate_refund"]
    assert result.get("refund_id") == stub["refund_id"]
    assert result.get("refund_initiated") == stub["refund_initiated"]
    assert result.get("status") == stub["status"]


def test_verify_identity_returns_kyc_verified():
    agent = _load_tool_agent("tool_verify_identity")
    result = agent.handle({})
    stub = STUB_RESPONSES["verify_identity"]
    assert result.get("kyc_status") == stub["kyc_status"]
    assert result.get("identity_verified") == stub["identity_verified"]
    assert result.get("status") == stub["status"]


def test_lookup_customer_returns_active_account():
    agent = _load_tool_agent("tool_lookup_customer")
    result = agent.handle({})
    stub = STUB_RESPONSES["lookup_customer"]
    assert result.get("account_status") == stub["account_status"]
    assert result.get("kyc_status") == stub["kyc_status"]
    assert result.get("status") == stub["status"]


def test_create_ticket_returns_ticket_id():
    agent = _load_tool_agent("tool_create_ticket")
    result = agent.handle({})
    stub = STUB_RESPONSES["create_ticket"]
    assert result.get("ticket_id") == stub["ticket_id"]
    assert result.get("status") == stub["status"]


def test_handoff_to_human_returns_handed_off():
    agent = _load_tool_agent("tool_handoff_to_human")
    result = agent.handle({})
    stub = STUB_RESPONSES["handoff_to_human"]
    assert result.get("handed_off") == stub["handed_off"]
    assert result.get("status") == stub["status"]


# ---------------------------------------------------------------------------
# STUB_TOOLS (workflow engine contract)
# ---------------------------------------------------------------------------


def test_stub_tools_verify_identity():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    result = STUB_TOOLS["verify_identity"]({}, {})
    assert result["kyc_status"] == "verified"
    assert result["identity_verified"] is True


def test_stub_tools_initiate_refund():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    result = STUB_TOOLS["initiate_refund"]({}, {})
    assert result["refund_id"] == "DEMO-REF-001"
    assert result["refund_status"] == "success"
    assert result["refund_initiated"] is True


def test_stub_tools_lookup_customer():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    result = STUB_TOOLS["lookup_customer"]({}, {})
    assert result["account_status"] == "active"
    assert result["kyc_status"] == "verified"
    assert result["customer_found"] is True


def test_stub_tools_create_ticket():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    result = STUB_TOOLS["create_ticket"]({}, {})
    assert result["ticket_id"] == "DEMO-TKT-001"
    assert result["ticket_status"] == "created"


def test_stub_tools_handoff_to_human():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    result = STUB_TOOLS["handoff_to_human"]({}, {})
    assert result["handed_off"] is True
    assert result["handoff_agent"] == "human_ops_team"


def test_stub_tools_lookup_payment():
    from app.runtime.tools.stub_tools import STUB_TOOLS

    slots = {"amount": 250.0}
    result = STUB_TOOLS["lookup_payment"](slots, {})
    assert result["payment_found"] is True
    assert result["settlement_status"] == "settled"
    assert result["transaction_age_days"] == 5
    assert result["original_transaction_amount"] == 250.0
