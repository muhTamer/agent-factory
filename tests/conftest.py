# tests/conftest.py
"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Auto-generate tool operator agents needed by test_tool_operators.py
# ---------------------------------------------------------------------------

TOOL_AGENTS_TO_GENERATE = [
    ("initiate_refund_tool", "initiate_refund"),
    ("create_ticket_tool", "create_ticket"),
    ("lookup_payment_tool", "lookup_payment"),
    ("verify_identity_tool", "verify_identity"),
    ("handoff_tool", "handoff_to_human"),
]


@pytest.fixture(scope="session", autouse=True)
def _generate_tool_operator_agents():
    """Generate tool operator agent stubs so tests can import them."""
    from app.shared.tool_operator import build_agent

    for agent_id, tool_name in TOOL_AGENTS_TO_GENERATE:
        gen_dir = REPO_ROOT / "generated" / agent_id
        if not (gen_dir / "agent.py").exists():
            build_agent(agent_id, {"tool": tool_name}, gen_dir)
