# app/runtime/tools/stub_tools.py
"""
Demo stub tool implementations.

Each tool follows the workflow engine contract:
    tool(slots: dict, context: dict) -> dict   # slot updates applied to FSM

All stubs return happy-path results so the workflow progresses end-to-end
during demos without requiring real backend integrations.
"""
from __future__ import annotations

from typing import Any, Dict


def _verify_identity(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kyc_status": "verified",
        "identity_verified": True,
    }


def _lookup_payment(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "payment_found": True,
        "settlement_status": "settled",
        "original_transaction_amount": slots.get("amount", 100.0),
        "transaction_age_days": 5,
    }


def _initiate_refund(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "refund_id": "DEMO-REF-001",
        "refund_status": "success",
        "refund_initiated": True,
    }


def _create_ticket(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticket_id": "DEMO-TKT-001",
        "ticket_status": "created",
    }


def _handoff_to_human(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "handed_off": True,
        "handoff_agent": "human_ops_team",
    }


def _lookup_customer(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "account_status": "active",
        "kyc_status": "verified",
        "customer_found": True,
    }


# Callable stubs passed to GenericWorkflowEngine(tools=STUB_TOOLS)
# Keyed by the tool name used in FSM actions: call:<tool_name>
STUB_TOOLS: Dict[str, Any] = {
    "verify_identity": _verify_identity,
    "lookup_payment": _lookup_payment,
    "initiate_refund": _initiate_refund,
    "create_ticket": _create_ticket,
    "handoff_to_human": _handoff_to_human,
    "lookup_customer": _lookup_customer,
}

# Agent-level responses returned by tool_operator agents when called as standalone agents
STUB_RESPONSES: Dict[str, Dict[str, Any]] = {
    "verify_identity": {
        "status": "verified",
        "kyc_status": "verified",
        "identity_verified": True,
        "message": "[DEMO] Identity verified successfully.",
    },
    "lookup_payment": {
        "status": "found",
        "payment_found": True,
        "settlement_status": "settled",
        "transaction_age_days": 5,
        "message": "[DEMO] Payment record located and valid.",
    },
    "initiate_refund": {
        "status": "success",
        "refund_id": "DEMO-REF-001",
        "refund_initiated": True,
        "message": "[DEMO] Refund initiated successfully.",
    },
    "create_ticket": {
        "status": "created",
        "ticket_id": "DEMO-TKT-001",
        "message": "[DEMO] Support ticket created.",
    },
    "handoff_to_human": {
        "status": "handed_off",
        "handed_off": True,
        "message": "[DEMO] Case handed off to human operator.",
    },
    "lookup_customer": {
        "status": "found",
        "account_status": "active",
        "kyc_status": "verified",
        "message": "[DEMO] Customer record found.",
    },
}
