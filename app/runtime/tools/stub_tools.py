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
        "original_transaction_amount": slots.get("amount", 10000.0),
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


def _update_profile(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "profile_updated": True,
        "field_changed": slots.get("field", "address"),
        "message": "[DEMO] Profile updated successfully.",
    }


def _update_account(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "account_updated": True,
        "action": slots.get("action", "update"),
        "message": "[DEMO] Account action completed successfully.",
    }


def _generate_statement(
    slots: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "statement_generated": True,
        "period": slots.get("period", "3 months"),
        "format": "PDF",
        "message": "[DEMO] Statement generated and available for download.",
    }


def _freeze_account(slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "account_frozen": True,
        "freeze_reason": slots.get("reason", "security"),
        "message": "[DEMO] Account frozen pending investigation.",
    }


def _initiate_transfer(
    slots: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "transfer_initiated": True,
        "transfer_id": "DEMO-TRF-001",
        "amount": slots.get("amount", 0),
        "message": "[DEMO] Transfer initiated successfully.",
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
    "update_profile": _update_profile,
    "update_account": _update_account,
    "generate_statement": _generate_statement,
    "freeze_account": _freeze_account,
    "initiate_transfer": _initiate_transfer,
}

# Map abstract API-style names from factory_spec.json to concrete
# stub tool names so agents can find their tools regardless of
# which naming convention the spec uses.
TOOL_ALIASES: Dict[str, str] = {
    "PaymentsAPI": "lookup_payment",
    "IdentityVerificationAPI": "verify_identity",
    "TicketingSystemAPI": "create_ticket",
    "EscalationWorkflowTool": "create_ticket",
    "CRM": "lookup_customer",
    "AuditLogger": "create_ticket",
    "ConversationLogger": "create_ticket",
    "RAG_Retriever": "",  # handled internally by the engine, not a callable tool
    "AccountManagementAPI": "update_account",
    "ProfileAPI": "update_profile",
    "StatementAPI": "generate_statement",
    "TransferAPI": "initiate_transfer",
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
    "update_profile": {
        "status": "success",
        "profile_updated": True,
        "message": "[DEMO] Profile updated successfully.",
    },
    "update_account": {
        "status": "success",
        "account_updated": True,
        "message": "[DEMO] Account action completed successfully.",
    },
    "generate_statement": {
        "status": "success",
        "statement_generated": True,
        "message": "[DEMO] Statement generated and available for download.",
    },
    "freeze_account": {
        "status": "success",
        "account_frozen": True,
        "message": "[DEMO] Account frozen pending investigation.",
    },
    "initiate_transfer": {
        "status": "success",
        "transfer_initiated": True,
        "transfer_id": "DEMO-TRF-001",
        "message": "[DEMO] Transfer initiated successfully.",
    },
}
