# tests/fixtures/mock_mcp_server.py
"""
Mock MCP server for manual & integration testing.

Exposes all domain tools (mirrors the 6 built-in stubs) plus utility tools.
Supports multiple scenarios via special IDs / prefixes:
  - customer_id starting with "SUSPENDED" → suspended account
  - customer_id starting with "FROZEN"    → frozen account
  - customer_id starting with "FAIL"      → lookup failure
  - transaction_id starting with "OLD"     → transaction_age_days = 120
  - transaction_id starting with "UNSETTLED" → settlement_status = "pending"
  - amount > 5000                          → approval_required = True
  - severity = "high"                      → auto-escalate in triage

Run with:  python tests/fixtures/mock_mcp_server.py
Uses stdio transport (stdin/stdout).
"""

import json
from mcp.server.fastmcp import FastMCP

server = FastMCP("demo-server")


# ── Utility tools ───────────────────────────────────────────────────────


@server.tool()
def echo(message: str) -> str:
    """Echo back the message. Useful for connectivity testing."""
    return message


@server.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the result as a string."""
    return str(a + b)


# ── Customer domain tools ──────────────────────────────────────────────


@server.tool()
def lookup_customer(customer_id: str) -> str:
    """Look up a customer by their ID.

    Returns account_status, kyc_status, and customer_found.
    Use customer_id prefix to simulate scenarios:
      - "SUSPENDED-..." → suspended account
      - "FROZEN-..."    → frozen account
      - "FAIL-..."      → customer not found
    """
    if customer_id.upper().startswith("FAIL"):
        return json.dumps(
            {
                "customer_id": customer_id,
                "customer_found": False,
                "message": "Customer not found in system.",
            }
        )

    account_status = "active"
    kyc_status = "verified"

    if customer_id.upper().startswith("SUSPENDED"):
        account_status = "suspended"
    elif customer_id.upper().startswith("FROZEN"):
        account_status = "frozen"
        kyc_status = "unverified"

    return json.dumps(
        {
            "customer_id": customer_id,
            "account_status": account_status,
            "kyc_status": kyc_status,
            "customer_found": True,
        }
    )


@server.tool()
def verify_identity(customer_id: str) -> str:
    """Verify the identity (KYC) of a customer.

    Returns kyc_status and identity_verified.
    Customer IDs starting with "FROZEN" fail verification.
    """
    if customer_id.upper().startswith("FROZEN"):
        return json.dumps(
            {
                "customer_id": customer_id,
                "kyc_status": "unverified",
                "identity_verified": False,
                "message": "Identity verification failed — account frozen.",
            }
        )

    return json.dumps(
        {
            "customer_id": customer_id,
            "kyc_status": "verified",
            "identity_verified": True,
            "message": "Identity verified successfully.",
        }
    )


@server.tool()
def lookup_payment(transaction_id: str, amount: float = 0.0) -> str:
    """Look up a payment/transaction by its ID.

    Returns payment_found, settlement_status, transaction_age_days,
    and original_transaction_amount.

    Scenarios via transaction_id prefix:
      - "OLD-..."       → transaction_age_days = 120 (outside 90-day window)
      - "UNSETTLED-..." → settlement_status = "pending"
      - "FAIL-..."      → payment not found
    """
    if transaction_id.upper().startswith("FAIL"):
        return json.dumps(
            {
                "transaction_id": transaction_id,
                "payment_found": False,
                "message": "Payment record not found.",
            }
        )

    settlement_status = "settled"
    transaction_age_days = 5

    if transaction_id.upper().startswith("OLD"):
        transaction_age_days = 120
    if transaction_id.upper().startswith("UNSETTLED"):
        settlement_status = "pending"
        transaction_age_days = 2

    return json.dumps(
        {
            "transaction_id": transaction_id,
            "payment_found": True,
            "settlement_status": settlement_status,
            "original_transaction_amount": amount if amount > 0 else 10000.0,
            "transaction_age_days": transaction_age_days,
        }
    )


@server.tool()
def initiate_refund(
    customer_id: str,
    transaction_id: str,
    amount: float,
    reason: str = "",
) -> str:
    """Initiate a refund for a customer transaction.

    Returns refund_id, refund_status, and approval info.
    Amounts > 5000 require manager approval.
    """
    approval_required = amount > 5000

    if approval_required:
        return json.dumps(
            {
                "refund_id": f"REF-{transaction_id[:8]}",
                "refund_status": "pending_approval",
                "refund_initiated": False,
                "approval_required": True,
                "message": f"Refund of {amount} requires manager approval (threshold: 5000).",
            }
        )

    return json.dumps(
        {
            "refund_id": f"REF-{transaction_id[:8]}",
            "refund_status": "success",
            "refund_initiated": True,
            "approval_required": False,
            "message": f"Refund of {amount} initiated successfully.",
        }
    )


@server.tool()
def create_ticket(
    customer_id: str,
    category: str = "general",
    severity: str = "low",
    description: str = "",
) -> str:
    """Create a support ticket for a customer issue.

    Returns ticket_id and ticket_status.
    Categories: billing, service, fraud, product, general.
    Severity: low, medium, high.
    """
    return json.dumps(
        {
            "ticket_id": f"TKT-{customer_id[:6]}-001",
            "ticket_status": "created",
            "category": category,
            "severity": severity,
            "message": f"Support ticket created for {category} issue ({severity} severity).",
        }
    )


@server.tool()
def handoff_to_human(
    customer_id: str,
    reason: str = "",
    ticket_id: str = "",
) -> str:
    """Escalate / hand off a case to a human operator.

    Returns handed_off status and assigned team.
    """
    return json.dumps(
        {
            "handed_off": True,
            "handoff_agent": "human_ops_team",
            "customer_id": customer_id,
            "ticket_id": ticket_id,
            "message": f"Case escalated to human operator. Reason: {reason or 'not specified'}",
        }
    )


# ── Complaint-specific tools ──────────────────────────────────────────


@server.tool()
def create_complaint_record(
    customer_id: str,
    category: str,
    severity: str = "medium",
    description: str = "",
) -> str:
    """Create a formal complaint record.

    Categories: billing, service, fraud, product, regulatory.
    Severity: low, medium, high.
    """
    return json.dumps(
        {
            "complaint_id": f"CMP-{customer_id[:6]}-001",
            "status": "open",
            "category": category,
            "severity": severity,
            "message": f"Complaint recorded for category={category}, severity={severity}.",
        }
    )


@server.tool()
def compute_compensation(
    complaint_id: str,
    category: str,
    severity: str,
    estimated_impact_value: float = 0.0,
) -> str:
    """Compute a compensation estimate for a complaint.

    Returns compensation_amount and whether approval is required (> 500).
    """
    base = {"low": 50, "medium": 200, "high": 500}
    amount = base.get(severity, 100)
    if estimated_impact_value > 0:
        amount = min(estimated_impact_value * 0.25, 2000)

    return json.dumps(
        {
            "complaint_id": complaint_id,
            "compensation_amount": amount,
            "approval_required": amount > 500,
            "message": f"Compensation estimate: {amount}. "
            + ("Manager approval required." if amount > 500 else "Auto-approved."),
        }
    )


@server.tool()
def apply_triage_rules(
    complaint_id: str,
    category: str,
    severity: str,
) -> str:
    """Apply triage rules to route a complaint to the right team.

    High severity or fraud/regulatory → escalated.
    """
    escalated = severity == "high" or category in ("fraud", "regulatory")
    owner = "escalation_team" if escalated else "frontline_team"

    return json.dumps(
        {
            "complaint_id": complaint_id,
            "escalated": escalated,
            "assigned_owner": owner,
            "message": f"Triaged to {owner}." + (" (escalated)" if escalated else ""),
        }
    )


if __name__ == "__main__":
    server.run(transport="stdio")
