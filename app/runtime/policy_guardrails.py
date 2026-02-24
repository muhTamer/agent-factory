from __future__ import annotations

import re
from typing import Any, Dict

from app.runtime.guardrails import GuardResult, Guardrails
from app.runtime.policy_pack import PolicyPack

# Patterns indicating a refund/action was executed (not just discussed)
_REFUND_INITIATED_PATTERN = re.compile(
    r"(refund\s+(has been|was|is)\s+(initiated|processed|approved|completed)"
    r"|refund_id\s*[:=]"
    r"|successfully\s+refunded"
    r"|refund\s+of\s+(EUR|USD|\$)\s*\d+.*(?:initiated|processed))",
    re.IGNORECASE,
)


class PolicyGuardrails(Guardrails):
    def __init__(self, pack: PolicyPack):
        self.pack = pack

    def pre(self, query: str, context: Dict[str, Any]) -> GuardResult:
        # length check
        if len(query) > self.pack.max_query_chars:
            return GuardResult(
                allowed=False,
                reason=f"query_too_long>{self.pack.max_query_chars}",
            )

        # intent-aware blocking
        intent = context.get("intent")
        if intent and intent in self.pack.intent_rules:
            rule = self.pack.intent_rules[intent]
            if rule.get("mode") == "block":
                return GuardResult(
                    allowed=False,
                    reason=rule.get("reason", f"intent_blocked:{intent}"),
                )

        return GuardResult(allowed=True)

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        # Check for blocked phrases in response text
        text = response.get("text", "") or response.get("answer", "") or ""
        for phrase in self.pack.blocked_phrases:
            if phrase.lower() in text.lower():
                return GuardResult(
                    allowed=False,
                    reason=f"blocked_phrase:{phrase}",
                )

        # Check for unauthorized refund initiation (hallucinated actions)
        # If the orchestration was informational (no concrete transaction in query),
        # block responses that claim a refund was actually processed.
        original_query = context.get("original_query", "")
        if original_query and not re.search(
            r"(order\s*#?\d|transaction\s*#?\d|EUR\s*\d|USD\s*\d|\$\d)",
            original_query,
            re.IGNORECASE,
        ):
            if _REFUND_INITIATED_PATTERN.search(text):
                return GuardResult(
                    allowed=False,
                    reason="refund_initiated_without_transaction_details",
                )

        return GuardResult(allowed=True)
