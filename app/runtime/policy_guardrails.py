from __future__ import annotations

from typing import Any, Dict

from app.runtime.guardrails import GuardResult, Guardrails
from app.runtime.policy_pack import PolicyPack


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
        return GuardResult(allowed=True)
