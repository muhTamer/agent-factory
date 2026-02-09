# app/runtime/policy_guardrails.py
from __future__ import annotations

from typing import Any, Dict

from app.runtime.guardrails import GuardResult, Guardrails
from app.runtime.policy_pack import PolicyPack


class PolicyGuardrails(Guardrails):
    def __init__(self, pack: PolicyPack):
        self.pack = pack

    def pre(self, query: str, context: Dict[str, Any]) -> GuardResult:
        q = query or ""

        if len(q) > self.pack.max_query_chars:
            return GuardResult(allowed=False, reason=f"query_too_long>{self.pack.max_query_chars}")

        lowered = q.lower()
        for phrase in self.pack.blocked_phrases:
            if phrase.lower() in lowered:
                return GuardResult(allowed=False, reason=f"blocked_phrase:{phrase}")

        # (later) pii_redaction / tool restrictions / jailbreak detection
        return GuardResult(allowed=True)

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        # (later) response compliance checks / redaction
        return GuardResult(allowed=True)
