from __future__ import annotations

import re
from typing import Any, Dict, List

from app.runtime.guardrails import GuardResult, Guardrails
from app.runtime.policy_pack import GuardrailRule, PolicyPack


class PolicyGuardrails(Guardrails):
    def __init__(self, pack: PolicyPack):
        self.pack = pack
        self._pii_redactor = None
        if pack.pii_redaction:
            from app.governance.pii_redactor import PIIRedactor

            self._pii_redactor = PIIRedactor()

        # Pre-resolve rule references for fast lookup
        self._hallucination_rule = pack.get_rule("hallucination_action_claims")
        self._transaction_ctx_rule = pack.get_rule("transaction_context")
        self._tx_slot_keys = set(pack.transaction_slot_keys)

        # Collect tone-control rules (categories: tone, internal)
        self._tone_rules: List[GuardrailRule] = [
            r for r in pack.guardrail_rules if r.category in ("tone", "internal")
        ]

    # ------------------------------------------------------------------
    # Pre-guardrails
    # ------------------------------------------------------------------

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

        # PII redaction on incoming query
        if self._pii_redactor:
            redacted_q, records = self._pii_redactor.redact(query)
            if records:
                return GuardResult(
                    allowed=True,
                    mutated_query=redacted_q,
                    mutated_context={
                        **context,
                        "_pii_redactions_pre": [r.to_dict() for r in records],
                    },
                )

        return GuardResult(allowed=True)

    # ------------------------------------------------------------------
    # Post-guardrails
    # ------------------------------------------------------------------

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        # Check for blocked phrases in response text
        text = response.get("text", "") or response.get("answer", "") or ""
        for phrase in self.pack.blocked_phrases:
            if phrase.lower() in text.lower():
                return GuardResult(
                    allowed=False,
                    reason=f"blocked_phrase:{phrase}",
                )

        # ── Hallucination detection (config-driven) ──────────────────
        #
        # If the hallucination rule is enabled, check whether the response
        # falsely claims an action was performed without transaction evidence.
        if (
            self._hallucination_rule
            and self._hallucination_rule.enabled
            and self._hallucination_rule.compiled_patterns
        ):
            original_query = context.get("original_query", "")
            has_transaction_context = self._detect_transaction_context(
                original_query, context
            )

            # Domain agents that only retrieved knowledge are informational —
            # they're describing policy, not claiming to have executed an action.
            is_informational = response.get(
                "knowledge_retrieved"
            ) is True and not response.get("tools_used")
            is_clarifying = (
                response.get("needs_input") is True
                or response.get("domain_agent_clarification") is True
            )

            if (
                original_query
                and not has_transaction_context
                and not is_informational
                and not is_clarifying
            ):
                for pat in self._hallucination_rule.compiled_patterns:
                    if pat.search(text):
                        return GuardResult(
                            allowed=False,
                            reason="hallucination_action_claim_without_context",
                        )

        # PII redaction on outgoing response
        if self._pii_redactor:
            redacted_resp, pii_records = self._pii_redactor.redact_dict(response)
            if pii_records:
                response = redacted_resp

        # Tone control: strip customer-inappropriate patterns from all
        # text fields (text, answer, chat.messages).
        mutated = self._apply_tone_control(response)
        if mutated is not response:
            return GuardResult(allowed=True, mutated_response=mutated)

        return GuardResult(allowed=True)

    # ------------------------------------------------------------------
    # Transaction context detection (config-driven)
    # ------------------------------------------------------------------

    def _detect_transaction_context(
        self, original_query: str, context: Dict[str, Any]
    ) -> bool:
        """Check if query or accumulated slots contain transaction evidence."""
        # Check query against transaction context patterns
        if (
            original_query
            and self._transaction_ctx_rule
            and self._transaction_ctx_rule.enabled
        ):
            for pat in self._transaction_ctx_rule.compiled_patterns:
                if pat.search(original_query):
                    return True

        # Multi-turn: check accumulated slots for transaction identifiers
        acc_slots = context.get("_accumulated_slots") or {}
        for key in self._tx_slot_keys:
            if acc_slots.get(key):
                return True

        # A pinned workflow agent with populated slots is always legitimate
        if context.get("pinned_agent_type") == "workflow_runner" and acc_slots:
            return True

        return False

    # ------------------------------------------------------------------
    # Tone control helpers (config-driven)
    # ------------------------------------------------------------------

    def _sanitize_text(self, text: str) -> str:
        """Remove tone-violating patterns from a single string."""
        cleaned = text
        for rule in self._tone_rules:
            if not rule.enabled:
                continue
            for pat in rule.compiled_patterns:
                cleaned = pat.sub(" ", cleaned)
        # Collapse whitespace introduced by removals
        cleaned = re.sub(r"  +", " ", cleaned).strip()
        return cleaned

    def _apply_tone_control(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new response dict if any text was sanitized, else return the same object."""
        # Skip entirely if no tone rules are enabled
        if not any(r.enabled for r in self._tone_rules):
            return response

        changed = False

        # Sanitize top-level text / answer
        for key in ("text", "answer"):
            val = response.get(key)
            if isinstance(val, str) and val:
                cleaned = self._sanitize_text(val)
                if cleaned != val:
                    if not changed:
                        response = dict(response)  # shallow copy on first mutation
                        changed = True
                    response[key] = cleaned

        # Sanitize voice chat messages
        chat = response.get("chat")
        if isinstance(chat, dict):
            msgs = chat.get("messages")
            if isinstance(msgs, list):
                new_msgs = []
                msgs_changed = False
                for m in msgs:
                    if isinstance(m, str):
                        c = self._sanitize_text(m)
                        if c != m:
                            msgs_changed = True
                        new_msgs.append(c)
                    else:
                        new_msgs.append(m)
                if msgs_changed:
                    if not changed:
                        response = dict(response)
                        changed = True
                    response["chat"] = {**chat, "messages": new_msgs}

        return response
