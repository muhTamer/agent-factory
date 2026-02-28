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

# Tone guardrail: patterns that should never appear in customer-facing text.
# Each tuple is (compiled_regex, empty_string_replacement_or_None_to_strip).
_TONE_STRIP_PATTERNS = [
    # Never ask the customer "is this urgent?" — urgency is an internal triage flag
    re.compile(
        r"[\s,]*\b(?:is\s+this|would\s+you\s+(?:say|consider)\s+(?:this|it))\s+urgent\??\s*",
        re.IGNORECASE,
    ),
    # Never promise async follow-up the system can't deliver
    re.compile(
        r"I(?:'ve| have)\s+(?:forwarded|escalated|sent)\s+(?:this|your|the)\b[^.!?]*[.!?]?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"I\s+will\s+(?:get back to you|follow up|notify you|let you know)\b[^.!?]*[.!?]?\s*",
        re.IGNORECASE,
    ),
    # Strip internal jargon that leaks through
    re.compile(r"\b(?:workflow|FSM|slot[s]?|router|guardrail|pipeline)\b", re.IGNORECASE),
    # Strip internal file references — customers should never see source filenames
    # like "BankFAQs.csv" or "refunds_policy.yaml".  Matches "filename.ext" for
    # common data/config extensions, including surrounding context like
    # "(source: BankFAQs.csv)" or "from refunds_policy.yaml".
    re.compile(
        r"(?:\(?(?:source|from|in|per|see|ref)\s*:\s*)?"
        r"\b\w+\.(?:csv|yaml|yml|json|txt|md|tsv|xlsx?|pdf)\b"
        r"\)?",
        re.IGNORECASE,
    ),
]


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
        #
        # IMPORTANT: In multi-turn workflows the current message may be
        # "Yes, proceed" while all transaction details live in accumulated
        # slots from earlier turns.  We must check BOTH the current query
        # AND the accumulated context before blocking.
        original_query = context.get("original_query", "")
        has_transaction_context = (
            bool(
                re.search(
                    r"(order\s*#?\d|transaction\s*#?\d|EUR\s*\d|USD\s*\d|\$\d)",
                    original_query,
                    re.IGNORECASE,
                )
            )
            if original_query
            else False
        )

        # Multi-turn: if a workflow agent is pinned with accumulated slots
        # containing transaction identifiers, the refund is legitimate.
        if not has_transaction_context:
            acc_slots = context.get("_accumulated_slots") or {}
            if (
                acc_slots.get("payment_id")
                or acc_slots.get("transaction_id")
                or acc_slots.get("refund_id")
            ):
                has_transaction_context = True
            # A pinned workflow_runner with populated slots is always legitimate
            if context.get("pinned_agent_type") == "workflow_runner" and acc_slots:
                has_transaction_context = True

        if original_query and not has_transaction_context:
            if _REFUND_INITIATED_PATTERN.search(text):
                return GuardResult(
                    allowed=False,
                    reason="refund_initiated_without_transaction_details",
                )

        # Tone control: strip customer-inappropriate patterns from all
        # text fields (text, answer, chat.messages).
        mutated = self._apply_tone_control(response)
        if mutated is not response:
            return GuardResult(allowed=True, mutated_response=mutated)

        return GuardResult(allowed=True)

    # ------------------------------------------------------------------
    # Tone control helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove tone-violating patterns from a single string."""
        cleaned = text
        for pat in _TONE_STRIP_PATTERNS:
            cleaned = pat.sub(" ", cleaned)
        # Collapse whitespace introduced by removals
        cleaned = re.sub(r"  +", " ", cleaned).strip()
        return cleaned

    def _apply_tone_control(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new response dict if any text was sanitized, else return the same object."""
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
