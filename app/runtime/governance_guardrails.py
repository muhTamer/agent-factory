# app/runtime/governance_guardrails.py
"""
Governance-Aware Guardrails for RQ3 Trade-off Evaluation

Wraps PolicyGuardrails with governance-level-aware feature toggles.
Each pre/post check can be independently enabled or disabled based on
the GovernanceConfig, and every decision is logged as a governance event
for RQ3 metrics collection.
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.runtime.governance_config import GovernanceConfig
from app.runtime.guardrails import GuardResult, Guardrails
from app.runtime.policy_guardrails import PolicyGuardrails
from app.runtime.policy_pack import PolicyPack


class GovernanceAwareGuardrails(Guardrails):
    """
    Wraps PolicyGuardrails with governance-level-aware feature toggles.

    Delegates to PolicyGuardrails for actual check logic, but skips
    checks that are disabled at the current governance level.
    Records governance events for RQ3 metrics collection.
    """

    def __init__(self, pack: PolicyPack, config: GovernanceConfig):
        # Exposed for spine intent inference (spine.py line 743:
        # getattr(self.guardrails, "pack", None))
        self.pack = pack
        self._inner = PolicyGuardrails(pack)
        self.config = config
        self._events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Pre-guardrails
    # ------------------------------------------------------------------

    def pre(self, query: str, context: Dict[str, Any]) -> GuardResult:
        if not self.config.pre_checks_enabled:
            self._log("governance_pre_skip", check="all", reason="pre_checks_disabled")
            return GuardResult(allowed=True)

        # Query length check (always runs when pre_checks_enabled)
        if len(query) > self.config.max_query_chars:
            self._log(
                "governance_pre_block",
                check="query_length",
                query_len=len(query),
                limit=self.config.max_query_chars,
            )
            return GuardResult(
                allowed=False,
                reason=f"query_too_long>{self.config.max_query_chars}",
            )

        # Intent blocking check
        if not self.config.intent_blocking_enabled:
            self._log("governance_pre_skip", check="intent_blocking")
            return GuardResult(allowed=True)

        # Delegate full pre-check to PolicyGuardrails
        result = self._inner.pre(query, context)
        action = "blocked" if not result.allowed else "allowed"
        self._log(
            "governance_pre_check", check="full", action=action, reason=result.reason
        )
        return result

    # ------------------------------------------------------------------
    # Post-guardrails
    # ------------------------------------------------------------------

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        text = response.get("text", "") or response.get("answer", "") or ""
        text_lower = text.lower()

        # 1. Blocked phrase enforcement
        if self.config.blocked_phrase_enforcement:
            # Check base phrases (shared by MEDIUM + HIGH)
            all_phrases = list(self.pack.blocked_phrases)
            # HIGH adds extra compliance phrases via additional_blocked_phrases
            all_phrases.extend(self.config.additional_blocked_phrases)
            for phrase in all_phrases:
                if phrase.lower() in text_lower:
                    self._log(
                        "governance_post_block",
                        check="blocked_phrase",
                        phrase=phrase,
                    )
                    return GuardResult(allowed=False, reason=f"blocked_phrase:{phrase}")
        else:
            self._log("governance_post_skip", check="blocked_phrase")

        # 2. Hallucination detection
        if self.config.hallucination_detection:
            if self.config.hallucination_strict:
                # STRICT mode (HIGH): check patterns directly, ignoring
                # the informational/clarifying bypasses that let agents
                # claim actions without transaction evidence.
                hallucination_blocked = self._check_hallucination_strict(
                    text, response, context
                )
                if hallucination_blocked:
                    self._log(
                        "governance_post_block",
                        check="hallucination_strict",
                        reason=hallucination_blocked,
                    )
                    return GuardResult(allowed=False, reason=hallucination_blocked)
            else:
                # STANDARD mode (MEDIUM): delegate to PolicyGuardrails
                # which honours informational/clarifying bypasses.
                inner_result = self._inner.post(response, context)
                if not inner_result.allowed and "hallucination" in (
                    inner_result.reason or ""
                ):
                    self._log(
                        "governance_post_block",
                        check="hallucination",
                        reason=inner_result.reason,
                    )
                    return inner_result
        else:
            self._log("governance_post_skip", check="hallucination")

        # 3. Tone control
        if self.config.tone_control_enabled:
            mutated = self._inner._apply_tone_control(response)
            if mutated is not response:
                if self.config.tone_violation_action == "block":
                    # HIGH: block responses containing jargon entirely
                    self._log(
                        "governance_post_block",
                        check="tone_violation_block",
                    )
                    return GuardResult(
                        allowed=False,
                        reason="tone_violation:internal_jargon_detected",
                    )
                else:
                    # MEDIUM: silently strip jargon (mutate)
                    self._log("governance_post_mutate", check="tone_control")
                    return GuardResult(allowed=True, mutated_response=mutated)
        else:
            self._log("governance_post_skip", check="tone_control")

        self._log("governance_post_allow")
        return GuardResult(allowed=True)

    # ------------------------------------------------------------------
    # Strict hallucination detection (HIGH governance)
    # ------------------------------------------------------------------

    def _check_hallucination_strict(
        self, text: str, response: Dict[str, Any], context: Dict[str, Any]
    ) -> str | None:
        """Check for hallucinated action claims WITHOUT the informational bypass.

        In strict mode we only skip if the query genuinely contains
        transaction evidence (order #, transaction ID, EUR amount, etc.).
        We do NOT skip just because the agent happened to retrieve knowledge.

        Returns the block reason string if blocked, else None.
        """
        rule = self._inner._hallucination_rule
        if not rule or not rule.enabled or not rule.compiled_patterns:
            return None

        original_query = context.get("original_query", "")
        if not original_query:
            return None

        # Only bypass if real transaction evidence is present in the query
        has_tx = self._inner._detect_transaction_context(original_query, context)
        if has_tx:
            return None

        # Check response text against hallucination patterns
        for pat in rule.compiled_patterns:
            if pat.search(text):
                return "hallucination_action_claim_without_context"

        return None

    # ------------------------------------------------------------------
    # Event logging for RQ3 metrics
    # ------------------------------------------------------------------

    def _log(self, event_type: str, **data: Any) -> None:
        action = "skipped"
        if "block" in event_type:
            action = "blocked"
        elif "mutate" in event_type:
            action = "mutated"
        elif "allow" in event_type:
            action = "allowed"
        elif "skip" in event_type:
            action = "skipped"

        self._events.append(
            {
                "type": event_type,
                "action": action,
                "level": self.config.level.value,
                **data,
            }
        )

    def get_events(self) -> List[Dict[str, Any]]:
        """Return all governance events and clear the log."""
        events = list(self._events)
        self._events.clear()
        return events

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return a copy of all governance events without clearing."""
        return list(self._events)
