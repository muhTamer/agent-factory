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
        self._log("governance_pre_check", check="full", action=action, reason=result.reason)
        return result

    # ------------------------------------------------------------------
    # Post-guardrails
    # ------------------------------------------------------------------

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        text = response.get("text", "") or response.get("answer", "") or ""

        # 1. Blocked phrase enforcement
        if self.config.blocked_phrase_enforcement:
            for phrase in self.pack.blocked_phrases:
                if phrase.lower() in text.lower():
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
            # Delegate to inner post and check if it blocks for hallucination
            inner_result = self._inner.post(response, context)
            if not inner_result.allowed and "hallucination" in (inner_result.reason or ""):
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
                self._log("governance_post_mutate", check="tone_control")
                return GuardResult(allowed=True, mutated_response=mutated)
        else:
            self._log("governance_post_skip", check="tone_control")

        self._log("governance_post_allow")
        return GuardResult(allowed=True)

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
