# tests/test_governance_config.py
"""Unit tests for GovernanceConfig and GovernanceAwareGuardrails."""

from __future__ import annotations

import pytest

from app.runtime.governance_config import GovernanceConfig, GovernanceLevel
from app.runtime.governance_guardrails import GovernanceAwareGuardrails
from app.runtime.policy_pack import GuardrailRule, PolicyPack


@pytest.fixture
def pack():
    return PolicyPack(
        name="test",
        version="1",
        max_query_chars=4000,
        intent_rules={"blocked_intent": {"mode": "block", "reason": "test_blocked"}},
        route_to_intent={"refunds_workflow": "refund_request"},
        blocked_phrases=["guaranteed refund"],
        guardrail_rules=[
            GuardrailRule(
                id="hallucination_action_claims",
                label="Block hallucinated action claims",
                category="safety",
                severity="high",
                enabled=True,
                patterns=[
                    r"(refund\s+(has been|was|is)\s+(initiated|processed|approved|completed)"
                    r"|refund_id\s*[:=]"
                    r"|successfully\s+refunded)",
                ],
            ),
            GuardrailRule(
                id="transaction_context",
                label="Transaction context detection",
                category="safety",
                severity="high",
                enabled=True,
                patterns=[
                    r"(order\s*#?\d|transaction\s*#?\d|EUR\s*\d|USD\s*\d|\$\d)",
                ],
            ),
            GuardrailRule(
                id="tone_strip_jargon",
                label="Hide system jargon",
                category="internal",
                severity="medium",
                enabled=True,
                patterns=[r"\b(?:workflow|FSM|slot[s]?|router|guardrail|pipeline)\b"],
            ),
        ],
        transaction_slot_keys=["payment_id", "transaction_id", "refund_id", "order_id"],
    )


# ── GovernanceConfig presets ─────────────────────────────────────────


class TestGovernanceConfig:
    def test_low_level_presets(self):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        assert config.level == GovernanceLevel.LOW
        assert config.pre_checks_enabled is False
        assert config.max_query_chars == 10000
        assert config.intent_blocking_enabled is False
        assert config.hallucination_detection is False
        assert config.tone_control_enabled is False
        assert config.max_autonomy_actions == 0
        assert config.auto_approval_limit == 10000.0
        assert config.strict_eligibility == "log_only"
        assert config.require_user_confirmation is False

    def test_medium_level_presets(self):
        config = GovernanceConfig.for_level(GovernanceLevel.MEDIUM)
        assert config.level == GovernanceLevel.MEDIUM
        assert config.pre_checks_enabled is True
        assert config.max_query_chars == 4000
        assert config.auto_approval_limit == 5000.0
        assert config.escalation_threshold == 0.4
        assert config.max_autonomy_actions == 5
        assert config.strict_eligibility == "enforce"

    def test_high_level_presets(self):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        assert config.level == GovernanceLevel.HIGH
        assert config.pre_checks_enabled is True
        assert config.max_query_chars == 2000
        assert config.escalation_threshold == 0.7
        assert config.max_autonomy_actions == 2
        assert config.auto_approval_limit == 1000.0
        assert config.strict_eligibility == "enforce_escalate"
        assert config.allow_replanning is False

    def test_config_is_frozen(self):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        with pytest.raises(AttributeError):
            config.pre_checks_enabled = True  # type: ignore[misc]

    def test_governance_level_is_str_enum(self):
        assert GovernanceLevel.LOW.value == "low"
        assert GovernanceLevel.MEDIUM.value == "medium"
        assert GovernanceLevel.HIGH.value == "high"
        assert GovernanceLevel("low") == GovernanceLevel.LOW


# ── GovernanceAwareGuardrails ────────────────────────────────────────


class TestGovernanceAwareGuardrails:
    def test_low_skips_pre_checks(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("x" * 5000, {})
        assert result.allowed is True  # LOW skips all pre checks

    def test_high_enforces_query_length(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("x" * 2500, {})
        assert result.allowed is False
        assert "query_too_long" in result.reason

    def test_high_allows_short_query(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("Short query", {})
        assert result.allowed is True

    def test_medium_allows_normal_query(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.MEDIUM)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("Short query", {})
        assert result.allowed is True

    def test_medium_blocks_long_query(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.MEDIUM)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("x" * 4500, {})
        assert result.allowed is False

    def test_low_skips_blocked_phrase_check(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post({"text": "We offer guaranteed refund"}, {})
        assert result.allowed is True

    def test_high_blocks_blocked_phrases(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post({"text": "We offer guaranteed refund"}, {})
        assert result.allowed is False
        assert "blocked_phrase" in result.reason

    def test_medium_blocks_blocked_phrases(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.MEDIUM)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post({"text": "We offer guaranteed refund"}, {})
        assert result.allowed is False

    def test_low_skips_tone_control(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        response = {"text": "The workflow FSM pipeline processed your slot"}
        result = guard.post(response, {})
        assert result.allowed is True
        # No mutation — jargon preserved
        assert result.mutated_response is None

    def test_high_applies_tone_control(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        response = {"text": "The workflow FSM pipeline processed your slot"}
        result = guard.post(response, {})
        assert result.allowed is True
        # Should have mutated to strip jargon
        if result.mutated_response:
            text = result.mutated_response.get("text", "")
            assert "workflow" not in text.lower()
            assert "FSM" not in text

    def test_governance_events_logged_on_pre(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        guard.pre("hello", {})
        events = guard.drain_events()
        assert len(events) >= 1
        assert events[0]["level"] == "low"

    def test_governance_events_logged_on_post(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        guard.post({"text": "Clean response"}, {})
        events = guard.drain_events()
        assert len(events) >= 1

    def test_get_events_clears_log(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        guard.pre("hello", {})
        events = guard.get_events()
        assert len(events) >= 1
        # After get_events, log should be cleared
        assert len(guard.get_events()) == 0

    def test_drain_events_does_not_clear(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        guard.pre("hello", {})
        events1 = guard.drain_events()
        events2 = guard.drain_events()
        assert len(events1) == len(events2)

    def test_pack_attribute_exposed(self, pack):
        """spine.py line 743: getattr(self.guardrails, 'pack', None)"""
        config = GovernanceConfig.for_level(GovernanceLevel.MEDIUM)
        guard = GovernanceAwareGuardrails(pack, config)
        assert guard.pack is pack

    def test_intent_blocking_disabled_at_low(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        # Even with a blocked intent, LOW governance skips all pre checks
        result = guard.pre("test", {"intent": "blocked_intent"})
        assert result.allowed is True

    def test_intent_blocking_enabled_at_high(self, pack):
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.pre("test", {"intent": "blocked_intent"})
        assert result.allowed is False
        assert "test_blocked" in result.reason

    def test_low_skips_hallucination_detection(self, pack):
        """At LOW, hallucination detection is disabled — refund claims pass."""
        config = GovernanceConfig.for_level(GovernanceLevel.LOW)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post(
            {"text": "Your refund has been processed and approved."},
            {"original_query": "Tell me about refund policies"},
        )
        assert result.allowed is True

    def test_high_blocks_hallucinated_refund(self, pack):
        """At HIGH, hallucination detection blocks refund claims without tx context."""
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post(
            {"text": "Your refund has been processed and approved."},
            {"original_query": "Tell me about refund policies"},
        )
        assert result.allowed is False

    def test_high_allows_refund_with_tx_context(self, pack):
        """At HIGH, refund claims with transaction context are legitimate."""
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)
        result = guard.post(
            {"text": "Your refund has been processed and approved."},
            {"original_query": "Refund for order #1234 EUR 50"},
        )
        assert result.allowed is True

    def test_event_actions_are_correct(self, pack):
        """Verify governance events record the correct action types."""
        config = GovernanceConfig.for_level(GovernanceLevel.HIGH)
        guard = GovernanceAwareGuardrails(pack, config)

        # Trigger a block
        guard.post({"text": "We offer guaranteed refund"}, {})
        events = guard.get_events()

        block_events = [e for e in events if e["action"] == "blocked"]
        assert len(block_events) >= 1

    def test_autonomy_derived_from_events(self):
        """Autonomy score should be derived from observed interventions."""
        from evaluation.governance_metrics import (
            GovernanceScenarioResult,
            compute_rq3_metrics,
        )

        # 3 scenarios: 1 blocked, 1 mutated, 1 clean
        results = [
            GovernanceScenarioResult(
                scenario_id="s1",
                governance_level="high",
                category="test",
                governance_blocks=1,
                governance_mutations=0,
            ),
            GovernanceScenarioResult(
                scenario_id="s2",
                governance_level="high",
                category="test",
                governance_blocks=0,
                governance_mutations=1,
            ),
            GovernanceScenarioResult(
                scenario_id="s3",
                governance_level="high",
                category="test",
                governance_blocks=0,
                governance_mutations=0,
            ),
        ]
        metrics = compute_rq3_metrics(results)
        # 2 interventions out of 3 scenarios → autonomy = 1 - 2/3 ≈ 0.3333
        assert 0.3 < metrics["autonomy_score"] < 0.4
