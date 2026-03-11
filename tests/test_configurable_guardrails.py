# tests/test_configurable_guardrails.py
"""Tests for configurable guardrail rules: PolicyPack, PolicyGuardrails, and toggles."""
from __future__ import annotations

import json
import tempfile


from app.runtime.policy_pack import GuardrailRule, PolicyPack, _DEFAULT_RULES
from app.runtime.policy_guardrails import PolicyGuardrails


# ── GuardrailRule ────────────────────────────────────────────────────


class TestGuardrailRule:
    def test_from_dict_round_trip(self):
        d = {
            "id": "test_rule",
            "label": "Test Rule",
            "description": "A test rule",
            "category": "safety",
            "severity": "high",
            "enabled": True,
            "patterns": [r"hello\s+world"],
        }
        rule = GuardrailRule.from_dict(d)
        assert rule.id == "test_rule"
        assert rule.label == "Test Rule"
        assert rule.enabled is True
        assert rule.to_dict() == d

    def test_compiled_patterns_lazy(self):
        rule = GuardrailRule(id="test", label="test", patterns=[r"\bfoo\b", r"\bbar\b"])
        assert rule._compiled is None
        pats = rule.compiled_patterns
        assert len(pats) == 2
        assert pats[0].search("foo")
        assert not pats[0].search("baz")

    def test_invalidate_cache(self):
        rule = GuardrailRule(id="test", label="test", patterns=[r"\bfoo\b"])
        _ = rule.compiled_patterns  # populate cache
        assert rule._compiled is not None
        rule.invalidate_cache()
        assert rule._compiled is None

    def test_empty_patterns(self):
        rule = GuardrailRule(id="test", label="test", patterns=[])
        assert rule.compiled_patterns == []


# ── PolicyPack with rules ────────────────────────────────────────────


class TestPolicyPackRules:
    def test_default_rules_loaded_when_no_rules_in_json(self):
        """When policy_pack.json has no guardrail_rules, defaults are used."""
        data = {"name": "test", "version": "1"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            pack = PolicyPack.load(f.name)

        assert len(pack.guardrail_rules) == len(_DEFAULT_RULES)
        assert pack.guardrail_rules[0].id == "hallucination_action_claims"

    def test_custom_rules_loaded(self):
        data = {
            "name": "healthcare",
            "version": "1",
            "guardrail_rules": [
                {
                    "id": "hallucination_prescription",
                    "label": "Block hallucinated prescriptions",
                    "category": "safety",
                    "severity": "high",
                    "enabled": True,
                    "patterns": [r"prescription\s+(has been|was)\s+(issued|filled)"],
                }
            ],
            "transaction_slot_keys": ["patient_id", "appointment_id"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            pack = PolicyPack.load(f.name)

        assert len(pack.guardrail_rules) == 1
        assert pack.guardrail_rules[0].id == "hallucination_prescription"
        assert pack.transaction_slot_keys == ["patient_id", "appointment_id"]

    def test_get_rule(self):
        pack = PolicyPack(
            guardrail_rules=[
                GuardrailRule(id="r1", label="R1"),
                GuardrailRule(id="r2", label="R2"),
            ]
        )
        assert pack.get_rule("r1").label == "R1"
        assert pack.get_rule("r2").label == "R2"
        assert pack.get_rule("r3") is None

    def test_get_enabled_rules(self):
        pack = PolicyPack(
            guardrail_rules=[
                GuardrailRule(id="r1", label="R1", enabled=True, category="safety"),
                GuardrailRule(id="r2", label="R2", enabled=False, category="safety"),
                GuardrailRule(id="r3", label="R3", enabled=True, category="tone"),
            ]
        )
        enabled = pack.get_enabled_rules()
        assert len(enabled) == 2
        assert {r.id for r in enabled} == {"r1", "r3"}

        safety = pack.get_enabled_rules(category="safety")
        assert len(safety) == 1
        assert safety[0].id == "r1"

    def test_set_rule_enabled(self):
        pack = PolicyPack(
            guardrail_rules=[
                GuardrailRule(id="r1", label="R1", enabled=True),
            ]
        )
        assert pack.set_rule_enabled("r1", False) is True
        assert pack.guardrail_rules[0].enabled is False

        assert pack.set_rule_enabled("nonexistent", True) is False

    def test_save_and_load_round_trip(self):
        pack = PolicyPack(
            name="fintech",
            version="3",
            guardrail_rules=[
                GuardrailRule(
                    id="test_rule",
                    label="Test",
                    category="safety",
                    patterns=[r"\btest\b"],
                ),
            ],
            transaction_slot_keys=["payment_id"],
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            pack.save(f.name)
            loaded = PolicyPack.load(f.name)

        assert loaded.name == "fintech"
        assert len(loaded.guardrail_rules) == 1
        assert loaded.guardrail_rules[0].id == "test_rule"
        assert loaded.transaction_slot_keys == ["payment_id"]


# ── PolicyGuardrails with configurable rules ─────────────────────────


def _make_pack(**overrides) -> PolicyPack:
    defaults = {
        "name": "test",
        "guardrail_rules": [
            GuardrailRule(
                id="hallucination_action_claims",
                label="Block hallucinated actions",
                category="safety",
                enabled=True,
                patterns=[
                    r"(refund\s+(has been|was|is)\s+(initiated|processed|approved|completed)"
                    r"|successfully\s+refunded)",
                ],
            ),
            GuardrailRule(
                id="transaction_context",
                label="Transaction context",
                category="safety",
                enabled=True,
                patterns=[r"(order\s*#?\d|transaction\s*#?\d|\$\d)"],
            ),
            GuardrailRule(
                id="tone_strip_jargon",
                label="Strip jargon",
                category="internal",
                enabled=True,
                patterns=[r"\b(?:workflow|FSM|slots?)\b"],
            ),
        ],
        "transaction_slot_keys": ["payment_id", "transaction_id", "refund_id"],
    }
    defaults.update(overrides)
    return PolicyPack(**defaults)


class TestConfigurableHallucinationDetection:
    def test_blocks_hallucinated_refund_without_context(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "Your refund has been processed."},
            {"original_query": "Tell me about refund policies"},
        )
        assert result.allowed is False
        assert "hallucination" in result.reason

    def test_allows_refund_with_transaction_context(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "Your refund has been processed."},
            {"original_query": "Refund for order #1234"},
        )
        assert result.allowed is True

    def test_allows_refund_with_slot_context(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "Your refund has been processed."},
            {
                "original_query": "Yes, proceed",
                "_accumulated_slots": {"transaction_id": "TXN-123"},
            },
        )
        assert result.allowed is True

    def test_hallucination_disabled_allows_through(self):
        """When hallucination rule is disabled, false claims pass through."""
        pack = _make_pack()
        pack.get_rule("hallucination_action_claims").enabled = False
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "Your refund has been processed."},
            {"original_query": "Tell me about refund policies"},
        )
        assert result.allowed is True

    def test_informational_response_not_blocked(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        # knowledge_retrieved=True signals an informational response — skip hallucination check
        result = guard.post(
            {
                "text": "The refund has been approved per policy.",
                "knowledge_retrieved": True,
            },
            {"original_query": "What is your refund policy?"},
        )
        assert result.allowed is True

    def test_clarifying_response_not_blocked(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {
                "text": "Your refund has been initiated. Could you confirm?",
                "needs_input": True,
            },
            {"original_query": "I want a refund"},
        )
        assert result.allowed is True


class TestConfigurableToneControl:
    def test_strips_jargon_when_enabled(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "The workflow FSM processed your slots correctly."},
            {},
        )
        assert result.allowed is True
        text = (result.mutated_response or {}).get("text", "")
        assert "workflow" not in text.lower()
        assert "FSM" not in text

    def test_jargon_preserved_when_disabled(self):
        pack = _make_pack()
        pack.get_rule("tone_strip_jargon").enabled = False
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {"text": "The workflow FSM processed your slots correctly."},
            {},
        )
        assert result.allowed is True
        # No mutation since the only tone rule is disabled
        assert result.mutated_response is None

    def test_strips_jargon_from_chat_messages(self):
        pack = _make_pack()
        guard = PolicyGuardrails(pack)
        result = guard.post(
            {
                "text": "OK",
                "chat": {"messages": ["The workflow completed."]},
            },
            {},
        )
        assert result.allowed is True
        if result.mutated_response:
            msgs = result.mutated_response.get("chat", {}).get("messages", [])
            for m in msgs:
                assert "workflow" not in m.lower()


class TestCustomVerticalRules:
    def test_healthcare_hallucination_patterns(self):
        """Demonstrate custom healthcare rules work the same way."""
        pack = PolicyPack(
            name="healthcare",
            guardrail_rules=[
                GuardrailRule(
                    id="hallucination_action_claims",
                    label="Block hallucinated prescriptions",
                    category="safety",
                    enabled=True,
                    patterns=[r"prescription\s+(has been|was)\s+(issued|filled|approved)"],
                ),
                GuardrailRule(
                    id="transaction_context",
                    label="Patient context",
                    category="safety",
                    enabled=True,
                    patterns=[r"patient\s*(id|#|MRN)\s*:?\s*\d"],
                ),
            ],
            transaction_slot_keys=["patient_id", "appointment_id"],
        )
        guard = PolicyGuardrails(pack)

        # Blocks hallucinated prescription without patient context
        result = guard.post(
            {"text": "Your prescription has been issued."},
            {"original_query": "What medications do you offer?"},
        )
        assert result.allowed is False

        # Allows with patient context
        result2 = guard.post(
            {"text": "Your prescription has been issued."},
            {"original_query": "Prescription for patient #12345"},
        )
        assert result2.allowed is True

        # Allows with slot context
        result3 = guard.post(
            {"text": "Your prescription has been issued."},
            {
                "original_query": "Yes please",
                "_accumulated_slots": {"patient_id": "P-123"},
            },
        )
        assert result3.allowed is True

    def test_no_rules_means_no_checks(self):
        """A pack with empty rules performs no hallucination or tone checks."""
        pack = PolicyPack(name="empty", guardrail_rules=[])
        guard = PolicyGuardrails(pack)

        result = guard.post(
            {"text": "Your refund has been processed. The workflow FSM is done."},
            {"original_query": "Hello"},
        )
        assert result.allowed is True
        assert result.mutated_response is None
