from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GuardrailRule:
    """A single configurable guardrail rule with patterns and a toggle."""

    id: str
    label: str
    description: str = ""
    category: str = "general"  # safety | tone | internal | privacy
    enabled: bool = True
    severity: str = "medium"  # low | medium | high
    patterns: List[str] = field(default_factory=list)

    # Compiled patterns (populated lazily)
    _compiled: Optional[List[re.Pattern]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def compiled_patterns(self) -> List[re.Pattern]:
        if self._compiled is None:
            self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        return self._compiled

    def invalidate_cache(self) -> None:
        self._compiled = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "enabled": self.enabled,
            "severity": self.severity,
            "patterns": self.patterns,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GuardrailRule":
        return GuardrailRule(
            id=str(d.get("id", "")),
            label=str(d.get("label", d.get("id", ""))),
            description=str(d.get("description", "")),
            category=str(d.get("category", "general")),
            enabled=bool(d.get("enabled", True)),
            severity=str(d.get("severity", "medium")),
            patterns=list(d.get("patterns", [])),
        )


# ── Default rules (fintech) ─────────────────────────────────────────
# These are used when no rules are provided in the policy pack JSON.
# They match the previously-hardcoded patterns in policy_guardrails.py.

_DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "id": "hallucination_action_claims",
        "label": "Block hallucinated action claims",
        "description": "Blocks responses that claim an action was performed (e.g. refund processed) without evidence of a real workflow execution.",
        "category": "safety",
        "severity": "high",
        "enabled": True,
        "patterns": [
            r"(refund\s+(has been|was|is)\s+(initiated|processed|approved|completed)"
            r"|refund_id\s*[:=]"
            r"|successfully\s+refunded"
            r"|refund\s+of\s+(EUR|USD|\$)\s*\d+.*(?:initiated|processed))",
        ],
    },
    {
        "id": "transaction_context",
        "label": "Transaction context detection",
        "description": "Pattern to detect when the user query contains a real transaction/case/record identifier. Used by hallucination detection to avoid false positives.",
        "category": "safety",
        "severity": "high",
        "enabled": True,
        "patterns": [
            r"(order\s*(id|#|no\.?|number)?\s*:?\s*\d"
            r"|transaction\s*(id|#|no\.?|number)?\s*:?\s*\d"
            r"|ORD-\d"
            r"|EUR\s*\d|USD\s*\d|\$\d)",
        ],
    },
    {
        "id": "tone_urgency",
        "label": "Strip urgency questions",
        "description": "Remove questions like 'Is this urgent?' — urgency is an internal triage signal, not a customer question.",
        "category": "tone",
        "severity": "low",
        "enabled": True,
        "patterns": [
            r"[\s,]*\b(?:is\s+this|would\s+you\s+(?:say|consider)\s+(?:this|it))\s+urgent\??\s*",
        ],
    },
    {
        "id": "tone_async_promises",
        "label": "Strip async follow-up promises",
        "description": "Remove promises like 'I have forwarded your case' or 'I will get back to you' that the system cannot fulfil.",
        "category": "tone",
        "severity": "medium",
        "enabled": True,
        "patterns": [
            r"I(?:'ve| have)\s+(?:forwarded|escalated|sent)\s+(?:this|your|the)\b[^.!?]*[.!?]?\s*",
            r"I\s+will\s+(?:get back to you|follow up|notify you|let you know)\b[^.!?]*[.!?]?\s*",
        ],
    },
    {
        "id": "tone_strip_jargon",
        "label": "Hide system jargon",
        "description": "Remove internal system terms like 'workflow', 'FSM', 'pipeline', 'slots' from customer-facing text.",
        "category": "internal",
        "severity": "medium",
        "enabled": True,
        "patterns": [
            r"\b(?:workflow|FSM|slot[s]?|router|guardrail|pipeline)\b",
        ],
    },
    {
        "id": "tone_strip_file_refs",
        "label": "Hide internal file names",
        "description": "Remove references to internal files like 'BankFAQs.csv' or 'refunds_policy.yaml' from customer-facing text.",
        "category": "internal",
        "severity": "medium",
        "enabled": True,
        "patterns": [
            r"(?:\(?(?:source|from|in|per|see|ref)\s*:\s*)?"
            r"\b\w+\.(?:csv|yaml|yml|json|txt|md|tsv|xlsx?|pdf)\b"
            r"\)?",
        ],
    },
]


@dataclass
class PolicyPack:
    name: str = "default"
    version: str = "0"
    max_query_chars: int = 4000

    intent_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    route_to_intent: Dict[str, str] = field(default_factory=dict)

    blocked_phrases: list[str] = field(default_factory=list)
    pii_redaction: bool = False

    # Configurable guardrail rules (replaces hardcoded patterns)
    guardrail_rules: List[GuardrailRule] = field(default_factory=list)

    # Slot keys that indicate a legitimate multi-turn workflow is in progress
    # (used by hallucination detection to avoid false positives)
    transaction_slot_keys: List[str] = field(default_factory=list)

    def get_rule(self, rule_id: str) -> Optional[GuardrailRule]:
        for r in self.guardrail_rules:
            if r.id == rule_id:
                return r
        return None

    def get_enabled_rules(self, category: Optional[str] = None) -> List[GuardrailRule]:
        rules = [r for r in self.guardrail_rules if r.enabled]
        if category:
            rules = [r for r in rules if r.category == category]
        return rules

    def set_rule_enabled(self, rule_id: str, enabled: bool) -> bool:
        rule = self.get_rule(rule_id)
        if rule is None:
            return False
        rule.enabled = enabled
        return True

    def rules_summary(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.guardrail_rules]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        data = {
            "name": self.name,
            "version": self.version,
            "max_query_chars": self.max_query_chars,
            "intent_rules": self.intent_rules,
            "route_to_intent": self.route_to_intent,
            "blocked_phrases": self.blocked_phrases,
            "pii_redaction": self.pii_redaction,
            "guardrail_rules": [r.to_dict() for r in self.guardrail_rules],
            "transaction_slot_keys": self.transaction_slot_keys,
        }
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "PolicyPack":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))

        # Parse guardrail rules from JSON, or fall back to defaults
        raw_rules = data.get("guardrail_rules")
        if isinstance(raw_rules, list) and raw_rules:
            rules = [
                GuardrailRule.from_dict(r) for r in raw_rules if isinstance(r, dict)
            ]
        else:
            rules = [GuardrailRule.from_dict(r) for r in _DEFAULT_RULES]

        tx_slot_keys = data.get("transaction_slot_keys")
        if not isinstance(tx_slot_keys, list):
            tx_slot_keys = [
                "payment_id",
                "transaction_id",
                "refund_id",
                "order_id",
            ]

        return PolicyPack(
            name=data.get("name", "default"),
            version=str(data.get("version", "0")),
            max_query_chars=int(data.get("max_query_chars", 4000)),
            intent_rules=dict(data.get("intent_rules", {})),
            route_to_intent=dict(data.get("route_to_intent", {})),
            blocked_phrases=list(data.get("blocked_phrases", [])),
            pii_redaction=bool(data.get("pii_redaction", False)),
            guardrail_rules=rules,
            transaction_slot_keys=tx_slot_keys,
        )
