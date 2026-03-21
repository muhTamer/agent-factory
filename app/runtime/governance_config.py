# app/runtime/governance_config.py
"""
Configurable Governance Levels for RQ3 Trade-off Evaluation

Defines three governance presets (LOW, MEDIUM, HIGH) that parameterize
all safety/compliance checks across the runtime pipeline:
  - Pre-guardrails (query validation, intent blocking)
  - Post-guardrails (blocked phrases, hallucination detection, tone control)
  - AOP autonomy (replanning, escalation thresholds, user confirmation)
  - Policy engine (auto-approval limits, risk checks, eligibility strictness)

Each level represents a different point on the safety-autonomy trade-off curve.
Running the same scenarios under all three levels produces the comparison data
needed to answer RQ3.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GovernanceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class GovernanceConfig:
    """Configurable governance knobs for RQ3 trade-off evaluation."""

    level: GovernanceLevel

    # Pre-guardrail knobs
    pre_checks_enabled: bool = True
    max_query_chars: int = 4000
    intent_blocking_enabled: bool = True

    # Post-guardrail knobs
    blocked_phrase_enforcement: bool = True
    hallucination_detection: bool = True
    tone_control_enabled: bool = True

    # Strictness knobs that differentiate MEDIUM from HIGH
    hallucination_strict: bool = False  # HIGH: skip informational bypass
    tone_violation_action: str = "mutate"  # "mutate" | "block"
    additional_blocked_phrases: tuple = ()  # HIGH: extra compliance phrases

    # AOP / Autonomy knobs
    allow_replanning: bool = True
    require_user_confirmation: bool = True
    escalation_threshold: float = 0.4
    max_autonomy_actions: int = 5  # 0 = unlimited

    # Policy engine knobs
    auto_approval_limit: float = 5000.0
    risk_check_enabled: bool = True
    strict_eligibility: str = "enforce"  # "log_only" | "enforce" | "enforce_escalate"

    @staticmethod
    def for_level(level: GovernanceLevel) -> GovernanceConfig:
        """Factory method returning preset config for each governance level."""
        if level == GovernanceLevel.LOW:
            return GovernanceConfig(
                level=level,
                pre_checks_enabled=False,
                max_query_chars=10000,
                intent_blocking_enabled=False,
                blocked_phrase_enforcement=False,
                hallucination_detection=False,
                tone_control_enabled=False,
                hallucination_strict=False,
                tone_violation_action="mutate",
                additional_blocked_phrases=(),
                allow_replanning=True,
                require_user_confirmation=False,
                escalation_threshold=0.1,
                max_autonomy_actions=0,  # unlimited
                auto_approval_limit=10000.0,
                risk_check_enabled=False,
                strict_eligibility="log_only",
            )
        elif level == GovernanceLevel.HIGH:
            return GovernanceConfig(
                level=level,
                pre_checks_enabled=True,
                max_query_chars=2000,
                intent_blocking_enabled=True,
                blocked_phrase_enforcement=True,
                hallucination_detection=True,
                tone_control_enabled=True,
                hallucination_strict=True,
                tone_violation_action="block",
                additional_blocked_phrases=(
                    "100% guaranteed",
                    "no risk",
                    "promise you",
                    "absolutely certain",
                    "I can confirm your refund",
                    "your refund has been",
                    "approved your",
                ),
                allow_replanning=False,
                require_user_confirmation=True,
                escalation_threshold=0.7,
                max_autonomy_actions=2,
                auto_approval_limit=1000.0,
                risk_check_enabled=True,
                strict_eligibility="enforce_escalate",
            )
        else:  # MEDIUM (default / current behaviour)
            return GovernanceConfig(
                level=level,
                pre_checks_enabled=True,
                max_query_chars=4000,
                intent_blocking_enabled=True,
                blocked_phrase_enforcement=True,
                hallucination_detection=True,
                tone_control_enabled=True,
                hallucination_strict=False,
                tone_violation_action="mutate",
                additional_blocked_phrases=(),
                allow_replanning=True,
                require_user_confirmation=True,
                escalation_threshold=0.4,
                max_autonomy_actions=5,
                auto_approval_limit=5000.0,
                risk_check_enabled=True,
                strict_eligibility="enforce",
            )
