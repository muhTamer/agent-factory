# app/runtime/policy/policy_ast.py
"""
Policy Abstract Syntax Tree (AST) - Canonical representation of compiled rules.

This module defines the immutable data structures that represent executable
policy rules after compilation from human-readable formats (YAML, PDF, text).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime


class RuleType(str, Enum):
    """Types of executable rules in the policy system."""

    ELIGIBILITY = "eligibility"
    AMOUNT_THRESHOLD = "amount_threshold"
    APPROVAL = "approval"
    RISK_CONTROL = "risk_control"
    EXECUTION = "execution"
    ESCALATION = "escalation"
    PRIVACY = "privacy"
    TRANSITION = "transition"  # For workflow state transitions
    TOOL_CALL = "tool_call"  # For determining when to call tools


class OperatorType(str, Enum):
    """Supported comparison operators in rule conditions."""

    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"  # regex


class LogicOperator(str, Enum):
    """Logical operators for combining conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass(frozen=True)
class Citation:
    """Provenance information linking a rule to its source."""

    source_file: str
    section: Optional[str] = None
    rule_id: Optional[str] = None
    line_range: Optional[tuple[int, int]] = None
    excerpt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "section": self.section,
            "rule_id": self.rule_id,
            "line_range": list(self.line_range) if self.line_range else None,
            "excerpt": self.excerpt,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Condition:
    """
    A single condition in a rule.

    Examples:
        Condition("kyc_status", OperatorType.EQ, "verified")
        Condition("transaction_age_days", OperatorType.LE, 90)
        Condition("account_status", OperatorType.IN, ["active", "pending"])
    """

    slot_name: str
    operator: OperatorType
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "operator": self.operator.value,
            "value": self.value,
        }

    def __str__(self) -> str:
        if self.operator == OperatorType.IN:
            return f"{self.slot_name} in {self.value}"
        elif self.operator == OperatorType.NOT_IN:
            return f"{self.slot_name} not in {self.value}"
        else:
            return f"{self.slot_name} {self.operator.value} {self.value}"


@dataclass(frozen=True)
class ConditionGroup:
    """
    A group of conditions combined with a logical operator.

    Supports nested groups for complex logic.
    """

    operator: LogicOperator
    conditions: List[Union[Condition, ConditionGroup]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator.value,
            "conditions": [
                c.to_dict() if isinstance(c, (Condition, ConditionGroup)) else c
                for c in self.conditions
            ],
        }


@dataclass(frozen=True)
class Action:
    """
    An action to be taken when a rule matches.

    Examples:
        Action("approve", {"auto": True})
        Action("escalate", {"team": "risk_review"})
        Action("transition", {"next_state": "approval_needed"})
        Action("call_tool", {"tool": "payment_processor", "method": "refund"})
    """

    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class CompiledRule:
    """
    A single executable rule in the policy system.

    This is the canonical representation after compilation. All runtime
    decisions should be made by evaluating these rules, not by calling LLMs.
    """

    rule_id: str
    rule_type: RuleType
    description: str

    # Core logic
    conditions: Union[Condition, ConditionGroup, None]
    actions: List[Action]

    # Metadata
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    citations: List[Citation] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Provenance
    compiled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    compiler_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type.value,
            "description": self.description,
            "conditions": self.conditions.to_dict() if self.conditions else None,
            "actions": [a.to_dict() for a in self.actions],
            "priority": self.priority,
            "enabled": self.enabled,
            "citations": [c.to_dict() for c in self.citations],
            "tags": self.tags,
            "compiled_at": self.compiled_at,
            "compiler_version": self.compiler_version,
        }


@dataclass(frozen=True)
class CompiledPolicyPack:
    """
    A complete set of compiled rules from one or more policy sources.

    This is what gets loaded at runtime and used by workflow agents
    to make deterministic decisions.
    """

    policy_id: str
    version: str
    domain: str  # e.g., "fintech", "retail", "telco"

    rules: List[CompiledRule]

    # Metadata
    source_files: List[str] = field(default_factory=list)
    source_hash: Optional[str] = None  # For change detection
    compiled_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Optional: slot schema extracted from rules
    slot_schema: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "domain": self.domain,
            "rules": [r.to_dict() for r in self.rules],
            "source_files": self.source_files,
            "source_hash": self.source_hash,
            "compiled_at": self.compiled_at,
            "slot_schema": self.slot_schema,
        }

    def get_rules_by_type(self, rule_type: RuleType) -> List[CompiledRule]:
        """Get all rules of a specific type."""
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rule_by_id(self, rule_id: str) -> Optional[CompiledRule]:
        """Get a specific rule by ID."""
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        return None


# Convenience type aliases
RuleCondition = Union[Condition, ConditionGroup]
