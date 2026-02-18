# app/runtime/policy/policy_parser.py
"""
Policy Parser - Converts human-readable policy formats into AST.

Supports:
- YAML policies (structured)
- Plain text policies (LLM-assisted extraction)
- PDF policies (OCR + LLM extraction)
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.runtime.policy.policy_ast import (
    Action,
    Citation,
    CompiledRule,
    Condition,
    ConditionGroup,
    LogicOperator,
    OperatorType,
    RuleType,
)


class PolicyParser:
    """Base class for policy parsers."""

    def parse(self, source: Union[str, Path, Dict]) -> List[CompiledRule]:
        """Parse policy source into compiled rules."""
        raise NotImplementedError


class YAMLPolicyParser(PolicyParser):
    """
    Parser for structured YAML policies.

    Handles the format used in refunds_policy.yaml with sections like:
    - eligibility
    - amount_rules
    - risk_controls
    - execution
    - escalation
    - privacy
    """

    def parse(self, source: Union[str, Path, Dict]) -> List[CompiledRule]:
        """Parse YAML policy file."""
        if isinstance(source, (str, Path)):
            source = Path(source)
            with open(source, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            source_file = str(source)
        else:
            data = source
            source_file = "inline"

        rules = []

        # Parse eligibility rules
        if "eligibility" in data:
            rules.extend(
                self._parse_eligibility_section(data["eligibility"], source_file, "eligibility")
            )

        # Parse amount rules
        if "amount_rules" in data:
            rules.extend(
                self._parse_amount_rules_section(data["amount_rules"], source_file, "amount_rules")
            )

        # Parse risk controls
        if "risk_controls" in data:
            rules.extend(
                self._parse_risk_controls_section(
                    data["risk_controls"], source_file, "risk_controls"
                )
            )

        # Parse execution rules
        if "execution" in data:
            rules.extend(self._parse_execution_section(data["execution"], source_file, "execution"))

        # Parse escalation rules
        if "escalation" in data:
            rules.extend(
                self._parse_escalation_section(data["escalation"], source_file, "escalation")
            )

        # Parse privacy rules
        if "privacy" in data:
            rules.extend(self._parse_privacy_section(data["privacy"], source_file, "privacy"))

        return rules

    def _parse_eligibility_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse eligibility rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"eligibility_{idx}")
            description = item.get("description", "").strip()
            outcome = item.get("outcome", "eligible")

            # Parse conditions
            conditions = self._parse_conditions(item.get("conditions", []))

            # Determine action based on outcome
            if outcome == "eligible":
                action = Action("allow", {"reason": "Eligibility criteria met"})
            else:
                rationale = item.get("rationale", "Eligibility criteria not met")
                action = Action("block", {"reason": rationale})

            citation = Citation(
                source_file=source_file,
                section=section,
                rule_id=rule_id,
                excerpt=description[:200] if description else None,
            )

            rule = CompiledRule(
                rule_id=rule_id,
                rule_type=RuleType.ELIGIBILITY,
                description=description,
                conditions=conditions,
                actions=[action],
                citations=[citation],
                priority=50 + idx,  # Earlier rules have slightly higher priority
            )

            rules.append(rule)

        return rules

    def _parse_amount_rules_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse amount threshold rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"amount_{idx}")
            description = item.get("description", "").strip()
            threshold = item.get("threshold")
            currency = item.get("currency", "EUR")
            approval = item.get("approval", "auto")

            # Build conditions
            conditions_list = []

            if threshold is not None:
                # Parse the conditions if present
                if item.get("conditions"):
                    conditions_list.append(self._parse_conditions(item["conditions"]))
                else:
                    # Create threshold condition
                    if approval == "manager_required":
                        op = OperatorType.GT
                    else:
                        op = OperatorType.LE

                    conditions_list.append(Condition("refund_amount_requested", op, threshold))

            if item.get("conditions"):
                conditions_list.append(self._parse_conditions(item["conditions"]))

            # Combine conditions with AND
            if len(conditions_list) == 1:
                conditions = conditions_list[0]
            elif len(conditions_list) > 1:
                conditions = ConditionGroup(LogicOperator.AND, conditions_list)
            else:
                conditions = None

            # Determine action
            if approval == "auto":
                action = Action("approve", {"method": "auto", "threshold": threshold})
            elif approval == "manager_required":
                action = Action(
                    "escalate",
                    {
                        "team": "manager",
                        "reason": "Amount exceeds auto-approval threshold",
                    },
                )
            else:
                action = Action(approval, {"threshold": threshold})

            citation = Citation(
                source_file=source_file,
                section=section,
                rule_id=rule_id,
                excerpt=description[:200] if description else None,
                metadata={"threshold": threshold, "currency": currency},
            )

            rule = CompiledRule(
                rule_id=rule_id,
                rule_type=RuleType.AMOUNT_THRESHOLD,
                description=description,
                conditions=conditions,
                actions=[action],
                citations=[citation],
                priority=40 + idx,
            )

            rules.append(rule)

        return rules

    def _parse_risk_controls_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse risk control rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"risk_{idx}")
            description = item.get("description", "").strip()

            # Parse conditions (note: might be 'condition' or 'conditions')
            raw_conditions = item.get("condition") or item.get("conditions", [])
            conditions = self._parse_conditions(raw_conditions)

            # Parse action
            action_type = item.get("action", "escalate")
            action = Action(
                action_type,
                {"reason": (description[:100] if description else "Risk control triggered")},
            )

            citation = Citation(
                source_file=source_file,
                section=section,
                rule_id=rule_id,
                excerpt=description[:200] if description else None,
            )

            rule = CompiledRule(
                rule_id=rule_id,
                rule_type=RuleType.RISK_CONTROL,
                description=description,
                conditions=conditions,
                actions=[action],
                citations=[citation],
                priority=30 + idx,  # High priority for risk controls
            )

            rules.append(rule)

        return rules

    def _parse_execution_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse execution rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"execution_{idx}")
            description = item.get("description", "").strip()

            # Handle nested rules structure
            nested_rules = item.get("rules", [])
            if nested_rules:
                for ridx, nested in enumerate(nested_rules):
                    # Parse if-else structure
                    if_clause = nested.get("if")

                    if if_clause:
                        # Extract condition from if statement
                        condition_str = if_clause.split("==")[0].strip()
                        condition_val = (
                            if_clause.split("==")[1].strip() if "==" in if_clause else "true"
                        )

                        # Parse value (remove quotes)
                        if condition_val.startswith(("'", '"')):
                            condition_val = condition_val[1:-1]
                        elif condition_val.lower() == "true":
                            condition_val = True
                        elif condition_val.lower() == "false":
                            condition_val = False

                        conditions = Condition(condition_str, OperatorType.EQ, condition_val)
                        action_type = nested.get("action", "execute")

                        action = Action(
                            action_type,
                            {"description": description[:100] if description else ""},
                        )

                        citation = Citation(
                            source_file=source_file,
                            section=section,
                            rule_id=f"{rule_id}_{ridx}",
                            excerpt=description[:200] if description else None,
                        )

                        rule = CompiledRule(
                            rule_id=f"{rule_id}_{ridx}",
                            rule_type=RuleType.EXECUTION,
                            description=description,
                            conditions=conditions,
                            actions=[action],
                            citations=[citation],
                            priority=60 + idx * 10 + ridx,
                        )

                        rules.append(rule)
            else:
                # Simple execution rule
                action_type = item.get("action", "execute")
                action = Action(
                    action_type,
                    {"description": description[:100] if description else ""},
                )

                citation = Citation(
                    source_file=source_file,
                    section=section,
                    rule_id=rule_id,
                    excerpt=description[:200] if description else None,
                    metadata=item.get("sla", {}),
                )

                rule = CompiledRule(
                    rule_id=rule_id,
                    rule_type=RuleType.EXECUTION,
                    description=description,
                    conditions=None,
                    actions=[action],
                    citations=[citation],
                    priority=60 + idx,
                )

                rules.append(rule)

        return rules

    def _parse_escalation_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse escalation rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"escalation_{idx}")
            description = item.get("description", "").strip()

            action = Action(
                "escalate",
                {"reason": (description[:100] if description else "Escalation triggered")},
            )

            citation = Citation(
                source_file=source_file,
                section=section,
                rule_id=rule_id,
                excerpt=description[:200] if description else None,
            )

            rule = CompiledRule(
                rule_id=rule_id,
                rule_type=RuleType.ESCALATION,
                description=description,
                conditions=None,  # Escalation rules are typically conditional in workflow logic
                actions=[action],
                citations=[citation],
                priority=70 + idx,
            )

            rules.append(rule)

        return rules

    def _parse_privacy_section(
        self, items: List[Dict], source_file: str, section: str
    ) -> List[CompiledRule]:
        """Parse privacy rules."""
        rules = []

        for idx, item in enumerate(items):
            rule_id = item.get("id", f"privacy_{idx}")
            description = item.get("description", "").strip()
            action_type = item.get("action", "redact")

            action = Action(
                action_type,
                {"reason": description[:100] if description else "Privacy policy"},
            )

            citation = Citation(
                source_file=source_file,
                section=section,
                rule_id=rule_id,
                excerpt=description[:200] if description else None,
            )

            rule = CompiledRule(
                rule_id=rule_id,
                rule_type=RuleType.PRIVACY,
                description=description,
                conditions=None,
                actions=[action],
                citations=[citation],
                priority=20 + idx,  # Very high priority for privacy
            )

            rules.append(rule)

        return rules

    def _parse_conditions(
        self, raw_conditions: Union[List, Dict, str]
    ) -> Optional[Union[Condition, ConditionGroup]]:
        """
        Parse conditions from various formats.

        Handles:
        - List of condition strings: ["kyc_status == 'verified'", "account_status == 'active'"]
        - Single condition string: "kyc_status == 'verified'"
        - Dict with operator: {"and": [...], "or": [...]}
        """
        if not raw_conditions:
            return None

        # String format
        if isinstance(raw_conditions, str):
            return self._parse_condition_string(raw_conditions)

        # List format (implicit AND)
        if isinstance(raw_conditions, list):
            if len(raw_conditions) == 0:
                return None

            parsed = [self._parse_condition_string(c) for c in raw_conditions]
            parsed = [p for p in parsed if p is not None]

            if len(parsed) == 0:
                return None
            elif len(parsed) == 1:
                return parsed[0]
            else:
                return ConditionGroup(LogicOperator.AND, parsed)

        return None

    def _parse_condition_string(self, cond_str: str) -> Optional[Condition]:
        """
        Parse a single condition string.

        Examples:
            "kyc_status == 'verified'"
            "transaction_age_days <= 90"
            "account_status in ['active', 'pending']"
        """
        cond_str = cond_str.strip()

        # Try to parse with regex
        # Pattern: slot_name operator value
        for op_str, op_type in [
            (" in ", OperatorType.IN),
            (" not in ", OperatorType.NOT_IN),
            ("==", OperatorType.EQ),
            ("!=", OperatorType.NE),
            ("<=", OperatorType.LE),
            (">=", OperatorType.GE),
            ("<", OperatorType.LT),
            (">", OperatorType.GT),
        ]:
            if op_str in cond_str:
                parts = cond_str.split(op_str, 1)
                if len(parts) == 2:
                    slot_name = parts[0].strip()
                    value_str = parts[1].strip()

                    # Parse value
                    value = self._parse_value(value_str)

                    return Condition(slot_name, op_type, value)

        return None

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value from a string."""
        value_str = value_str.strip()

        # String (quoted)
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        if value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        # List
        if value_str.startswith("[") and value_str.endswith("]"):
            # Simple list parsing
            inner = value_str[1:-1]
            items = [i.strip().strip('"').strip("'") for i in inner.split(",")]
            return [i for i in items if i]

        # Number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass

        # Boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # Return as-is
        return value_str
