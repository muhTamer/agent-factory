# app/runtime/policy/rule_engine.py
"""
Deterministic Rule Engine

Evaluates compiled rules against runtime slot values WITHOUT any LLM calls.
This is where the magic happens: workflow decisions are made by evaluating
rules, not by asking an LLM to guess.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.runtime.policy.policy_ast import (
    Action,
    CompiledPolicyPack,
    CompiledRule,
    Condition,
    ConditionGroup,
    LogicOperator,
    OperatorType,
    RuleType,
)


class RuleEvaluationResult:
    """Result of evaluating a rule against slot values."""

    def __init__(
        self,
        matched: bool,
        rule: CompiledRule,
        actions: List[Action],
        explanation: str = "",
    ):
        self.matched = matched
        self.rule = rule
        self.actions = actions if matched else []
        self.explanation = explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "rule_id": self.rule.rule_id,
            "rule_type": self.rule.rule_type.value,
            "actions": [a.to_dict() for a in self.actions],
            "explanation": self.explanation,
            "citations": [c.to_dict() for c in self.rule.citations],
        }


class RuleEngine:
    """
    Deterministic rule evaluation engine.

    Key principle: ALL decisions are made by evaluating rules against
    slot values. No LLM calls at runtime.
    """

    def __init__(self, policy_pack: CompiledPolicyPack):
        self.policy_pack = policy_pack
        self.rules_by_type: Dict[RuleType, List[CompiledRule]] = {}

        # Index rules by type for faster lookup
        for rule in policy_pack.rules:
            if rule.rule_type not in self.rules_by_type:
                self.rules_by_type[rule.rule_type] = []
            self.rules_by_type[rule.rule_type].append(rule)

        # Sort by priority (lower = higher priority)
        for rule_type in self.rules_by_type:
            self.rules_by_type[rule_type].sort(key=lambda r: r.priority)

    def evaluate_all(
        self,
        slots: Dict[str, Any],
        rule_type: Optional[RuleType] = None,
    ) -> List[RuleEvaluationResult]:
        """
        Evaluate all rules (or rules of a specific type) against slot values.

        Args:
            slots: Current slot values from the workflow
            rule_type: Optional filter for specific rule type

        Returns:
            List of evaluation results for matched rules
        """
        results = []

        rules_to_eval = []
        if rule_type:
            rules_to_eval = self.rules_by_type.get(rule_type, [])
        else:
            rules_to_eval = self.policy_pack.rules

        for rule in rules_to_eval:
            if not rule.enabled:
                continue

            result = self.evaluate_rule(rule, slots)
            if result.matched:
                results.append(result)

        return results

    def evaluate_rule(
        self,
        rule: CompiledRule,
        slots: Dict[str, Any],
    ) -> RuleEvaluationResult:
        """
        Evaluate a single rule against slot values.

        Returns:
            RuleEvaluationResult indicating if rule matched and what actions to take
        """
        if not rule.conditions:
            # Rule with no conditions always matches
            return RuleEvaluationResult(
                matched=True,
                rule=rule,
                actions=rule.actions,
                explanation="Rule has no conditions (always applies)",
            )

        matched, explanation = self._evaluate_condition(rule.conditions, slots)

        return RuleEvaluationResult(
            matched=matched,
            rule=rule,
            actions=rule.actions if matched else [],
            explanation=explanation,
        )

    def _evaluate_condition(
        self,
        condition: Union[Condition, ConditionGroup],
        slots: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Recursively evaluate a condition or condition group.

        Returns:
            (matched: bool, explanation: str)
        """
        if isinstance(condition, Condition):
            return self._evaluate_single_condition(condition, slots)

        elif isinstance(condition, ConditionGroup):
            return self._evaluate_condition_group(condition, slots)

        return False, "Invalid condition type"

    def _evaluate_single_condition(
        self,
        condition: Condition,
        slots: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Evaluate a single condition."""
        slot_value = slots.get(condition.slot_name)

        # Handle missing slot
        if slot_value is None:
            # Special case: checking if something is missing
            if condition.operator == OperatorType.EQ and condition.value is None:
                return True, f"{condition.slot_name} is None (as expected)"
            else:
                return False, f"{condition.slot_name} is missing from slots"

        # Evaluate based on operator
        try:
            matched = self._apply_operator(
                slot_value,
                condition.operator,
                condition.value,
            )

            if matched:
                explanation = f"{condition.slot_name}={slot_value} satisfies {condition}"
            else:
                explanation = f"{condition.slot_name}={slot_value} does not satisfy {condition}"

            return matched, explanation

        except Exception as e:
            return False, f"Evaluation error: {e}"

    def _apply_operator(
        self,
        slot_value: Any,
        operator: OperatorType,
        condition_value: Any,
    ) -> bool:
        """Apply a comparison operator."""

        if operator == OperatorType.EQ:
            return slot_value == condition_value

        elif operator == OperatorType.NE:
            return slot_value != condition_value

        elif operator == OperatorType.LT:
            return self._compare_numeric(slot_value, condition_value, lambda a, b: a < b)

        elif operator == OperatorType.LE:
            return self._compare_numeric(slot_value, condition_value, lambda a, b: a <= b)

        elif operator == OperatorType.GT:
            return self._compare_numeric(slot_value, condition_value, lambda a, b: a > b)

        elif operator == OperatorType.GE:
            return self._compare_numeric(slot_value, condition_value, lambda a, b: a >= b)

        elif operator == OperatorType.IN:
            if not isinstance(condition_value, (list, set, tuple)):
                return False
            return slot_value in condition_value

        elif operator == OperatorType.NOT_IN:
            if not isinstance(condition_value, (list, set, tuple)):
                return False
            return slot_value not in condition_value

        elif operator == OperatorType.CONTAINS:
            if isinstance(slot_value, str):
                return str(condition_value) in slot_value
            elif isinstance(slot_value, (list, set, tuple)):
                return condition_value in slot_value
            return False

        elif operator == OperatorType.MATCHES:
            if isinstance(slot_value, str) and isinstance(condition_value, str):
                return bool(re.match(condition_value, slot_value))
            return False

        return False

    def _compare_numeric(self, a: Any, b: Any, op) -> bool:
        """Helper for numeric comparisons."""
        try:
            # Try to convert to numbers for comparison
            a_num = float(a) if not isinstance(a, (int, float)) else a
            b_num = float(b) if not isinstance(b, (int, float)) else b
            return op(a_num, b_num)
        except (ValueError, TypeError):
            return False

    def _evaluate_condition_group(
        self,
        group: ConditionGroup,
        slots: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Evaluate a group of conditions with logical operator."""
        results = []
        explanations = []

        for condition in group.conditions:
            matched, explanation = self._evaluate_condition(condition, slots)
            results.append(matched)
            explanations.append(explanation)

        # Apply logical operator
        if group.operator == LogicOperator.AND:
            final_result = all(results)
            op_str = "AND"
        elif group.operator == LogicOperator.OR:
            final_result = any(results)
            op_str = "OR"
        elif group.operator == LogicOperator.NOT:
            final_result = not all(results)
            op_str = "NOT"
        else:
            final_result = False
            op_str = "UNKNOWN"

        explanation = f"{op_str}({', '.join(explanations)})"

        return final_result, explanation

    def check_eligibility(self, slots: Dict[str, Any]) -> Tuple[bool, List[RuleEvaluationResult]]:
        """
        Check if current state is eligible based on eligibility rules.

        Returns:
            (is_eligible: bool, matched_rules: List[RuleEvaluationResult])
        """
        results = self.evaluate_all(slots, RuleType.ELIGIBILITY)

        # Check if any blocking rules matched
        for result in results:
            for action in result.actions:
                if action.action_type == "block":
                    return False, results

        # If we have matched rules and none blocked, we're eligible
        if results:
            return True, results

        # No rules matched - default to eligible (permissive)
        return True, []

    def check_approval_required(
        self, slots: Dict[str, Any]
    ) -> Tuple[bool, Optional[RuleEvaluationResult]]:
        """
        Check if approval is required based on amount threshold rules.

        Returns:
            (approval_required: bool, matching_rule: Optional[RuleEvaluationResult])
        """
        results = self.evaluate_all(slots, RuleType.AMOUNT_THRESHOLD)

        for result in results:
            for action in result.actions:
                if action.action_type == "escalate":
                    return True, result

        return False, None

    def check_risk_controls(self, slots: Dict[str, Any]) -> List[RuleEvaluationResult]:
        """
        Check all risk control rules.

        Returns:
            List of matched risk control rules
        """
        return self.evaluate_all(slots, RuleType.RISK_CONTROL)

    def get_next_transition(
        self,
        current_state: str,
        slots: Dict[str, Any],
    ) -> Optional[str]:
        """
        Determine next workflow state based on transition rules.

        Returns:
            Next state name, or None if no transition rules match
        """
        # Add current_state to slots for evaluation
        eval_slots = {**slots, "current_state": current_state}

        results = self.evaluate_all(eval_slots, RuleType.TRANSITION)

        # Return first matching transition
        for result in results:
            for action in result.actions:
                if action.action_type == "transition":
                    return action.parameters.get("next_state")

        return None

    def should_call_tool(
        self,
        tool_name: str,
        slots: Dict[str, Any],
    ) -> Tuple[bool, Optional[RuleEvaluationResult]]:
        """
        Check if a tool should be called based on tool_call rules.

        Returns:
            (should_call: bool, matching_rule: Optional[RuleEvaluationResult])
        """
        results = self.evaluate_all(slots, RuleType.TOOL_CALL)

        for result in results:
            for action in result.actions:
                if action.action_type == "call_tool" and action.parameters.get("tool") == tool_name:
                    return True, result

        return False, None
