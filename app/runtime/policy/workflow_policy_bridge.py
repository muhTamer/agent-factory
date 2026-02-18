# app/runtime/policy/workflow_policy_bridge.py
"""
Workflow-Policy Bridge

Connects the rule engine to workflow runners, enabling deterministic
decisions based on compiled policies instead of LLM guesses.

This is the KEY integration point that stops workflows from getting stuck.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.runtime.policy.policy_ast import CompiledPolicyPack, RuleType
from app.runtime.policy.rule_engine import RuleEngine, RuleEvaluationResult


class WorkflowPolicyBridge:
    """
    Bridge between workflow execution and policy rules.

    Provides workflow-specific methods that evaluate rules to make
    deterministic decisions about:
    - State transitions
    - Tool calls
    - Approvals
    - Escalations
    - Data validation

    Usage in workflow runner:
        bridge = WorkflowPolicyBridge(policy_pack)

        # Check if approval is needed
        approval_needed, rule = bridge.check_approval_needed(slots)

        # Get next state based on rules
        next_state = bridge.get_next_state(current_state, slots)

        # Check if tool should be called
        should_call, rule = bridge.should_call_tool("payment_processor", slots)
    """

    def __init__(self, policy_pack: CompiledPolicyPack):
        self.policy_pack = policy_pack
        self.engine = RuleEngine(policy_pack)

    def check_eligibility(
        self,
        slots: Dict[str, Any],
    ) -> Tuple[bool, str, List[RuleEvaluationResult]]:
        """
        Check if request is eligible based on eligibility rules.

        Returns:
            (is_eligible: bool, reason: str, matched_rules: List[RuleEvaluationResult])
        """
        is_eligible, results = self.engine.check_eligibility(slots)

        # Build reason from matched rules
        if not is_eligible:
            # Find blocking rule
            for result in results:
                for action in result.actions:
                    if action.action_type == "block":
                        reason = action.parameters.get("reason", "Eligibility check failed")
                        return False, reason, results

        if results:
            reason = "Eligibility criteria met"
        else:
            reason = "No specific eligibility rules (default: eligible)"

        return is_eligible, reason, results

    def check_approval_needed(
        self,
        slots: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[RuleEvaluationResult]]:
        """
        Determine if approval is needed based on amount thresholds and other rules.

        This replaces LLM guessing about approval_needed vs no_approval_needed.

        Returns:
            (approval_needed: bool, reason: str, matched_rule: Optional[RuleEvaluationResult])
        """
        approval_needed, result = self.engine.check_approval_required(slots)

        if approval_needed and result:
            reason = None
            for action in result.actions:
                if action.action_type == "escalate":
                    reason = action.parameters.get("reason", "Approval required")
                    break

            return True, reason, result

        return False, None, None

    def get_required_approvals(
        self,
        slots: Dict[str, Any],
    ) -> List[str]:
        """
        Get list of approval teams/roles required.

        Returns:
            List of team/role names that need to approve
        """
        teams = []

        results = self.engine.evaluate_all(slots, RuleType.AMOUNT_THRESHOLD)

        for result in results:
            for action in result.actions:
                if action.action_type == "escalate":
                    team = action.parameters.get("team")
                    if team and team not in teams:
                        teams.append(team)

        return teams

    def check_risk_controls(
        self,
        slots: Dict[str, Any],
    ) -> Tuple[bool, List[str], List[RuleEvaluationResult]]:
        """
        Check all risk control rules.

        Returns:
            (has_risk_flags: bool, risk_reasons: List[str], matched_rules: List[RuleEvaluationResult])
        """
        results = self.engine.check_risk_controls(slots)

        if not results:
            return False, [], []

        reasons = []
        for result in results:
            for action in result.actions:
                reason = action.parameters.get("reason", result.rule.description)
                if reason not in reasons:
                    reasons.append(reason)

        return True, reasons, results

    def get_next_state(
        self,
        current_state: str,
        slots: Dict[str, Any],
        allowed_transitions: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Determine next workflow state based on transition rules.

        This is the KEY method that prevents workflows from getting stuck.
        Instead of LLM guessing the next event, we evaluate rules.

        Args:
            current_state: Current workflow state
            slots: Current slot values
            allowed_transitions: Optional list of allowed next states (from FSM)

        Returns:
            Next state name, or None if no rules match
        """
        next_state = self.engine.get_next_transition(current_state, slots)

        # Validate against allowed transitions if provided
        if next_state and allowed_transitions:
            if next_state not in allowed_transitions:
                print(
                    f"[POLICY-BRIDGE] Rule suggested {next_state} but not in allowed: {allowed_transitions}"
                )
                return None

        return next_state

    def should_call_tool(
        self,
        tool_name: str,
        slots: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[RuleEvaluationResult]]:
        """
        Check if a tool should be called based on rules.

        Returns:
            (should_call: bool, tool_params: Optional[Dict], matched_rule: Optional[RuleEvaluationResult])
        """
        should_call, result = self.engine.should_call_tool(tool_name, slots)

        if should_call and result:
            # Extract tool parameters from action
            for action in result.actions:
                if action.action_type == "call_tool":
                    return True, action.parameters, result

        return should_call, None, result

    def get_missing_slots(
        self,
        current_state: str,
        slots: Dict[str, Any],
    ) -> List[str]:
        """
        Determine which slots are missing for the current state.

        This helps the workflow know what to ask for next.

        Returns:
            List of missing slot names
        """
        # Get all rules that might apply to current state
        all_results = self.engine.evaluate_all(slots)

        missing = []

        # Analyze evaluation results to find slots that were missing
        for result in all_results:
            if not result.matched and "missing from slots" in result.explanation:
                # Parse slot name from explanation
                # Format: "slot_name is missing from slots"
                parts = result.explanation.split(" is missing from slots")
                if parts:
                    slot_name = parts[0].strip()
                    if slot_name not in missing:
                        missing.append(slot_name)

        # Also check slot schema for required fields
        for slot_name, slot_info in self.policy_pack.slot_schema.items():
            if slot_info.get("required") and slot_name not in slots:
                if slot_name not in missing:
                    missing.append(slot_name)

        return missing

    def validate_slots(
        self,
        slots: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate slot values against schema.

        Returns:
            (is_valid: bool, errors: List[str])
        """
        errors = []

        for slot_name, slot_value in slots.items():
            if slot_name not in self.policy_pack.slot_schema:
                continue  # Unknown slot, skip validation

            schema = self.policy_pack.slot_schema[slot_name]
            expected_type = schema.get("type", "any")

            # Type validation
            if expected_type == "integer" and not isinstance(slot_value, int):
                try:
                    int(slot_value)
                except (ValueError, TypeError):
                    errors.append(
                        f"{slot_name} must be an integer, got {type(slot_value).__name__}"
                    )

            elif expected_type == "number" and not isinstance(slot_value, (int, float)):
                try:
                    float(slot_value)
                except (ValueError, TypeError):
                    errors.append(f"{slot_name} must be a number, got {type(slot_value).__name__}")

            elif expected_type == "boolean" and not isinstance(slot_value, bool):
                errors.append(f"{slot_name} must be a boolean, got {type(slot_value).__name__}")

            elif expected_type == "string" and not isinstance(slot_value, str):
                errors.append(f"{slot_name} must be a string, got {type(slot_value).__name__}")

            # Value validation (if enum-like)
            possible_values = schema.get("possible_values")
            if possible_values and slot_value not in possible_values:
                errors.append(f"{slot_name}={slot_value} not in allowed values: {possible_values}")

        return len(errors) == 0, errors

    def get_audit_trail(
        self,
        slots: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get full audit trail of which rules matched and why.

        Returns:
            List of rule evaluation results with citations
        """
        all_results = self.engine.evaluate_all(slots)

        audit = []
        for result in all_results:
            if result.matched:
                audit.append(result.to_dict())

        return audit

    def explain_decision(
        self,
        decision_type: str,
        slots: Dict[str, Any],
    ) -> str:
        """
        Generate human-readable explanation of a decision.

        Args:
            decision_type: Type of decision (e.g., "approval", "eligibility")
            slots: Slot values that led to decision

        Returns:
            Human-readable explanation with citations
        """
        explanation_parts = [f"Decision: {decision_type}"]

        if decision_type == "approval":
            needed, reason, result = self.check_approval_needed(slots)
            if needed:
                explanation_parts.append("Result: Approval required")
                explanation_parts.append(f"Reason: {reason}")
            else:
                explanation_parts.append("Result: No approval required")

            if result:
                for citation in result.rule.citations:
                    explanation_parts.append(
                        f"Policy: {citation.source_file} - {citation.section} - {citation.rule_id}"
                    )
                    if citation.excerpt:
                        explanation_parts.append(f'  "{citation.excerpt[:100]}..."')

        elif decision_type == "eligibility":
            eligible, reason, results = self.check_eligibility(slots)
            explanation_parts.append(f"Result: {'Eligible' if eligible else 'Not eligible'}")
            explanation_parts.append(f"Reason: {reason}")

            for result in results:
                for citation in result.rule.citations:
                    explanation_parts.append(f"Policy: {citation.source_file} - {citation.rule_id}")

        return "\n".join(explanation_parts)
