# app/runtime/policy/llm_policy_compiler.py
"""
LLM-Assisted Policy Compiler

Uses LLM to extract structured rules from unstructured policy documents
(plain text, PDFs, Word docs). This is compilation-time LLM usage, NOT runtime.

The LLM's job: transform messy human policy → structured JSON rules
Runtime: evaluate structured rules deterministically (no LLM)
"""
from __future__ import annotations

import json
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


def compile_unstructured_policy(
    *,
    policy_text: str,
    source_file: str,
    domain: str,
    llm_client,
    model: str = "gpt-4o-mini",
) -> List[CompiledRule]:
    """
    Use LLM to extract structured rules from unstructured policy text.

    This is a ONE-TIME compilation step. The output rules are then used
    deterministically at runtime.

    Args:
        policy_text: Raw policy document text
        source_file: Path to source file (for citations)
        domain: Business domain (fintech, retail, etc.)
        llm_client: LLM client for making API calls
        model: Model to use for compilation

    Returns:
        List of compiled rules
    """

    system_prompt = """You are a policy compilation expert. Your job is to extract structured, executable rules from unstructured policy documents.

CRITICAL: You are converting human-readable policies into machine-executable rules. The output will be used at RUNTIME to make deterministic decisions WITHOUT any LLM involvement.

Your output MUST be valid JSON matching this schema:

{
  "rules": [
    {
      "rule_id": "unique_identifier",
      "rule_type": "eligibility|amount_threshold|risk_control|execution|escalation|privacy|transition|tool_call",
      "description": "Human-readable description",
      "conditions": {
        "operator": "and|or",
        "conditions": [
          {
            "slot_name": "field_name",
            "operator": "==|!=|<|<=|>|>=|in|not_in",
            "value": "value or [list]"
          }
        ]
      },
      "actions": [
        {
          "action_type": "approve|block|escalate|call_tool|transition|etc",
          "parameters": {}
        }
      ],
      "priority": 50,
      "section": "section name from source",
      "excerpt": "relevant quote from policy"
    }
  ]
}

RULES FOR EXTRACTION:

1. **Slot Names**: Extract field names from conditions (e.g., "kyc_status", "refund_amount", "transaction_age_days")
2. **Operators**: Map natural language to operators:
   - "is", "equals", "must be" → ==
   - "is not" → !=
   - "less than", "under", "below" → <
   - "at most", "up to" → <=
   - "more than", "over", "above" → >
   - "at least" → >=
   - "is one of", "is in" → in
   
3. **Values**: Extract literal values, numbers, lists
4. **Actions**: Identify what should happen when conditions match
5. **Priority**: Lower number = higher priority. Use:
   - 10-20: Privacy/security rules
   - 30-40: Risk controls
   - 50-60: Eligibility checks
   - 70-80: Amount thresholds
   - 90-100: General execution

6. **Citations**: Include section names and relevant excerpts for auditability

IMPORTANT: Be precise with slot names and values. These will be evaluated against actual runtime data.
"""

    user_prompt = f"""Policy Domain: {domain}
Source File: {source_file}

Policy Text:
{policy_text}

Extract ALL executable rules from this policy. Focus on:
- Eligibility criteria
- Amount thresholds
- Risk controls
- Execution rules
- Escalation conditions
- Privacy requirements
- Any conditional logic that determines workflow behavior

Return ONLY valid JSON matching the schema. No preamble, no explanation."""

    try:
        # Use the existing LLM client
        response = llm_client.chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
        )

        # Parse response
        if isinstance(response, str):
            response = json.loads(response)

        # Convert JSON rules to AST
        rules = []
        for rule_data in response.get("rules", []):
            rule = _json_to_compiled_rule(rule_data, source_file)
            if rule:
                rules.append(rule)

        return rules

    except Exception as e:
        print(f"[POLICY-COMPILE] LLM extraction failed: {e}")
        return []


def _json_to_compiled_rule(data: Dict[str, Any], source_file: str) -> Optional[CompiledRule]:
    """Convert JSON rule data to CompiledRule AST."""
    try:
        rule_id = data.get("rule_id", "unknown")
        rule_type = RuleType(data.get("rule_type", "execution"))
        description = data.get("description", "")
        priority = data.get("priority", 50)

        # Parse conditions
        conditions = None
        if data.get("conditions"):
            conditions = _parse_condition_json(data["conditions"])

        # Parse actions
        actions = []
        for action_data in data.get("actions", []):
            action = Action(
                action_type=action_data.get("action_type", "execute"),
                parameters=action_data.get("parameters", {}),
            )
            actions.append(action)

        # Build citation
        citation = Citation(
            source_file=source_file,
            section=data.get("section"),
            rule_id=rule_id,
            excerpt=data.get("excerpt"),
        )

        return CompiledRule(
            rule_id=rule_id,
            rule_type=rule_type,
            description=description,
            conditions=conditions,
            actions=actions,
            priority=priority,
            citations=[citation],
        )

    except Exception as e:
        print(f"[POLICY-COMPILE] Failed to parse rule {data.get('rule_id')}: {e}")
        return None


def _parse_condition_json(cond_data: Dict[str, Any]) -> Optional[Union[Condition, ConditionGroup]]:
    """Parse condition from JSON format."""
    if not cond_data:
        return None

    # Single condition
    if "slot_name" in cond_data:
        return Condition(
            slot_name=cond_data["slot_name"],
            operator=OperatorType(cond_data.get("operator", "==")),
            value=cond_data["value"],
        )

    # Condition group
    if "operator" in cond_data and "conditions" in cond_data:
        operator = LogicOperator(cond_data["operator"])
        conditions = [_parse_condition_json(c) for c in cond_data["conditions"]]
        conditions = [c for c in conditions if c is not None]

        if conditions:
            return ConditionGroup(operator, conditions)

    return None


def extract_policy_from_pdf(
    pdf_path: Path,
    domain: str,
    llm_client,
    model: str = "gpt-4o-mini",
) -> List[CompiledRule]:
    """
    Extract rules from a PDF policy document.

    Uses PDF text extraction + LLM compilation.
    """
    try:
        # Extract text from PDF
        import PyPDF2

        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"

        # Compile using LLM
        return compile_unstructured_policy(
            policy_text=text,
            source_file=str(pdf_path),
            domain=domain,
            llm_client=llm_client,
            model=model,
        )

    except Exception as e:
        print(f"[POLICY-COMPILE] PDF extraction failed for {pdf_path}: {e}")
        return []


def extract_policy_from_text(
    text_path: Path,
    domain: str,
    llm_client,
    model: str = "gpt-4o-mini",
) -> List[CompiledRule]:
    """Extract rules from plain text policy document."""
    try:
        text = text_path.read_text(encoding="utf-8")

        return compile_unstructured_policy(
            policy_text=text,
            source_file=str(text_path),
            domain=domain,
            llm_client=llm_client,
            model=model,
        )

    except Exception as e:
        print(f"[POLICY-COMPILE] Text extraction failed for {text_path}: {e}")
        return []
