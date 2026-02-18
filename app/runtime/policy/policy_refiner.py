# app/runtime/policy/policy_refiner.py
"""
Policy Refiner - Polish customer-provided policies before compilation

This module handles the scenario where customers provide "messy" or incomplete
YAML policies that need refinement before they can be compiled into rules.

Pipeline:
  Customer YAML (messy) 
    → LLM Refiner (normalize, validate, enhance)
    → Clean YAML
    → Standard Compiler
    → Executable Rules

Use cases:
- Inconsistent formatting
- Missing required fields
- Ambiguous conditions
- Natural language instead of structured format
- Incomplete slot definitions
"""
from __future__ import annotations

import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class PolicyRefinementResult:
    """Result of policy refinement process."""

    def __init__(
        self,
        refined_yaml: str,
        original_yaml: str,
        issues_found: List[str],
        fixes_applied: List[str],
        confidence: float,
        needs_review: bool,
    ):
        self.refined_yaml = refined_yaml
        self.original_yaml = original_yaml
        self.issues_found = issues_found
        self.fixes_applied = fixes_applied
        self.confidence = confidence
        self.needs_review = needs_review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "refined_yaml": self.refined_yaml,
            "original_yaml": self.original_yaml,
            "issues_found": self.issues_found,
            "fixes_applied": self.fixes_applied,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }


class PolicyRefiner:
    """
    Refines customer-provided policies using LLM assistance.

    This is a PREPROCESSING step before compilation. The LLM helps normalize
    and validate the policy, but the actual rule compilation is still deterministic.
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.llm_client = llm_client
        self.model = model

    def refine_policy(
        self,
        policy_yaml: str,
        domain: str = "general",
        validation_strict: bool = True,
    ) -> PolicyRefinementResult:
        """
        Refine a customer-provided policy YAML.

        Args:
            policy_yaml: Raw YAML from customer
            domain: Business domain for context
            validation_strict: If True, fail on critical issues; if False, attempt fixes

        Returns:
            PolicyRefinementResult with refined YAML and details
        """
        # Step 1: Analyze the policy
        analysis = self._analyze_policy(policy_yaml, domain)

        # Step 2: If no issues, return as-is
        if not analysis["issues"]:
            return PolicyRefinementResult(
                refined_yaml=policy_yaml,
                original_yaml=policy_yaml,
                issues_found=[],
                fixes_applied=[],
                confidence=1.0,
                needs_review=False,
            )

        # Step 3: Attempt to fix issues
        refined = self._refine_with_llm(policy_yaml, analysis, domain)

        # Step 4: Validate the refined version
        validation = self._validate_refined_policy(refined["yaml"], domain)

        return PolicyRefinementResult(
            refined_yaml=refined["yaml"],
            original_yaml=policy_yaml,
            issues_found=analysis["issues"],
            fixes_applied=refined["fixes"],
            confidence=refined["confidence"],
            needs_review=validation["needs_review"] or refined["confidence"] < 0.8,
        )

    def _analyze_policy(self, policy_yaml: str, domain: str) -> Dict[str, Any]:
        """Analyze policy for issues using LLM."""

        system_prompt = """You are a policy document analyzer. Your job is to identify issues in customer-provided policy YAML files that would prevent them from being compiled into executable rules.

Analyze the policy and identify:
1. **Structural issues**: Missing required sections, invalid YAML syntax
2. **Formatting issues**: Inconsistent formatting, unclear structure
3. **Semantic issues**: Ambiguous conditions, unclear slot names
4. **Completeness issues**: Missing metadata, incomplete rule definitions
5. **Type issues**: Wrong data types, invalid operators

Return JSON with:
{
  "issues": [
    {
      "type": "structural|formatting|semantic|completeness|type",
      "severity": "critical|warning|info",
      "description": "Clear description of the issue",
      "location": "Where in the policy (section/line)",
      "suggestion": "How to fix it"
    }
  ],
  "overall_quality": "good|fair|poor",
  "can_auto_fix": true/false
}

Be specific and actionable. If there are no issues, return empty issues array."""

        user_prompt = f"""Domain: {domain}

Policy YAML to analyze:
```yaml
{policy_yaml}
```

Analyze this policy and identify all issues that would prevent successful compilation."""

        try:
            response = self.llm_client.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
            )

            if isinstance(response, str):
                response = json.loads(response)

            return response

        except Exception as e:
            print(f"[POLICY-REFINER] Analysis failed: {e}")
            return {
                "issues": [],
                "overall_quality": "unknown",
                "can_auto_fix": False,
            }

    def _refine_with_llm(
        self,
        policy_yaml: str,
        analysis: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        """Use LLM to refine the policy based on analysis."""

        system_prompt = """You are a policy refinement expert. Your job is to transform customer-provided policy YAML into clean, compilable format.

Follow these rules:
1. **Preserve intent**: Keep the original policy's meaning intact
2. **Standardize format**: Use consistent YAML structure
3. **Complete missing fields**: Add required metadata with sensible defaults
4. **Clarify ambiguity**: Convert natural language to structured conditions
5. **Normalize slot names**: Use snake_case, descriptive names
6. **Add type hints**: Ensure slot types are clear
7. **Add metadata**: Include policy_id, version, domain if missing

Standard YAML structure:
```yaml
metadata:
  policy_id: unique_identifier
  domain: domain_name
  version: "1.0"
  description: Brief description

eligibility:
  - id: rule_id
    description: What this rule checks
    conditions:
      - slot_name operator value
    outcome: eligible|ineligible
    rationale: Why (if ineligible)

amount_rules:
  - id: rule_id
    description: What this rule does
    threshold: number
    currency: EUR|USD
    approval: auto|manager_required
    conditions: (optional)
      - slot_name operator value

risk_controls:
  - id: rule_id
    description: What this checks
    condition:
      - slot_name operator value
    action: escalate_risk_review|compliance_hold|etc

execution:
  - id: rule_id
    description: How to execute
    rules: (optional)
      - if condition:
          action: action_name
```

Operators: ==, !=, <, <=, >, >=, in, not in

Return JSON:
{
  "yaml": "refined YAML as string",
  "fixes": ["List of fixes applied"],
  "confidence": 0.0-1.0,
  "notes": "Any important notes about the refinement"
}"""

        issues_summary = "\n".join(
            [
                f"- [{issue['severity']}] {issue['description']} → {issue['suggestion']}"
                for issue in analysis.get("issues", [])
            ]
        )

        user_prompt = f"""Domain: {domain}

Issues found:
{issues_summary}

Original policy:
```yaml
{policy_yaml}
```

Please refine this policy to fix all issues while preserving the original intent."""

        try:
            response = self.llm_client.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
            )

            if isinstance(response, str):
                response = json.loads(response)

            return {
                "yaml": response.get("yaml", policy_yaml),
                "fixes": response.get("fixes", []),
                "confidence": float(response.get("confidence", 0.5)),
                "notes": response.get("notes", ""),
            }

        except Exception as e:
            print(f"[POLICY-REFINER] Refinement failed: {e}")
            return {
                "yaml": policy_yaml,
                "fixes": [],
                "confidence": 0.0,
                "notes": f"Refinement failed: {e}",
            }

    def _validate_refined_policy(
        self,
        refined_yaml: str,
        domain: str,
    ) -> Dict[str, Any]:
        """Validate the refined policy."""

        try:
            # Parse YAML
            data = yaml.safe_load(refined_yaml)

            issues = []

            # Check required sections
            if "metadata" not in data:
                issues.append("Missing metadata section")

            # Check at least one rule section exists
            rule_sections = [
                "eligibility",
                "amount_rules",
                "risk_controls",
                "execution",
                "escalation",
                "privacy",
            ]

            has_rules = any(section in data for section in rule_sections)
            if not has_rules:
                issues.append("No rule sections found")

            # Validate rule IDs are unique
            all_ids = []
            for section in rule_sections:
                if section in data:
                    for rule in data[section]:
                        if isinstance(rule, dict) and "id" in rule:
                            all_ids.append(rule["id"])

            duplicates = [rid for rid in set(all_ids) if all_ids.count(rid) > 1]
            if duplicates:
                issues.append(f"Duplicate rule IDs: {duplicates}")

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "needs_review": len(issues) > 0,
            }

        except yaml.YAMLError as e:
            return {
                "valid": False,
                "issues": [f"Invalid YAML syntax: {e}"],
                "needs_review": True,
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "needs_review": True,
            }

    def interactive_refinement(
        self,
        policy_yaml: str,
        domain: str,
    ) -> PolicyRefinementResult:
        """
        Interactive refinement with multiple passes if needed.

        This attempts multiple refinement passes until confidence is high
        or maximum iterations reached.
        """
        max_iterations = 3
        current_yaml = policy_yaml
        all_fixes = []
        all_issues = []

        for iteration in range(max_iterations):
            print(f"[REFINE] Iteration {iteration + 1}/{max_iterations}")

            result = self.refine_policy(current_yaml, domain)

            all_issues.extend(result.issues_found)
            all_fixes.extend(result.fixes_applied)

            # If confidence is high enough, we're done
            if result.confidence >= 0.9:
                print(f"[REFINE] High confidence ({result.confidence:.2f}), stopping")
                break

            # If no fixes were applied, we can't improve further
            if not result.fixes_applied:
                print("[REFINE] No more fixes possible")
                break

            # Use refined version for next iteration
            current_yaml = result.refined_yaml

        return PolicyRefinementResult(
            refined_yaml=current_yaml,
            original_yaml=policy_yaml,
            issues_found=list(set(all_issues)),  # Deduplicate
            fixes_applied=all_fixes,
            confidence=result.confidence,
            needs_review=result.needs_review,
        )

    def batch_refine(
        self,
        policy_files: List[Path],
        output_dir: Path,
        domain: str,
    ) -> List[Tuple[Path, PolicyRefinementResult]]:
        """
        Refine multiple policy files in batch.

        Returns:
            List of (original_path, refinement_result) tuples
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for policy_file in policy_files:
            print(f"[REFINE] Processing {policy_file.name}...")

            try:
                original_yaml = policy_file.read_text(encoding="utf-8")

                result = self.interactive_refinement(original_yaml, domain)

                # Save refined version
                output_file = output_dir / f"{policy_file.stem}_refined.yaml"
                output_file.write_text(result.refined_yaml, encoding="utf-8")

                # Save refinement report
                report_file = output_dir / f"{policy_file.stem}_refinement_report.json"
                report_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

                print(f"[REFINE] Saved to {output_file}")
                print(f"[REFINE] Confidence: {result.confidence:.2f}")
                print(f"[REFINE] Needs review: {result.needs_review}")

                results.append((policy_file, result))

            except Exception as e:
                print(f"[REFINE] Failed to process {policy_file}: {e}")

        return results


def refine_policy_file(
    input_path: Path,
    output_path: Path,
    llm_client,
    domain: str = "general",
) -> PolicyRefinementResult:
    """
    Convenience function to refine a single policy file.

    Usage:
        result = refine_policy_file(
            Path("customer_policy.yaml"),
            Path("refined_policy.yaml"),
            llm_client,
            domain="fintech"
        )

        if result.needs_review:
            print("Manual review needed!")
            print("Issues:", result.issues_found)
        else:
            print("Policy ready for compilation!")
    """
    refiner = PolicyRefiner(llm_client)

    original_yaml = input_path.read_text(encoding="utf-8")
    result = refiner.interactive_refinement(original_yaml, domain)

    # Save refined version
    output_path.write_text(result.refined_yaml, encoding="utf-8")

    # Save report
    report_path = output_path.parent / f"{output_path.stem}_report.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    return result


def validate_customer_policy(
    policy_yaml: str,
    llm_client,
    domain: str = "general",
) -> Tuple[bool, List[str], List[str]]:
    """
    Quick validation of customer policy without refinement.

    Returns:
        (is_valid: bool, errors: List[str], warnings: List[str])
    """
    refiner = PolicyRefiner(llm_client)
    analysis = refiner._analyze_policy(policy_yaml, domain)

    errors = [
        issue["description"]
        for issue in analysis.get("issues", [])
        if issue["severity"] == "critical"
    ]

    warnings = [
        issue["description"]
        for issue in analysis.get("issues", [])
        if issue["severity"] == "warning"
    ]

    is_valid = len(errors) == 0

    return is_valid, errors, warnings
