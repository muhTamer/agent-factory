# app/runtime/policy/policy_compiler.py
"""
Policy Compilation Pipeline

Main orchestrator that:
1. Discovers policy files
2. Routes to appropriate parser (YAML, text, PDF)
3. Compiles rules using LLM when needed
4. Validates compiled rules
5. Stores with hash-based change detection
6. Provides auto-recompilation on policy changes
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.runtime.policy.policy_ast import CompiledPolicyPack, CompiledRule
from app.runtime.policy.policy_parser import YAMLPolicyParser
from app.runtime.policy.llm_policy_compiler import (
    extract_policy_from_pdf,
    extract_policy_from_text,
)


class PolicyCompiler:
    """
    Main policy compilation pipeline.

    Usage:
        compiler = PolicyCompiler(
            llm_client=llm_client,
            domain="fintech",
        )

        pack = compiler.compile_policies([
            "data/refunds_policy.yaml",
            "data/compliance_guide.pdf",
        ])

        # Save compiled pack
        pack_path = compiler.save_pack(pack, ".factory/policy_pack.json")

        # Auto-recompile on changes
        if compiler.needs_recompilation(pack, pack_path):
            pack = compiler.recompile(pack_path)
    """

    def __init__(
        self,
        llm_client=None,
        domain: str = "general",
        model: str = "gpt-4o-mini",
    ):
        self.llm_client = llm_client
        self.domain = domain
        self.model = model

    def compile_policies(
        self,
        policy_files: List[Union[str, Path]],
        policy_id: Optional[str] = None,
        version: str = "1.0",
    ) -> CompiledPolicyPack:
        """
        Compile multiple policy files into a single policy pack.

        Args:
            policy_files: List of paths to policy files (YAML, PDF, TXT, etc.)
            policy_id: Unique identifier for this policy pack
            version: Version string

        Returns:
            CompiledPolicyPack with all rules from all sources
        """
        all_rules: List[CompiledRule] = []
        source_files = []

        for policy_file in policy_files:
            path = Path(policy_file)
            if not path.exists():
                print(f"[POLICY-COMPILE] File not found: {path}")
                continue

            print(f"[POLICY-COMPILE] Compiling {path.name}...")

            rules = self._compile_single_file(path)
            all_rules.extend(rules)
            source_files.append(str(path))

            print(f"[POLICY-COMPILE] Extracted {len(rules)} rules from {path.name}")

        # Generate source hash for change detection
        source_hash = self._compute_source_hash([Path(f) for f in source_files])

        # Auto-generate policy_id if not provided
        if not policy_id:
            policy_id = f"{self.domain}_policy_{source_hash[:8]}"

        # Build slot schema from rules
        slot_schema = self._extract_slot_schema(all_rules)

        pack = CompiledPolicyPack(
            policy_id=policy_id,
            version=version,
            domain=self.domain,
            rules=all_rules,
            source_files=source_files,
            source_hash=source_hash,
            slot_schema=slot_schema,
        )

        print(f"[POLICY-COMPILE] Compiled {len(all_rules)} total rules")
        print(f"[POLICY-COMPILE] Extracted {len(slot_schema)} slot definitions")

        return pack

    def _compile_single_file(self, path: Path) -> List[CompiledRule]:
        """Compile a single policy file based on its format."""
        suffix = path.suffix.lower()

        # YAML format (structured)
        if suffix in [".yaml", ".yml"]:
            parser = YAMLPolicyParser()
            return parser.parse(path)

        # PDF format (unstructured, needs LLM)
        elif suffix == ".pdf":
            return extract_policy_from_pdf(
                path,
                domain=self.domain,
                llm_client=self.llm_client,
                model=self.model,
            )

        # Plain text format (unstructured, needs LLM)
        elif suffix in [".txt", ".md"]:
            return extract_policy_from_text(
                path,
                domain=self.domain,
                llm_client=self.llm_client,
                model=self.model,
            )

        # Unsupported format
        else:
            print(f"[POLICY-COMPILE] Unsupported format: {suffix}")
            return []

    def _compute_source_hash(self, files: List[Path]) -> str:
        """Compute combined hash of all source files for change detection."""
        hasher = hashlib.sha256()

        for path in sorted(files):
            if path.exists():
                hasher.update(path.read_bytes())

        return hasher.hexdigest()

    def _extract_slot_schema(self, rules: List[CompiledRule]) -> Dict[str, Dict[str, Any]]:
        """
        Extract slot schema from compiled rules.

        Analyzes all conditions to infer:
        - Required slots
        - Data types
        - Possible values
        """
        schema: Dict[str, Dict[str, Any]] = {}

        for rule in rules:
            self._extract_slots_from_conditions(rule.conditions, schema)

        return schema

    def _extract_slots_from_conditions(self, conditions, schema: Dict[str, Dict[str, Any]]):
        """Recursively extract slot names from conditions."""
        from app.runtime.policy.policy_ast import Condition, ConditionGroup

        if conditions is None:
            return

        if isinstance(conditions, Condition):
            slot_name = conditions.slot_name

            if slot_name not in schema:
                schema[slot_name] = {
                    "type": self._infer_type(conditions.value),
                    "required": False,
                    "description": "Extracted from rule conditions",
                    "possible_values": set(),
                }

            # Track possible values
            if isinstance(conditions.value, (str, int, float, bool)):
                if "possible_values" not in schema[slot_name]:
                    schema[slot_name]["possible_values"] = set()
                schema[slot_name]["possible_values"].add(conditions.value)

        elif isinstance(conditions, ConditionGroup):
            for cond in conditions.conditions:
                self._extract_slots_from_conditions(cond, schema)

    def _infer_type(self, value: Any) -> str:
        """Infer data type from a value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        else:
            return "any"

    def save_pack(
        self,
        pack: CompiledPolicyPack,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Save compiled policy pack to JSON file.

        Returns:
            Path to saved file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and serialize
        data = pack.to_dict()

        # Convert sets to lists for JSON serialization
        for slot_name, slot_info in data.get("slot_schema", {}).items():
            if "possible_values" in slot_info and isinstance(slot_info["possible_values"], set):
                slot_info["possible_values"] = sorted(list(slot_info["possible_values"]), key=str)

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[POLICY-COMPILE] Saved policy pack to {path}")

        return path

    def load_pack(self, pack_path: Union[str, Path]) -> CompiledPolicyPack:
        """Load compiled policy pack from JSON file."""
        from app.runtime.policy.policy_ast import (
            Action,
            Citation,
            RuleType,
        )

        path = Path(pack_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Reconstruct rules from dict
        rules = []
        for rule_data in data.get("rules", []):
            # Reconstruct conditions
            conditions = None
            if rule_data.get("conditions"):
                conditions = self._dict_to_conditions(rule_data["conditions"])

            # Reconstruct actions
            actions = [
                Action(
                    action_type=a["action_type"],
                    parameters=a.get("parameters", {}),
                )
                for a in rule_data.get("actions", [])
            ]

            # Reconstruct citations
            citations = [
                Citation(
                    source_file=c["source_file"],
                    section=c.get("section"),
                    rule_id=c.get("rule_id"),
                    line_range=tuple(c["line_range"]) if c.get("line_range") else None,
                    excerpt=c.get("excerpt"),
                    metadata=c.get("metadata", {}),
                )
                for c in rule_data.get("citations", [])
            ]

            rule = CompiledRule(
                rule_id=rule_data["rule_id"],
                rule_type=RuleType(rule_data["rule_type"]),
                description=rule_data["description"],
                conditions=conditions,
                actions=actions,
                priority=rule_data.get("priority", 50),
                enabled=rule_data.get("enabled", True),
                citations=citations,
                tags=rule_data.get("tags", []),
                compiled_at=rule_data.get("compiled_at", ""),
                compiler_version=rule_data.get("compiler_version", "1.0.0"),
            )

            rules.append(rule)

        pack = CompiledPolicyPack(
            policy_id=data["policy_id"],
            version=data["version"],
            domain=data["domain"],
            rules=rules,
            source_files=data.get("source_files", []),
            source_hash=data.get("source_hash"),
            compiled_at=data.get("compiled_at", ""),
            slot_schema=data.get("slot_schema", {}),
        )

        return pack

    def _dict_to_conditions(self, cond_data: Dict[str, Any]):
        """Reconstruct conditions from dict."""
        from app.runtime.policy.policy_ast import (
            Condition,
            ConditionGroup,
            LogicOperator,
            OperatorType,
        )

        # Single condition
        if "slot_name" in cond_data:
            return Condition(
                slot_name=cond_data["slot_name"],
                operator=OperatorType(cond_data["operator"]),
                value=cond_data["value"],
            )

        # Condition group
        if "operator" in cond_data and "conditions" in cond_data:
            operator = LogicOperator(cond_data["operator"])
            conditions = [self._dict_to_conditions(c) for c in cond_data["conditions"]]
            return ConditionGroup(operator, conditions)

        return None

    def needs_recompilation(
        self,
        pack: CompiledPolicyPack,
        source_files: Optional[List[Union[str, Path]]] = None,
    ) -> bool:
        """
        Check if policy pack needs recompilation due to source changes.

        Args:
            pack: Current policy pack
            source_files: Optional list of source files to check (uses pack.source_files if not provided)

        Returns:
            True if sources have changed and recompilation is needed
        """
        files = source_files or pack.source_files
        files = [Path(f) for f in files]

        current_hash = self._compute_source_hash(files)

        return current_hash != pack.source_hash

    def recompile(
        self,
        pack_path: Union[str, Path],
    ) -> CompiledPolicyPack:
        """
        Recompile policies from original sources.

        Args:
            pack_path: Path to existing policy pack (to read source files)

        Returns:
            New compiled policy pack
        """
        # Load existing pack to get source files
        existing_pack = self.load_pack(pack_path)

        # Recompile from sources
        new_pack = self.compile_policies(
            existing_pack.source_files,
            policy_id=existing_pack.policy_id,
            version=existing_pack.version,
        )

        # Save updated pack
        self.save_pack(new_pack, pack_path)

        return new_pack

    def validate_pack(self, pack: CompiledPolicyPack) -> List[str]:
        """
        Validate compiled policy pack for common issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check for duplicate rule IDs
        rule_ids = [r.rule_id for r in pack.rules]
        duplicates = [rid for rid in set(rule_ids) if rule_ids.count(rid) > 1]
        if duplicates:
            issues.append(f"Duplicate rule IDs found: {duplicates}")

        # Check for rules without citations
        no_citations = [r.rule_id for r in pack.rules if not r.citations]
        if no_citations:
            issues.append(f"Rules without citations (not auditable): {no_citations}")

        # Check for conflicting rules (same priority, overlapping conditions)
        # This is a simplified check - full conflict detection would be more complex
        priority_groups = {}
        for rule in pack.rules:
            if rule.priority not in priority_groups:
                priority_groups[rule.priority] = []
            priority_groups[rule.priority].append(rule)

        for priority, rules in priority_groups.items():
            if len(rules) > 3:
                rule_ids = [r.rule_id for r in rules]
                issues.append(
                    f"Many rules at priority {priority}: {rule_ids} (potential conflicts)"
                )

        return issues
