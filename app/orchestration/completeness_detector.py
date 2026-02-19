# app/orchestration/completeness_detector.py
"""
AOP Completeness Detector (Li et al. 2024)

Audits a task decomposition plan for:
  - Completeness: does the plan address ALL aspects of the original query?
  - Non-redundancy: are there overlapping subtasks that can be merged?

Uses an LLM call to perform semantic analysis of coverage and overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.llm_client import chat_json


@dataclass
class CompletenessResult:
    """Result of completeness and non-redundancy analysis."""

    complete: bool
    missing: List[str]  # Aspects of query not covered by any subtask
    redundant: List[Tuple[str, str]]  # Pairs of subtasks that overlap
    coverage_ratio: float  # 0.0-1.0
    reasoning: str


class CompletenessDetector:
    """
    Audits an AOP plan for completeness and non-redundancy.

    Uses LLM to analyse whether the set of subtasks fully covers
    all aspects/intents in the original query, and detects
    overlapping subtask pairs.
    """

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    def check(
        self,
        query: str,
        subtasks: List[str],
        assignments: Dict[str, str],
    ) -> CompletenessResult:
        """
        Check if plan covers the query completely and without redundancy.

        Args:
            query: The original user query.
            subtasks: List of decomposed subtask descriptions.
            assignments: Mapping of subtask → assigned agent_id.

        Returns:
            CompletenessResult with analysis.
        """
        if not subtasks:
            return CompletenessResult(
                complete=False,
                missing=["No subtasks defined"],
                redundant=[],
                coverage_ratio=0.0,
                reasoning="Empty plan — no subtasks to evaluate.",
            )

        messages = self._build_check_prompt(query, subtasks, assignments)

        try:
            raw = chat_json(messages=messages, model=self.model, temperature=1.0)
        except Exception as e:
            return CompletenessResult(
                complete=True,  # Fail-open: assume complete if LLM unavailable
                missing=[],
                redundant=[],
                coverage_ratio=1.0,
                reasoning=f"LLM unavailable — defaulting to complete. Error: {e}",
            )

        return self._parse_response(raw)

    def _build_check_prompt(
        self,
        query: str,
        subtasks: List[str],
        assignments: Dict[str, str],
    ) -> List[Dict[str, str]]:
        plan_lines = []
        for i, st in enumerate(subtasks, 1):
            agent = assignments.get(st, "unassigned")
            plan_lines.append(f"  {i}. [{agent}] {st}")
        plan_str = "\n".join(plan_lines)

        system = (
            "You are an AOP completeness auditor. "
            "Given an original user query and a plan of subtasks (each assigned to an agent), "
            "evaluate:\n"
            "1. COMPLETENESS: Does the plan address ALL aspects/intents in the query?\n"
            "2. NON-REDUNDANCY: Are there subtask pairs that overlap in scope?\n\n"
            "Return STRICT JSON with these keys:\n"
            '  "complete": true/false,\n'
            '  "missing": ["aspect not covered", ...],  (empty list if complete)\n'
            '  "redundant": [["subtask_a", "subtask_b"], ...],  (empty list if no redundancy)\n'
            '  "coverage_ratio": 0.0-1.0,  (fraction of query aspects covered)\n'
            '  "reasoning": "brief explanation"\n'
        )

        user = f"Original query: {query}\n\nPlan:\n{plan_str}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _parse_response(self, raw: Dict[str, Any]) -> CompletenessResult:
        """Parse LLM JSON response into CompletenessResult with defensive handling."""
        try:
            complete = bool(raw.get("complete", True))
            missing = raw.get("missing", [])
            if not isinstance(missing, list):
                missing = []
            missing = [str(m) for m in missing]

            redundant_raw = raw.get("redundant", [])
            if not isinstance(redundant_raw, list):
                redundant_raw = []
            redundant: List[Tuple[str, str]] = []
            for pair in redundant_raw:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    redundant.append((str(pair[0]), str(pair[1])))

            coverage = raw.get("coverage_ratio", 1.0 if complete else 0.0)
            try:
                coverage = max(0.0, min(1.0, float(coverage)))
            except (TypeError, ValueError):
                coverage = 1.0 if complete else 0.0

            reasoning = str(raw.get("reasoning", ""))

            return CompletenessResult(
                complete=complete,
                missing=missing,
                redundant=redundant,
                coverage_ratio=coverage,
                reasoning=reasoning,
            )
        except Exception:
            return CompletenessResult(
                complete=True,
                missing=[],
                redundant=[],
                coverage_ratio=1.0,
                reasoning="Failed to parse LLM response — defaulting to complete.",
            )
