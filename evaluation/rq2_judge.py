# evaluation/rq2_judge.py
"""
LLM-as-Judge for RQ2 Explanation Quality Evaluation (IEEE-Grounded)

Evaluates generated explanations against IEEE standards requirements:
  1. Faithfulness (IEEE 2894-R7) — traceable to specific processing steps
  2. Completeness (IEEE 2894-R4/R5, 3152-R1/R2) — provenance, decision rationale,
     AI disclosure, and agent identity
  3. Clarity (IEEE 2894-R2/R3) — stakeholder-appropriate per IEEE audience levels

Uses gpt-5-mini via app/llm_client.py. Scores each dimension 1-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from app.llm_client import chat_json

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI system explanations against IEEE standards for AI governance.

You will receive:
1. The original user query
2. The execution trace (ground truth of what the system actually did)
3. The generated explanation at a specific audience level
4. The raw system response

Evaluate the explanation on three IEEE-grounded dimensions, scoring each 1-5:

FAITHFULNESS (IEEE 2894-2024, Requirement R7 — Traceability):
IEEE 2894-R7 requires that explanations are "traceable to specific processing steps."
  5: Every claim maps to a concrete trace event (route, select, execute, guard); provenance cites actual data sources
  4: Most claims traceable, minor gaps in linking to specific steps
  3: Partially traceable — mentions key decisions but does not ground them in trace events
  2: Vague references to system behavior with no trace-level grounding
  1: Explanation is fabricated or contradicts the trace

COMPLETENESS (IEEE 2894-2024 R4/R5 + IEEE 3152-2024 R1/R2):
IEEE 2894-R4 requires provenance (data sources, citations). IEEE 2894-R5 requires decision rationale.
IEEE 3152-R1 requires AI-generated nature disclosure. IEEE 3152-R2 requires agent identity disclosure.
  5: Includes provenance linking claims to data sources; documents decision rationale for routing/agent selection; discloses AI nature and which agent(s) handled the request
  4: Covers most of the above, minor omissions (e.g., missing one agent in a delegation chain)
  3: Has decision rationale but lacks provenance, or vice versa; partial agent disclosure
  2: Missing both provenance and agent identity; only superficial rationale
  1: No provenance, no rationale, no agent disclosure

CLARITY (IEEE 2894-2024 R2/R3 — Stakeholder-Appropriate Explanations):
IEEE 2894-R2 requires user-appropriate (summary) explanations. IEEE 2894-R3 requires auditor-appropriate (detailed) explanations.
  For "summary" level (IEEE 2894-R2): Plain language for end users — what happened, why, and what it means for them. No technical jargon.
  For "detailed" level (IEEE 2894-R3): Structured for auditors/compliance officers — decision points, governance checks, standards met/unmet.
  For "full" level: Developer documentation — event-level trace walkthrough, timing, scores, agent IDs.
  5: Perfectly calibrated to the IEEE-specified audience with appropriate depth
  4: Mostly appropriate, minor audience mismatch
  3: Adequate content but wrong framing for the audience
  2: Significant audience mismatch (e.g., technical jargon in summary, or superficial detail in full)
  1: Completely wrong audience level

Return STRICT JSON:
{"faithfulness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "justification": "<2-3 sentences explaining which IEEE requirements were met or missed>"}"""


def _truncate(obj: Any, max_chars: int = 4000) -> str:
    """Serialize and truncate to avoid exceeding context limits."""
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n...truncated..."
    return text


@dataclass
class RQ2JudgeResult:
    """Result of judging one explanation level."""

    level: str  # "summary", "detailed", "full"
    faithfulness: int  # 1-5
    completeness: int  # 1-5
    clarity: int  # 1-5
    justification: str = ""


def _clamp(val: Any, lo: int = 1, hi: int = 5, default: int = 3) -> int:
    """Clamp a value to [lo, hi], returning default if not parseable."""
    try:
        v = int(val)
        return max(lo, min(hi, v))
    except (ValueError, TypeError):
        return default


class RQ2ExplanationJudge:
    """LLM-as-judge for explanation faithfulness, completeness, and clarity."""

    def __init__(self, model: str = "gpt-5-mini", temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        level: str,
        query: str,
        trace_events: List[Dict[str, Any]],
        response: Dict[str, Any],
        explanation: Dict[str, Any],
    ) -> RQ2JudgeResult:
        """Judge a single explanation level against the trace ground truth."""
        user_prompt = (
            f"AUDIENCE LEVEL: {level}\n\n"
            f'ORIGINAL QUERY: "{query}"\n\n'
            f"EXECUTION TRACE (ground truth):\n{_truncate(trace_events)}\n\n"
            f"RAW SYSTEM RESPONSE:\n{_truncate(response)}\n\n"
            f"GENERATED EXPLANATION:\n"
            f"Narrative: {explanation.get('narrative', '')}\n"
            f"Agents involved: {explanation.get('agents_involved', [])}\n"
            f"Decisions: {_truncate(explanation.get('decisions', []), 2000)}\n"
            f"Provenance: {_truncate(explanation.get('provenance', []), 2000)}\n\n"
            f"Evaluate this explanation's faithfulness, completeness, and clarity."
        )

        try:
            result = chat_json(
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=self.temperature,
            )

            return RQ2JudgeResult(
                level=level,
                faithfulness=_clamp(result.get("faithfulness")),
                completeness=_clamp(result.get("completeness")),
                clarity=_clamp(result.get("clarity")),
                justification=str(result.get("justification", ""))[:500],
            )
        except Exception as e:
            print(f"[RQ2 Judge] Error evaluating {level}: {e}")
            return RQ2JudgeResult(
                level=level,
                faithfulness=0,
                completeness=0,
                clarity=0,
                justification=f"Judge error: {e}",
            )
