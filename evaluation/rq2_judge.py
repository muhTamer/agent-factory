# evaluation/rq2_judge.py
"""
LLM-as-Judge for RQ2 Explanation Quality Evaluation

Evaluates generated explanations against execution traces on three dimensions:
  1. Faithfulness — does the explanation match what the trace shows happened?
  2. Completeness — does it cover all significant decisions?
  3. Clarity — is it appropriate for its intended audience level?

Uses gpt-5-mini via app/llm_client.py. Scores each dimension 1-5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from app.llm_client import chat_json

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the quality of AI system explanations.

You will receive:
1. The original user query
2. The execution trace (ground truth of what the system actually did)
3. The generated explanation at a specific audience level
4. The raw system response

Evaluate the explanation on three dimensions, scoring each 1-5:

FAITHFULNESS: Does the explanation accurately describe what the trace shows happened?
  5: Every claim in the explanation is directly supported by the trace
  4: Almost all claims supported, minor imprecisions
  3: Mostly accurate but some unsupported or misleading claims
  2: Several inaccurate claims or significant misrepresentation
  1: Explanation contradicts or is unrelated to the trace

COMPLETENESS: Does the explanation cover all significant decisions in the trace?
  5: All routing decisions, agent selections, guardrail checks, and subtask decompositions mentioned
  4: Most significant decisions covered, only minor omissions
  3: Covers the main decision but misses secondary ones
  2: Misses several important decisions
  1: Barely covers what happened

CLARITY: Is the explanation understandable for its intended audience?
  For "summary" level: Should be plain language, no jargon, suitable for end users
  For "detailed" level: Should be technical but well-structured, suitable for auditors
  For "full" level: Should be comprehensive developer documentation with event-level detail
  5: Perfectly matches audience expectations
  4: Mostly appropriate, minor issues
  3: Adequate but could be better targeted
  2: Poorly matched to audience
  1: Completely wrong audience level

Return STRICT JSON:
{"faithfulness": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "justification": "<2-3 sentences>"}"""


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
