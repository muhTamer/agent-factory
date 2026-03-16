# app/orchestration/scorer.py
"""
LLM-based Response Scorer for training data generation.

Scores agent responses on three dimensions (per AOP paper):
  1. Correctness  [0-1]
  2. Relevance    [0-1]
  3. Completeness [0-1]

Final score = average of the three dimensions.
Uses gpt-4o-mini via the existing app/llm_client.py.
"""

from __future__ import annotations

from typing import Any, Dict

from app.llm_client import chat_json

SCORER_SYSTEM_PROMPT = """You are an expert evaluator for multi-agent systems.
Score agent responses on 3 dimensions:

1. CORRECTNESS (0-1): Is the response factually accurate?
   - 1.0 = completely correct
   - 0.5 = partially correct
   - 0.0 = incorrect or nonsensical

2. RELEVANCE (0-1): Does it address the subtask?
   - 1.0 = directly addresses subtask
   - 0.5 = tangentially related
   - 0.0 = unrelated

3. COMPLETENESS (0-1): Does it fully resolve the subtask?
   - 1.0 = complete resolution
   - 0.5 = partial resolution
   - 0.0 = no resolution

Return STRICT JSON:
{"correctness": 0.0-1.0, "relevance": 0.0-1.0, "completeness": 0.0-1.0, "reasoning": "..."}"""


class LLMScorer:
    """Score agent responses using an LLM judge."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def score(self, subtask: str, response: Dict[str, Any]) -> float:
        """
        Score an agent's response quality for a given subtask.

        Args:
            subtask: The subtask description the agent was asked to solve.
            response: The agent's response dict (must contain readable text).

        Returns:
            Aggregated score in [0, 1] (average of correctness, relevance,
            completeness).
        """
        # Extract readable text from the response
        response_text = self._extract_text(response)
        if not response_text:
            return 0.0

        messages = [
            {"role": "system", "content": SCORER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"SUBTASK: {subtask}\n\n" f"AGENT RESPONSE:\n{response_text}"),
            },
        ]

        try:
            result = chat_json(messages=messages, model=self.model, temperature=0.0)
            correctness = float(result.get("correctness", 0.0))
            relevance = float(result.get("relevance", 0.0))
            completeness = float(result.get("completeness", 0.0))

            # Clamp to [0, 1]
            correctness = max(0.0, min(1.0, correctness))
            relevance = max(0.0, min(1.0, relevance))
            completeness = max(0.0, min(1.0, completeness))

            return (correctness + relevance + completeness) / 3.0
        except Exception as e:
            print(f"[Scorer] LLM scoring failed: {e}")
            return 0.0

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        """Extract readable text from an agent response dict."""
        if not response:
            return ""
        for key in ("text", "answer", "message", "response"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Fallback: stringify the whole dict
        return str(response)[:2000]
