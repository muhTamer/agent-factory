# evaluation/rq4/judge.py
"""
LLM-as-Judge — RQ4 Evaluation Engine

Simulates customer personas evaluating system responses on three dimensions:
  - Transparency (1-5): Can the customer understand the reasoning?
  - Trust (1-5): Does the customer trust the system?
  - Satisfaction (1-5): Is the customer satisfied with the response?

Supports two modes:
  - Real mode: Uses app.llm_client.chat_json for actual LLM evaluation
  - Mock mode: Returns deterministic persona-biased scores for CI/testing
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Protocol

from evaluation.rq4.personas import Persona
from evaluation.rq4.strategies import Strategy

# ── Result dataclass ──────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """Outcome of a single persona evaluation of a system response."""

    scenario_id: str
    strategy_name: str
    persona_name: str
    transparency: int  # 1-5
    trust: int  # 1-5
    satisfaction: int  # 1-5
    justification: str  # LLM-generated explanation
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("raw_response", None)
        return d


# ── Judge protocol ────────────────────────────────────────────────────


class Judge(Protocol):
    """Protocol for judge implementations (real and mock)."""

    def evaluate(
        self,
        persona: Persona,
        strategy: Strategy,
        scenario_id: str,
        scenario_description: str,
        query: str,
        response_text: str,
    ) -> JudgeResult: ...


# ── Real LLM Judge ────────────────────────────────────────────────────


class LLMJudge:
    """Uses a real LLM to evaluate responses from a persona's perspective."""

    def __init__(self, model: Optional[str] = None, temperature: float = 1.0):
        self.model = model
        self.temperature = temperature

    def evaluate(
        self,
        persona: Persona,
        strategy: Strategy,
        scenario_id: str,
        scenario_description: str,
        query: str,
        response_text: str,
    ) -> JudgeResult:
        from app.llm_client import chat_json

        system_prompt = self._build_system_prompt(persona)
        user_prompt = self._build_user_prompt(
            persona, scenario_description, query, response_text
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {"messages": messages}
        if self.temperature != 1.0:
            kwargs["temperature"] = self.temperature
        if self.model:
            kwargs["model"] = self.model

        raw = chat_json(**kwargs)

        return self._parse_response(raw, scenario_id, strategy.slug, persona.name)

    def _build_system_prompt(self, persona: Persona) -> str:
        return (
            "You are an expert evaluator simulating a specific customer persona.\n"
            "Your task is to evaluate a customer service chatbot response from "
            "the perspective of this persona.\n\n"
            f"PERSONA ROLE:\n{persona.judge_system_prompt}\n\n"
            "You MUST respond with valid JSON matching this exact schema:\n"
            "{\n"
            '  "transparency": <integer 1-5>,\n'
            '  "trust": <integer 1-5>,\n'
            '  "satisfaction": <integer 1-5>,\n'
            '  "justification": "<2-3 sentence explanation>"\n'
            "}\n\n"
            "Scoring guide:\n"
            "  1 = Very poor  2 = Poor  3 = Adequate  4 = Good  5 = Excellent\n\n"
            "Transparency: How well can you understand WHY the system responded "
            "this way? Does it explain its reasoning?\n"
            "Trust: How much do you trust this system to handle your request "
            "correctly and safely?\n"
            "Satisfaction: How satisfied are you with this response overall, "
            "given your specific needs and priorities?\n"
        )

    def _build_user_prompt(
        self,
        persona: Persona,
        scenario_description: str,
        query: str,
        response_text: str,
    ) -> str:
        return (
            f"SCENARIO: {scenario_description}\n\n"
            f"YOUR PRIORITIES (as {persona.name}): "
            f"{', '.join(persona.priorities)}\n\n"
            f'CUSTOMER QUERY:\n"{query}"\n\n'
            f'SYSTEM RESPONSE:\n"{response_text}"\n\n'
            "Now evaluate this interaction from your persona's perspective. "
            "Rate transparency, trust, and satisfaction (each 1-5) and "
            "provide a brief justification."
        )

    @staticmethod
    def _parse_response(
        raw: Dict[str, Any],
        scenario_id: str,
        strategy_name: str,
        persona_name: str,
    ) -> JudgeResult:
        """Parse LLM response into a JudgeResult, with fallbacks."""

        def _clamp(val: Any, default: int = 3) -> int:
            try:
                v = int(val)
                return max(1, min(5, v))
            except (TypeError, ValueError):
                return default

        return JudgeResult(
            scenario_id=scenario_id,
            strategy_name=strategy_name,
            persona_name=persona_name,
            transparency=_clamp(raw.get("transparency")),
            trust=_clamp(raw.get("trust")),
            satisfaction=_clamp(raw.get("satisfaction")),
            justification=str(raw.get("justification", "No justification provided.")),
            raw_response=raw,
        )


# ── Mock Judge (deterministic, no API calls) ──────────────────────────


# Persona-specific score biases: each persona naturally rates certain
# dimensions higher or lower based on their priorities.
_PERSONA_BIASES: Dict[str, Dict[str, Dict[str, int]]] = {
    # persona_slug -> strategy_slug -> {transparency, trust, satisfaction}
    "trust_seeker": {
        "baseline": {"transparency": 2, "trust": 2, "satisfaction": 3},
        "transparent": {"transparency": 5, "trust": 4, "satisfaction": 4},
        "empathetic": {"transparency": 3, "trust": 3, "satisfaction": 3},
        "proactive": {"transparency": 3, "trust": 3, "satisfaction": 4},
    },
    "efficiency_expert": {
        "baseline": {"transparency": 3, "trust": 3, "satisfaction": 4},
        "transparent": {"transparency": 4, "trust": 4, "satisfaction": 3},
        "empathetic": {"transparency": 3, "trust": 3, "satisfaction": 2},
        "proactive": {"transparency": 3, "trust": 3, "satisfaction": 4},
    },
    "tech_novice": {
        "baseline": {"transparency": 2, "trust": 2, "satisfaction": 2},
        "transparent": {"transparency": 3, "trust": 3, "satisfaction": 3},
        "empathetic": {"transparency": 3, "trust": 4, "satisfaction": 4},
        "proactive": {"transparency": 3, "trust": 3, "satisfaction": 4},
    },
    "frustrated_complainer": {
        "baseline": {"transparency": 2, "trust": 2, "satisfaction": 1},
        "transparent": {"transparency": 4, "trust": 3, "satisfaction": 3},
        "empathetic": {"transparency": 3, "trust": 4, "satisfaction": 5},
        "proactive": {"transparency": 3, "trust": 3, "satisfaction": 3},
    },
    "detail_oriented": {
        "baseline": {"transparency": 2, "trust": 3, "satisfaction": 3},
        "transparent": {"transparency": 5, "trust": 5, "satisfaction": 5},
        "empathetic": {"transparency": 3, "trust": 3, "satisfaction": 3},
        "proactive": {"transparency": 4, "trust": 4, "satisfaction": 4},
    },
    "first_time_user": {
        "baseline": {"transparency": 2, "trust": 3, "satisfaction": 3},
        "transparent": {"transparency": 3, "trust": 3, "satisfaction": 3},
        "empathetic": {"transparency": 3, "trust": 4, "satisfaction": 5},
        "proactive": {"transparency": 4, "trust": 4, "satisfaction": 4},
    },
    "regulatory_aware": {
        "baseline": {"transparency": 2, "trust": 2, "satisfaction": 2},
        "transparent": {"transparency": 5, "trust": 4, "satisfaction": 4},
        "empathetic": {"transparency": 2, "trust": 3, "satisfaction": 3},
        "proactive": {"transparency": 4, "trust": 4, "satisfaction": 4},
    },
}

# Per-category score adjustments (some scenarios are harder for all personas)
_CATEGORY_ADJUSTMENTS: Dict[str, int] = {
    "informational": 0,
    "transactional": 0,
    "complaint": -1,  # complaints are harder to satisfy
    "complex_multi_intent": 0,
    "trust_sensitive": -1,  # trust-sensitive scenarios score lower by default
}


class MockJudge:
    """Deterministic judge that returns persona-biased scores without API calls.

    Scores are derived from persona x strategy biases plus a small
    scenario-dependent perturbation (based on scenario_id hash) to
    simulate natural variance.
    """

    def evaluate(
        self,
        persona: Persona,
        strategy: Strategy,
        scenario_id: str,
        scenario_description: str,
        query: str,
        response_text: str,
    ) -> JudgeResult:
        # Look up base scores
        persona_scores = _PERSONA_BIASES.get(persona.slug, {})
        strategy_scores = persona_scores.get(
            strategy.slug,
            {"transparency": 3, "trust": 3, "satisfaction": 3},
        )

        # Small deterministic perturbation from scenario_id
        h = int(
            hashlib.md5(f"{scenario_id}:{persona.slug}".encode()).hexdigest()[:4], 16
        )
        perturbation = (h % 3) - 1  # -1, 0, or +1

        # Category adjustment
        category = self._extract_category(scenario_id)
        cat_adj = _CATEGORY_ADJUSTMENTS.get(category, 0)

        def _score(base: int) -> int:
            return max(1, min(5, base + perturbation + cat_adj))

        transparency = _score(strategy_scores["transparency"])
        trust = _score(strategy_scores["trust"])
        satisfaction = _score(strategy_scores["satisfaction"])

        justification = (
            f"[MOCK] As {persona.name}, the {strategy.slug} strategy "
            f"{'meets' if satisfaction >= 3 else 'does not meet'} my "
            f"expectations for {', '.join(persona.priorities[:2])}."
        )

        return JudgeResult(
            scenario_id=scenario_id,
            strategy_name=strategy.slug,
            persona_name=persona.name,
            transparency=transparency,
            trust=trust,
            satisfaction=satisfaction,
            justification=justification,
        )

    @staticmethod
    def _extract_category(scenario_id: str) -> str:
        """Extract category prefix from scenario_id (e.g. 'info_01' -> 'informational')."""
        prefix_map = {
            "info": "informational",
            "txn": "transactional",
            "comp": "complaint",
            "multi": "complex_multi_intent",
            "trust": "trust_sensitive",
        }
        prefix = scenario_id.split("_")[0]
        return prefix_map.get(prefix, "informational")
