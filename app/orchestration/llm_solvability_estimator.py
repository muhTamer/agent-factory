# app/orchestration/llm_solvability_estimator.py
"""
LLM-based Solvability Estimator

Uses an LLM to evaluate (subtask, agent) pairings by reasoning about
whether an agent's capabilities, domain, and tools match the subtask.

Compared to TF-IDF (lexical) and Neural (embedding+MLP), this estimator
leverages the LLM's semantic understanding to score agent-subtask fit.

Interface-compatible with SolvabilityEstimator and NeuralSolvabilityEstimator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.orchestration.performance_store import PerformanceStore


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class LLMSolvabilityScore:
    """Score for a single (subtask, agent) pairing."""

    agent_id: str
    subtask: str
    llm_score: float
    historical_performance: float
    combined_score: float
    reasoning: str


@dataclass
class LLMSolvabilityResult:
    """Result of LLM solvability estimation."""

    assignments: Dict[str, str]  # subtask → best agent_id
    scores: List[LLMSolvabilityScore]  # all evaluated pairs
    assignment_scores: Dict[str, float]  # subtask → best combined_score


# ── Estimator ───────────────────────────────────────────────────────


class LLMSolvabilityEstimator:
    """
    LLM reward model: uses an LLM to estimate solvability of each
    (subtask, agent) pairing by reasoning about semantic fit.

    Combined score = α · llm_score + β · historical_performance

    Interface-compatible with SolvabilityEstimator (TF-IDF) and
    NeuralSolvabilityEstimator (embedding+MLP).
    """

    def __init__(
        self,
        performance_store: PerformanceStore,
        alpha: float = 0.6,
        beta: float = 0.4,
        model: str = "gpt-5-mini",
        temperature: float = 0.1,
    ):
        self.store = performance_store
        self.alpha = alpha
        self.beta = beta
        self.model = model
        self.temperature = temperature

    # ── Public interface (matches SolvabilityEstimator) ──────────

    def estimate(
        self,
        subtasks: List[str],
        agent_catalog: Dict[str, Dict[str, Any]],
    ) -> LLMSolvabilityResult:
        """
        Estimate solvability for all (subtask, agent) pairs using LLM.

        Args:
            subtasks: Natural-language subtask descriptions.
            agent_catalog: Output of registry.all_meta() — {agent_id: metadata}.

        Returns:
            LLMSolvabilityResult with optimal assignments and all scores.
        """
        if not subtasks or not agent_catalog:
            return LLMSolvabilityResult(assignments={}, scores=[], assignment_scores={})

        # Build agent summaries for the prompt
        agent_summaries = {}
        for aid, meta in agent_catalog.items():
            agent_summaries[aid] = self._build_agent_summary(meta)

        # Call LLM once with all subtasks and all agents
        llm_scores = self._call_llm(subtasks, agent_summaries)

        all_scores: List[LLMSolvabilityScore] = []
        assignments: Dict[str, str] = {}
        assignment_scores: Dict[str, float] = {}

        for subtask in subtasks:
            best_agent = ""
            best_combined = -1.0

            for aid in agent_catalog:
                # LLM score
                llm_score = llm_scores.get(subtask, {}).get(aid, 0.0)

                # Historical performance
                hist = self.store.agent_avg_score(aid)

                # Combined
                combined = self.alpha * llm_score + self.beta * hist

                reasoning = (
                    f"llm={llm_score:.3f} hist={hist:.3f} " f"combined={combined:.3f}"
                )

                all_scores.append(
                    LLMSolvabilityScore(
                        agent_id=aid,
                        subtask=subtask,
                        llm_score=llm_score,
                        historical_performance=hist,
                        combined_score=combined,
                        reasoning=reasoning,
                    )
                )

                if combined > best_combined:
                    best_combined = combined
                    best_agent = aid

            if best_agent:
                assignments[subtask] = best_agent
                assignment_scores[subtask] = best_combined

        return LLMSolvabilityResult(
            assignments=assignments,
            scores=all_scores,
            assignment_scores=assignment_scores,
        )

    # ── LLM call ────────────────────────────────────────────────────

    def _call_llm(
        self,
        subtasks: List[str],
        agent_summaries: Dict[str, str],
    ) -> Dict[str, Dict[str, float]]:
        """Call LLM to score all (subtask, agent) pairs.

        Returns:
            Nested dict: {subtask: {agent_id: score_0_to_1}}.
        """
        from app.llm_client import chat_json

        system = (
            "You are a task-routing evaluator. For each subtask, score how well "
            "each agent can handle it on a scale of 0.0 to 1.0.\n\n"
            "Scoring criteria:\n"
            "  1. CAPABILITY FIT: Does the agent have the right tools, APIs, and "
            "capabilities to execute this subtask?\n"
            "  2. DOMAIN FIT: Does the agent's domain knowledge and policy coverage "
            "match the subtask's topic?\n"
            "  3. INTENT FIT: Is the subtask informational (needs FAQ/knowledge) or "
            "actionable (needs tools/policy)? Match accordingly:\n"
            "     - Informational subtask + agent with customer-facing docs → high score\n"
            "     - Actionable subtask + agent with internal policy & tools → high score\n"
            "     - Mismatch → low score\n\n"
            "Score guidelines:\n"
            "  0.9-1.0: Perfect fit — agent is clearly the right choice\n"
            "  0.6-0.8: Good fit — agent can handle it well\n"
            "  0.3-0.5: Partial fit — agent has some relevant capability\n"
            "  0.0-0.2: Poor fit — agent is not suited for this subtask\n\n"
            "Return STRICT JSON with this structure:\n"
            '{"scores": [\n'
            '  {"subtask": "...", "agent_id": "...", "score": 0.0-1.0, '
            '"reason": "brief explanation"},\n'
            "  ...\n"
            "]}\n\n"
            "You MUST return a score for EVERY (subtask, agent) combination."
        )

        user_data = {
            "subtasks": subtasks,
            "agents": {aid: summary for aid, summary in agent_summaries.items()},
        }

        try:
            raw = chat_json(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_data)},
                ],
                model=self.model,
                temperature=self.temperature,
                timeout=60,
            )
        except Exception as e:
            print(f"[LLM-Solvability] LLM call failed: {e}")
            # Return neutral scores on failure
            return {st: {aid: 0.5 for aid in agent_summaries} for st in subtasks}

        # Parse response
        result: Dict[str, Dict[str, float]] = {st: {} for st in subtasks}

        scores_list = raw.get("scores", [])
        for entry in scores_list:
            st = entry.get("subtask", "")
            aid = entry.get("agent_id", "")
            score = float(entry.get("score", 0.0))
            score = max(0.0, min(1.0, score))

            # Match subtask by exact or substring match
            matched_st = self._match_subtask(st, subtasks)
            if matched_st and aid in agent_summaries:
                result[matched_st][aid] = score

        # Fill any missing pairs with 0.0
        for st in subtasks:
            for aid in agent_summaries:
                if aid not in result[st]:
                    result[st][aid] = 0.0

        return result

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _match_subtask(candidate: str, subtasks: List[str]) -> Optional[str]:
        """Match LLM-returned subtask text to original subtask list."""
        if not candidate:
            return None
        # Exact match
        if candidate in subtasks:
            return candidate
        # Case-insensitive match
        lower = candidate.lower().strip()
        for st in subtasks:
            if st.lower().strip() == lower:
                return st
        # Substring match (LLM may truncate)
        for st in subtasks:
            if lower in st.lower() or st.lower() in lower:
                return st
        return None

    @staticmethod
    def _build_agent_summary(agent_meta: Dict[str, Any]) -> str:
        """Build a concise agent summary for the LLM prompt."""
        parts = []

        desc = agent_meta.get("description", "")
        if desc:
            parts.append(f"Description: {desc}")

        caps = agent_meta.get("capabilities", [])
        if isinstance(caps, list) and caps:
            parts.append(f"Capabilities: {', '.join(str(c) for c in caps)}")

        tools = agent_meta.get("available_tools", [])
        if isinstance(tools, list) and tools:
            parts.append(f"Tools: {', '.join(str(t) for t in tools)}")

        has_docs = agent_meta.get("has_customer_facing_docs", False)
        has_policy = agent_meta.get("has_internal_policy", False)
        if has_docs:
            parts.append("Has customer-facing documentation")
        if has_policy:
            parts.append("Has internal policy documents")

        doc_cats = agent_meta.get("document_categories", [])
        if doc_cats:
            parts.append(f"Document categories: {', '.join(str(c) for c in doc_cats)}")

        topics = agent_meta.get("coverage_topics", [])
        if topics:
            parts.append(f"Coverage topics: {', '.join(str(t) for t in topics)}")

        return " | ".join(parts)
