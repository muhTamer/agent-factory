# app/orchestration/solvability_estimator.py
"""
AOP Solvability Estimator (Li et al. 2024)

Estimates the probability that an agent can successfully execute a sub-task.
Uses a weighted combination of textual similarity and historical performance:

    Score = α · textual_similarity(subtask, agent_capabilities)
          + β · historical_performance(agent, similar_tasks)

α=0.6, β=0.4 by default (per Li et al.)

Textual similarity is computed via TF-IDF cosine similarity — deterministic,
fast, and testable without LLM calls.  The TF-IDF implementation reuses the
same tokenizer/cosine pattern from app/shared/rag.py.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from app.orchestration.performance_store import PerformanceStore

# ── TF-IDF helpers (same pattern as app/shared/rag.py) ──────────────

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s or "")]


def _tfidf_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Build a normalised TF-IDF sparse vector from a token list."""
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    vec: Dict[str, float] = {}
    norm = 0.0
    for t, f in tf.items():
        w = (1 + math.log(f)) * idf.get(t, 0.0)
        vec[t] = w
        norm += w * w
    norm = math.sqrt(max(1e-9, norm))
    for t in list(vec.keys()):
        vec[t] /= norm
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for t, w in a.items():
        w2 = b.get(t)
        if w2 is not None:
            s += w * w2
    return float(max(0.0, min(1.0, s)))


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class SolvabilityScore:
    """Score for a single (subtask, agent) pairing."""

    agent_id: str
    subtask: str
    textual_similarity: float
    historical_performance: float
    combined_score: float
    reasoning: str


@dataclass
class SolvabilityResult:
    """Result of estimating solvability across all agents for a set of subtasks."""

    assignments: Dict[str, str]  # subtask → best agent_id
    scores: List[SolvabilityScore]  # all evaluated pairs
    assignment_scores: Dict[str, float]  # subtask → best combined_score


# ── Estimator ────────────────────────────────────────────────────────


class SolvabilityEstimator:
    """
    AOP reward model: estimates solvability of each (subtask, agent) pairing.

    Fully deterministic — no LLM calls.  Uses TF-IDF cosine similarity for
    textual matching and PerformanceStore for historical performance.
    """

    def __init__(
        self,
        performance_store: PerformanceStore,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        self.store = performance_store
        self.alpha = alpha
        self.beta = beta

    def estimate(
        self,
        subtasks: List[str],
        agent_catalog: Dict[str, Dict[str, Any]],
    ) -> SolvabilityResult:
        """
        Estimate solvability for all (subtask, agent) pairs.

        Args:
            subtasks: List of natural-language subtask descriptions.
            agent_catalog: Output of registry.all_meta() — {agent_id: metadata_dict}.

        Returns:
            SolvabilityResult with optimal assignment and all scores.
        """
        if not subtasks or not agent_catalog:
            return SolvabilityResult(assignments={}, scores=[], assignment_scores={})

        # Build IDF from the combined corpus of subtask + agent texts.
        all_texts = list(subtasks) + [self._build_agent_text(m) for m in agent_catalog.values()]
        idf = self._build_idf(all_texts)

        # Pre-compute agent vectors.
        agent_vecs: Dict[str, Dict[str, float]] = {}
        for aid, meta in agent_catalog.items():
            agent_vecs[aid] = _tfidf_vec(_tok(self._build_agent_text(meta)), idf)

        all_scores: List[SolvabilityScore] = []
        assignments: Dict[str, str] = {}
        assignment_scores: Dict[str, float] = {}

        for subtask in subtasks:
            sub_vec = _tfidf_vec(_tok(subtask), idf)
            best_agent = ""
            best_combined = -1.0

            for aid, meta in agent_catalog.items():
                txt_sim = _cosine(sub_vec, agent_vecs[aid])
                hist_perf = self._historical_performance(aid)
                combined = self.alpha * txt_sim + self.beta * hist_perf

                reasoning = (
                    f"textual={txt_sim:.3f} (α={self.alpha}), "
                    f"historical={hist_perf:.3f} (β={self.beta}), "
                    f"combined={combined:.3f}"
                )

                score = SolvabilityScore(
                    agent_id=aid,
                    subtask=subtask,
                    textual_similarity=txt_sim,
                    historical_performance=hist_perf,
                    combined_score=combined,
                    reasoning=reasoning,
                )
                all_scores.append(score)

                if combined > best_combined:
                    best_combined = combined
                    best_agent = aid

            assignments[subtask] = best_agent
            assignment_scores[subtask] = best_combined

        return SolvabilityResult(
            assignments=assignments,
            scores=all_scores,
            assignment_scores=assignment_scores,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_idf(self, texts: List[str]) -> Dict[str, float]:
        """Compute IDF over a list of text documents."""
        df: Dict[str, int] = {}
        for text in texts:
            seen = set(_tok(text))
            for t in seen:
                df[t] = df.get(t, 0) + 1
        n = max(1, len(texts))
        return {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    def _textual_similarity(
        self, subtask: str, agent_meta: Dict[str, Any], idf: Dict[str, float]
    ) -> float:
        """TF-IDF cosine similarity between subtask and agent capability text."""
        sub_vec = _tfidf_vec(_tok(subtask), idf)
        agent_vec = _tfidf_vec(_tok(self._build_agent_text(agent_meta)), idf)
        return _cosine(sub_vec, agent_vec)

    def _historical_performance(self, agent_id: str) -> float:
        """Read average score from performance store (0.5 neutral prior)."""
        return self.store.agent_avg_score(agent_id)

    @staticmethod
    def _build_agent_text(agent_meta: Dict[str, Any]) -> str:
        """Concatenate agent metadata into a single text for TF-IDF."""
        parts = []
        desc = agent_meta.get("description", "")
        if desc:
            parts.append(str(desc))
        caps = agent_meta.get("capabilities", [])
        if isinstance(caps, list):
            parts.append(" ".join(str(c) for c in caps))
        atype = agent_meta.get("type", "")
        if atype:
            parts.append(str(atype))
        return " ".join(parts)
