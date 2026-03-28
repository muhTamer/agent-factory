# app/orchestration/neural_solvability_estimator.py
"""
Neural Solvability Estimator (AOP paper — Li et al. 2024)

Drop-in replacement for the TF-IDF SolvabilityEstimator that uses
neural embeddings (all-MiniLM-L6-v2) and a trained 3-layer MLP to
estimate the probability that an agent can solve a subtask.

    Score = α · neural_similarity(subtask, agent_capabilities)
          + β · historical_performance(agent, similar_tasks)

α=0.6, β=0.4 by default (per Li et al.)

The embedding model is FROZEN — only the MLP layers are trainable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from app.orchestration.performance_store import PerformanceStore

# Lazy-load sentence-transformers to keep import time low when unused.
_EMBEDDER_CACHE: Dict[str, Any] = {}


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return a cached SentenceTransformer instance (downloaded on first use)."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


# ── MLP architecture ────────────────────────────────────────────────


class RewardMLP(nn.Module):
    """3-layer MLP: 768 → 256 → 64 → 1 with ReLU + Dropout."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass
class NeuralSolvabilityScore:
    """Score for a single (subtask, agent) pairing."""

    agent_id: str
    subtask: str
    neural_similarity: float
    historical_performance: float
    combined_score: float
    reasoning: str


@dataclass
class NeuralSolvabilityResult:
    """Result of neural solvability estimation."""

    assignments: Dict[str, str]  # subtask → best agent_id
    scores: List[NeuralSolvabilityScore]  # all evaluated pairs
    assignment_scores: Dict[str, float]  # subtask → best combined_score


# ── Estimator ───────────────────────────────────────────────────────


class NeuralSolvabilityEstimator:
    """
    Neural reward model: estimates solvability using sentence embeddings
    and a trained MLP, combined with historical performance.

    Interface-compatible with SolvabilityEstimator (TF-IDF baseline).
    """

    DEFAULT_MODEL_PATH = Path("models/reward_mlp.pt")

    # Intent-aware scoring constants (shared with TF-IDF estimator)
    _INFORMATIONAL_PREFIX = "informational:"
    _ACTION_PREFIX = "action:"
    _ACTION_PENALTY = 0.3
    _INTENT_KIND_BONUS = 0.15

    def __init__(
        self,
        performance_store: PerformanceStore,
        alpha: float = 0.6,
        beta: float = 0.4,
        model_path: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_intent_scoring: bool = True,
    ):
        self.store = performance_store
        self.alpha = alpha
        self.beta = beta
        self.use_intent_scoring = use_intent_scoring
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load sentence embedding model (frozen — never fine-tuned)
        self.embedder = _get_embedder(embedding_model)

        # Initialise MLP
        self.mlp = RewardMLP(input_dim=768)

        # Load trained weights if available
        resolved_path = (
            model_path if model_path is not None else self.DEFAULT_MODEL_PATH
        )
        if resolved_path and resolved_path.exists():
            self.mlp.load_state_dict(
                torch.load(resolved_path, map_location=self.device, weights_only=True)
            )
            self._trained = True
        else:
            warnings.warn(
                f"No trained model found at {resolved_path}, using random MLP weights"
            )
            self._trained = False

        self.mlp.to(self.device)
        self.mlp.eval()

    # ── Public interface (matches SolvabilityEstimator) ──────────

    def estimate(
        self,
        subtasks: List[str],
        agent_catalog: Dict[str, Dict[str, Any]],
    ) -> NeuralSolvabilityResult:
        """
        Estimate solvability for all (subtask, agent) pairs.

        Args:
            subtasks: Natural-language subtask descriptions.
            agent_catalog: Output of registry.all_meta() — {agent_id: metadata}.

        Returns:
            NeuralSolvabilityResult with optimal assignments and all scores.
        """
        if not subtasks or not agent_catalog:
            return NeuralSolvabilityResult(
                assignments={}, scores=[], assignment_scores={}
            )

        # Build agent text strings (identical logic to TF-IDF estimator)
        agent_ids = list(agent_catalog.keys())
        agent_texts = [self._build_agent_text(agent_catalog[aid]) for aid in agent_ids]

        # Encode all texts in batch
        with torch.no_grad():
            subtask_embs = self.embedder.encode(
                subtasks, convert_to_tensor=True, show_progress_bar=False
            )
            agent_embs = self.embedder.encode(
                agent_texts, convert_to_tensor=True, show_progress_bar=False
            )
            subtask_embs = subtask_embs.to(self.device)
            agent_embs = agent_embs.to(self.device)

        all_scores: List[NeuralSolvabilityScore] = []
        assignments: Dict[str, str] = {}
        assignment_scores: Dict[str, float] = {}

        for i, subtask in enumerate(subtasks):
            best_agent = ""
            best_combined = -1.0

            # Determine intent from decomposer prefix label
            lower = subtask.lower().lstrip()
            if lower.startswith(self._INFORMATIONAL_PREFIX):
                subtask_intent = "informational"
            elif lower.startswith(self._ACTION_PREFIX):
                subtask_intent = "action"
            else:
                subtask_intent = "unknown"

            for j, aid in enumerate(agent_ids):
                # Concatenate subtask + agent embeddings → 768d
                concat_emb = torch.cat([subtask_embs[i], agent_embs[j]]).unsqueeze(0)

                # MLP forward pass
                with torch.no_grad():
                    neural_sim = self.mlp(concat_emb).item()

                # Historical performance from PerformanceStore
                hist_perf = self.store.agent_avg_score(aid)

                # Combined score (same formula as TF-IDF estimator)
                combined = self.alpha * neural_sim + self.beta * hist_perf

                # Intent-aware scoring (same logic as TF-IDF estimator)
                modifiers = ""
                if self.use_intent_scoring:
                    meta = agent_catalog[aid]
                    agent_kind = meta.get("agent_kind", "")

                    if subtask_intent == "informational":
                        if meta.get("requires_user_context"):
                            combined *= self._ACTION_PENALTY
                            modifiers += " [penalty: info->action_agent]"
                        if agent_kind == "knowledge_rag":
                            combined += self._INTENT_KIND_BONUS
                            modifiers += " [bonus: info->knowledge]"
                    elif subtask_intent == "action":
                        if not meta.get("requires_user_context"):
                            combined *= self._ACTION_PENALTY
                            modifiers += " [penalty: action->knowledge_agent]"
                        if agent_kind == "workflow_runner":
                            combined += self._INTENT_KIND_BONUS
                            modifiers += " [bonus: action->workflow]"

                reasoning = (
                    f"neural={neural_sim:.3f} (a={self.alpha}), "
                    f"historical={hist_perf:.3f} (b={self.beta}), "
                    f"combined={combined:.3f}{modifiers}"
                )

                score = NeuralSolvabilityScore(
                    agent_id=aid,
                    subtask=subtask,
                    neural_similarity=neural_sim,
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

        return NeuralSolvabilityResult(
            assignments=assignments,
            scores=all_scores,
            assignment_scores=assignment_scores,
        )

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _build_agent_text(agent_meta: Dict[str, Any]) -> str:
        """Concatenate agent metadata into a single text for embedding.

        MUST match SolvabilityEstimator._build_agent_text() exactly
        to ensure apples-to-apples comparison.
        """
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
        akind = agent_meta.get("agent_kind", "")
        if akind and akind != atype:
            parts.append(str(akind))
        return " ".join(parts)

    @property
    def is_trained(self) -> bool:
        """Return True if the MLP was loaded from a trained checkpoint."""
        return self._trained
