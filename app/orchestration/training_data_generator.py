# app/orchestration/training_data_generator.py
"""
Training Data Generator for the Neural Solvability Estimator.

Generates (subtask, agent_description, score) triples by:
  1. Decomposing diverse queries into subtasks via AOP
  2. Selecting top-l candidate agents per subtask (TF-IDF pre-ranking)
  3. Executing each candidate agent on the subtask
  4. Scoring the response with LLMScorer
  5. Saving results as training data JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.orchestration.scorer import LLMScorer
from app.orchestration.solvability_estimator import SolvabilityEstimator


class TrainingDataGenerator:
    """Generate training data for the neural reward model."""

    def __init__(
        self,
        aop_coordinator,
        registry,
        scorer: Optional[LLMScorer] = None,
        num_agents_per_subtask: Optional[int] = None,
    ):
        """
        Args:
            aop_coordinator: AOPCoordinator instance (for decomposition).
            registry: AgentRegistry instance.
            scorer: LLMScorer instance (created if None).
            num_agents_per_subtask: Number of agents to evaluate per subtask
                (defaults to half the registry).
        """
        self.aop = aop_coordinator
        self.registry = registry
        self.scorer = scorer or LLMScorer()
        all_meta = registry.all_meta()
        self.l = num_agents_per_subtask or max(1, len(all_meta) // 2)

    def generate_from_queries(
        self,
        queries: List[str],
        output_path: Path,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate training data from a list of user queries.

        Args:
            queries: Diverse user queries to decompose and evaluate.
            output_path: Path to save the training data JSON.
            context: Optional context dict passed to agent.handle().

        Returns:
            List of training examples.
        """
        context = context or {}
        training_data: List[Dict[str, Any]] = []
        agent_catalog = self.registry.all_meta()

        # Use existing TF-IDF estimator for fast pre-ranking
        estimator = SolvabilityEstimator(self.aop.store)

        total = len(queries)
        for qi, query in enumerate(queries, 1):
            print(f"[TrainGen] Query {qi}/{total}: {query[:80]}...")

            # Step 1: Decompose query into subtasks
            try:
                subtask_strs = self.aop._decompose(query, agent_catalog)
            except Exception as e:
                print(f"  [SKIP] decompose failed: {e}")
                continue

            if not subtask_strs:
                print("  [SKIP] no subtasks generated")
                continue

            # Step 2: For each subtask, select top-l agents and evaluate
            for subtask in subtask_strs:
                # Pre-rank agents using TF-IDF
                try:
                    solv_result = estimator.estimate([subtask], agent_catalog)
                except Exception as e:
                    print(f"  [SKIP] solvability failed: {e}")
                    continue

                # Sort by combined score, take top-l candidates
                scores_sorted = sorted(
                    solv_result.scores,
                    key=lambda s: s.combined_score,
                    reverse=True,
                )
                candidates = [s.agent_id for s in scores_sorted[: self.l]]

                for agent_id in candidates:
                    agent = self.registry.get(agent_id)
                    if not agent:
                        continue

                    # Step 3: Execute agent on subtask
                    try:
                        result = agent.handle(
                            {"query": subtask, "text": subtask, "context": context}
                        )
                    except Exception as e:
                        print(f"  [SKIP] agent {agent_id} failed: {e}")
                        continue

                    if not result or result.get("error"):
                        continue

                    # Step 4: Score the response
                    try:
                        score = self.scorer.score(subtask, result)
                    except Exception as e:
                        print(f"  [SKIP] scoring failed: {e}")
                        continue

                    # Build agent description text (same as estimator)
                    agent_meta = agent_catalog.get(agent_id, {})
                    agent_desc = self._build_agent_text(agent_meta)

                    example = {
                        "subtask": subtask,
                        "agent_description": agent_desc,
                        "score": round(score, 4),
                        "query": query,
                        "agent_id": agent_id,
                    }
                    training_data.append(example)
                    print(
                        f"  ✓ {agent_id} → score={score:.3f} "
                        f"(subtask: {subtask[:50]}...)"
                    )

        # Save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(training_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[TrainGen] Saved {len(training_data)} examples to {output_path}")

        return training_data

    @staticmethod
    def _build_agent_text(agent_meta: Dict[str, Any]) -> str:
        """Concatenate agent metadata (matches SolvabilityEstimator)."""
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
