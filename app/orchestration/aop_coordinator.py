# app/orchestration/aop_coordinator.py
"""
AOP Coordinator — Meta-Agent for Agent-Oriented Planning (Li et al. 2024)

Implements the 5-step control loop described in the thesis Theory chapter:
  1. Task decomposition       — LLM breaks query into atomic subtasks
  2. Agent selection           — SolvabilityEstimator scores (subtask, agent) pairs
  3. Completeness check        — CompletenessDetector audits plan coverage
  4. Execution                 — Delegate each subtask to its assigned agent
  5. Feedback loop             — Record results in PerformanceStore

Integration with RuntimeSpine:
  - Called when orchestration pattern == "hierarchical_delegation"
  - Returns Dict[str, Any] compatible with spine's _respond() format
  - Results flow through voice rendering and post-guardrails normally
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm_client import chat_json
from app.orchestration.completeness_detector import CompletenessDetector, CompletenessResult
from app.orchestration.performance_store import ExecutionRecord, PerformanceStore
from app.orchestration.solvability_estimator import SolvabilityEstimator, SolvabilityResult
from app.runtime.registry import AgentRegistry
from app.runtime.trace import Trace


@dataclass
class Subtask:
    """A decomposed subtask from the original query."""

    description: str
    assigned_agent_id: Optional[str] = None
    solvability_score: float = 0.0
    result: Optional[Dict[str, Any]] = None
    success: bool = False
    latency_ms: int = 0


@dataclass
class AOPResult:
    """Complete result of AOP orchestration cycle."""

    query: str
    subtasks: List[Subtask] = field(default_factory=list)
    completeness: Optional[CompletenessResult] = None
    solvability: Optional[SolvabilityResult] = None
    composite_response: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0
    orchestration_pattern: str = "hierarchical_delegation"


class AOPCoordinator:
    """
    Meta-agent implementing the 5-step Agent-Oriented Planning cycle.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        performance_store: PerformanceStore,
        estimator: Optional[SolvabilityEstimator] = None,
        completeness: Optional[CompletenessDetector] = None,
        model: str = "gpt-5-mini",
        max_retries: int = 1,
    ):
        self.registry = registry
        self.store = performance_store
        self.estimator = estimator or SolvabilityEstimator(performance_store)
        self.completeness = completeness or CompletenessDetector(model=model)
        self.model = model
        self.max_retries = max_retries

    def orchestrate(
        self,
        query: str,
        context: Dict[str, Any],
        trace: Optional[Trace] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full 5-step AOP cycle.

        Returns dict compatible with spine's response format.
        """
        start_ms = _now_ms()
        agent_catalog = self.registry.all_meta()

        if not agent_catalog:
            return {
                "error": "No agents available for delegation.",
                "orchestration_pattern": "hierarchical_delegation",
            }

        # ── Step 1: Task Decomposition ──
        subtask_strs = self._decompose(query, agent_catalog)
        if trace:
            trace.add("aop_decompose", subtasks=subtask_strs)

        if not subtask_strs:
            return {
                "error": "Failed to decompose query into subtasks.",
                "orchestration_pattern": "hierarchical_delegation",
            }

        # ── Step 2: Agent Selection (Solvability) ──
        solv_result = self._select_agents(subtask_strs, agent_catalog)
        if trace:
            trace.add("aop_solvability", assignments=solv_result.assignments)

        # Build Subtask objects
        subtasks = []
        for st_str in subtask_strs:
            st = Subtask(
                description=st_str,
                assigned_agent_id=solv_result.assignments.get(st_str),
                solvability_score=solv_result.assignment_scores.get(st_str, 0.0),
            )
            subtasks.append(st)

        # ── Step 3: Completeness Check ──
        comp_result = self._check_completeness(query, subtask_strs, solv_result.assignments)
        if trace:
            trace.add(
                "aop_completeness",
                complete=comp_result.complete,
                missing=comp_result.missing,
            )

        # If incomplete and retries remain, re-decompose with hints
        if not comp_result.complete and self.max_retries > 0:
            subtask_strs = self._re_decompose(query, agent_catalog, comp_result.missing)
            if subtask_strs:
                solv_result = self._select_agents(subtask_strs, agent_catalog)
                subtasks = [
                    Subtask(
                        description=st,
                        assigned_agent_id=solv_result.assignments.get(st),
                        solvability_score=solv_result.assignment_scores.get(st, 0.0),
                    )
                    for st in subtask_strs
                ]
                comp_result = self._check_completeness(query, subtask_strs, solv_result.assignments)
                if trace:
                    trace.add(
                        "aop_redecompose", subtasks=subtask_strs, complete=comp_result.complete
                    )

        # ── Step 4: Execution ──
        subtasks = self._execute_subtasks(subtasks, context)
        if trace:
            trace.add(
                "aop_execute",
                results=[
                    {
                        "subtask": st.description,
                        "agent": st.assigned_agent_id,
                        "success": st.success,
                    }
                    for st in subtasks
                ],
            )

        # ── Step 5: Feedback Loop ──
        self._record_feedback(subtasks)

        total_ms = _now_ms() - start_ms

        # ── Assemble Response ──
        return self._assemble_composite_response(
            query, subtasks, comp_result, solv_result, total_ms
        )

    # ── Step 1: Task Decomposition ──────────────────────────────────

    def _decompose(self, query: str, agent_catalog: Dict[str, Dict[str, Any]]) -> List[str]:
        """Use LLM to decompose a multi-intent query into atomic subtasks."""
        catalog_summary = []
        for aid, meta in agent_catalog.items():
            caps = meta.get("capabilities", [])
            desc = meta.get("description", "")
            catalog_summary.append(f"  - {aid}: {desc} (capabilities: {', '.join(caps)})")
        catalog_str = "\n".join(catalog_summary)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task decomposition module for an AOP multi-agent system.\n"
                    "Given a user query and available agents, break the query into atomic subtasks.\n"
                    "Each subtask should be a single, independent unit of work that one agent can handle.\n\n"
                    "Rules:\n"
                    "- Each subtask must be self-contained and actionable.\n"
                    "- Do NOT create subtasks for things no agent can handle.\n"
                    "- Return 1-5 subtasks (prefer fewer).\n\n"
                    'Return STRICT JSON: {"subtasks": ["subtask description 1", ...]}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nAvailable agents:\n{catalog_str}",
            },
        ]

        try:
            raw = chat_json(messages=messages, model=self.model, temperature=1.0)
            subtasks = raw.get("subtasks", [])
            if isinstance(subtasks, list):
                return [str(s).strip() for s in subtasks if str(s).strip()]
            return []
        except Exception as e:
            print(f"[AOP] decompose failed: {e}")
            return []

    def _re_decompose(
        self,
        query: str,
        agent_catalog: Dict[str, Dict[str, Any]],
        missing_aspects: List[str],
    ) -> List[str]:
        """Re-decompose with hints about missing aspects."""
        catalog_summary = []
        for aid, meta in agent_catalog.items():
            caps = meta.get("capabilities", [])
            desc = meta.get("description", "")
            catalog_summary.append(f"  - {aid}: {desc} (capabilities: {', '.join(caps)})")
        catalog_str = "\n".join(catalog_summary)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task decomposition module. A previous decomposition was incomplete.\n"
                    "Re-decompose the query, making sure to address the missing aspects.\n\n"
                    'Return STRICT JSON: {"subtasks": ["subtask description 1", ...]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Available agents:\n{catalog_str}\n\n"
                    f"Missing aspects from previous attempt:\n"
                    + "\n".join(f"  - {m}" for m in missing_aspects)
                ),
            },
        ]

        try:
            raw = chat_json(messages=messages, model=self.model, temperature=1.0)
            subtasks = raw.get("subtasks", [])
            if isinstance(subtasks, list):
                return [str(s).strip() for s in subtasks if str(s).strip()]
            return []
        except Exception:
            return []

    # ── Step 2: Agent Selection ─────────────────────────────────────

    def _select_agents(
        self, subtasks: List[str], agent_catalog: Dict[str, Dict[str, Any]]
    ) -> SolvabilityResult:
        """Score all (subtask, agent) pairs via SolvabilityEstimator."""
        return self.estimator.estimate(subtasks, agent_catalog)

    # ── Step 3: Completeness Check ──────────────────────────────────

    def _check_completeness(
        self,
        query: str,
        subtasks: List[str],
        assignments: Dict[str, str],
    ) -> CompletenessResult:
        """Audit plan for completeness and non-redundancy."""
        return self.completeness.check(query, subtasks, assignments)

    # ── Step 4: Execution ───────────────────────────────────────────

    def _execute_subtasks(
        self,
        subtasks: List[Subtask],
        context: Dict[str, Any],
    ) -> List[Subtask]:
        """Execute each subtask by delegating to its assigned agent."""
        for st in subtasks:
            if not st.assigned_agent_id:
                st.success = False
                st.result = {"error": "No agent assigned"}
                continue

            agent = self.registry.get(st.assigned_agent_id)
            if not agent:
                st.success = False
                st.result = {"error": f"Agent {st.assigned_agent_id} not found in registry"}
                continue

            t0 = _now_ms()
            try:
                result = agent.handle(
                    {"query": st.description, "text": st.description, "context": context}
                )
                st.result = result
                st.success = not result.get("error")
                # Use agent-reported score if available
                try:
                    st.solvability_score = float(result.get("score", st.solvability_score))
                except (TypeError, ValueError):
                    pass
            except Exception as e:
                st.result = {"error": str(e)}
                st.success = False

            st.latency_ms = _now_ms() - t0

        return subtasks

    # ── Step 5: Feedback Loop ───────────────────────────────────────

    def _record_feedback(self, subtasks: List[Subtask]) -> None:
        """Write execution results to performance store."""
        for st in subtasks:
            if not st.assigned_agent_id:
                continue
            try:
                self.store.append(
                    ExecutionRecord(
                        agent_id=st.assigned_agent_id,
                        subtask=st.description,
                        success=st.success,
                        score=st.solvability_score,
                        latency_ms=st.latency_ms,
                    )
                )
            except Exception as e:
                print(f"[AOP] feedback write failed for {st.assigned_agent_id}: {e}")

    # ── Response Assembly ───────────────────────────────────────────

    @staticmethod
    def _extract_readable_text(result: Dict[str, Any]) -> str:
        """
        Extract a human-readable string from an agent result dict.

        Agents return varied formats:
          - FAQ/RAG: {"answer": "...", "score": ...}
          - Workflow: {"message": "...", "current_state": ..., "slots": ...}
          - Tool: {"text": "...", "tool_result": ...}
          - Generic: {"response": "..."}
        """
        if not result:
            return ""

        # Direct text fields (preferred)
        for key in ("text", "answer", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Workflow-style: build a readable summary
        if "current_state" in result or "workflow_id" in result:
            parts = []
            if result.get("message"):
                parts.append(str(result["message"]))
            if result.get("action"):
                parts.append(f"Action: {result['action']}")
            if result.get("missing_slots"):
                missing = result["missing_slots"]
                if isinstance(missing, list):
                    parts.append(f"Need more info: {', '.join(str(s) for s in missing)}")
            if result.get("current_state"):
                parts.append(f"(State: {result['current_state']})")
            if parts:
                return " | ".join(parts)

        # Nested result dict
        if isinstance(result.get("result"), dict):
            return AOPCoordinator._extract_readable_text(result["result"])

        # Last resort: skip internal keys, show only short values
        skip = {
            "slots",
            "history",
            "mapper",
            "allowed_events",
            "slot_defs",
            "context",
            "thread_id",
            "workflow_id",
            "request_id",
            "router_plan",
        }
        summary_parts = []
        for k, v in result.items():
            if k in skip:
                continue
            s = str(v)
            if len(s) < 200:
                summary_parts.append(f"{k}: {s}")
        return " | ".join(summary_parts[:5]) if summary_parts else "(no readable content)"

    def _assemble_composite_response(
        self,
        query: str,
        subtasks: List[Subtask],
        completeness: CompletenessResult,
        solvability: SolvabilityResult,
        total_latency_ms: int,
    ) -> Dict[str, Any]:
        """Combine subtask results into a spine-compatible response dict."""
        # Collect individual answers
        answers = []
        for st in subtasks:
            if st.result and not st.result.get("error"):
                text = self._extract_readable_text(st.result)
                answers.append(f"[{st.assigned_agent_id}] {text}")
            else:
                err = st.result.get("error", "unknown error") if st.result else "no result"
                answers.append(f"[{st.assigned_agent_id}] Unable to complete: {err}")

        combined_text = "\n\n".join(answers)

        # Average score across successful subtasks
        successful = [st for st in subtasks if st.success]
        avg_score = (
            sum(st.solvability_score for st in successful) / len(successful) if successful else 0.0
        )

        return {
            "text": combined_text,
            "answer": combined_text,
            "score": avg_score,
            "orchestration_pattern": "hierarchical_delegation",
            "subtask_results": [
                {
                    "subtask": st.description,
                    "agent_id": st.assigned_agent_id,
                    "success": st.success,
                    "solvability_score": st.solvability_score,
                    "latency_ms": st.latency_ms,
                    "result": st.result,
                }
                for st in subtasks
            ],
            "completeness": {
                "complete": completeness.complete,
                "missing": completeness.missing,
                "coverage_ratio": completeness.coverage_ratio,
                "reasoning": completeness.reasoning,
            },
            "solvability": {
                "assignments": solvability.assignments,
                "assignment_scores": {
                    k: round(v, 4) for k, v in solvability.assignment_scores.items()
                },
            },
            "total_latency_ms": total_latency_ms,
        }


def _now_ms() -> int:
    return int(time.time() * 1000)
