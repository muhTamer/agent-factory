# app/governance/explainability.py
"""
Multi-Level Explainability Engine — IEEE 2894-2024 Alignment

Generates explanations at three granularity levels from existing
runtime traces, WITHOUT any LLM calls:

  SUMMARY   — User-facing: "Your question was answered using our FAQ system"
  DETAILED  — Auditor: agents involved, decision scores, policy citations
  FULL      — Developer: complete trace with timing and raw data

Maps directly to IEEE 2894-2024 requirements for explanation completeness,
accuracy, and audience-appropriateness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from app.runtime.trace import Trace


class ExplanationLevel(str, Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    FULL = "full"


@dataclass
class Explanation:
    """A single explanation at a specific granularity level."""

    level: ExplanationLevel
    narrative: str  # Human-readable explanation text
    agents_involved: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "narrative": self.narrative,
            "agents_involved": self.agents_involved,
            "decisions": self.decisions,
            "provenance": self.provenance,
            "metrics": self.metrics,
        }


# ── Pattern labels for user-facing narratives ────────────────────────

_PATTERN_LABELS = {
    "direct": "direct agent routing",
    "hierarchical_delegation": "multi-agent task decomposition (AOP)",
    "fsm_workflow": "structured workflow processing",
    "aop_task_menu": "task planning and selection",
    "aop_task_result": "task execution",
    "aop_plan_declined": "task plan declined by user",
}

_AGENT_TYPE_LABELS = {
    "faq_rag": "FAQ knowledge base",
    "workflow_runner": "workflow processor",
    "aop_coordinator": "task planning coordinator",
    "tool_operator": "system tool",
    "rag_fsm": "knowledge retrieval system",
}


class ExplainabilityEngine:
    """Generate multi-level explanations from runtime traces."""

    def generate(
        self, trace: Trace, response: Dict[str, Any], level: ExplanationLevel
    ) -> Explanation:
        """Generate an explanation at the specified level."""
        if level == ExplanationLevel.SUMMARY:
            return self._summary_from_trace(trace, response)
        elif level == ExplanationLevel.DETAILED:
            return self._detailed_from_trace(trace, response)
        else:
            return self._full_from_trace(trace, response)

    def generate_all_levels(self, trace: Trace, response: Dict[str, Any]) -> Dict[str, Explanation]:
        """Generate explanations at all three levels."""
        return {level.value: self.generate(trace, response, level) for level in ExplanationLevel}

    # ── Level 1: Summary (user-facing) ───────────────────────────────

    def _summary_from_trace(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """User-facing explanation: what happened in plain language."""
        pattern = response.get("orchestration_pattern", "direct")
        pattern_label = _PATTERN_LABELS.get(pattern, pattern)
        agents = self._extract_agents(trace, response)
        agent_labels = [_AGENT_TYPE_LABELS.get(a, a) for a in agents]

        # Build narrative
        if pattern == "hierarchical_delegation":
            subtask_count = len(response.get("subtask_results", []))
            narrative = (
                f"Your query was broken into {subtask_count} part(s) and "
                f"handled by {len(agents)} specialist agent(s) using {pattern_label}."
            )
        elif pattern == "fsm_workflow":
            state = response.get("current_state", "processing")
            narrative = (
                f"Your request is being processed through a {pattern_label}. "
                f"Current status: {state}."
            )
        elif pattern in ("aop_task_menu", "aop_task_result"):
            narrative = f"The system used {pattern_label} to address your request."
        else:
            if agent_labels:
                narrative = f"Your question was answered using our {agent_labels[0]}."
            else:
                narrative = "Your query was processed by the system."

        return Explanation(
            level=ExplanationLevel.SUMMARY,
            narrative=narrative,
            agents_involved=agents,
            metrics={"response_time_ms": self._total_latency(trace)},
        )

    # ── Level 2: Detailed (auditor-facing) ───────────────────────────

    def _detailed_from_trace(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """Auditor-facing explanation: decisions, scores, policies."""
        agents = self._extract_agents(trace, response)
        decisions = self._extract_decisions(trace, response)
        provenance = self._extract_provenance(trace, response)

        pattern = response.get("orchestration_pattern", "direct")
        parts = [f"Orchestration pattern: {pattern}."]

        # Routing decision
        for d in decisions:
            if d["stage"] == "route":
                parts.append(
                    f"Router selected '{d.get('primary', '?')}' with "
                    f"strategy '{d.get('strategy', '?')}'."
                )
            elif d["stage"] == "aop_solvability":
                assignments = d.get("assignments", {})
                parts.append(f"AOP assigned {len(assignments)} subtask(s) to agents.")
            elif d["stage"] == "aop_completeness":
                complete = d.get("complete", False)
                parts.append(f"Completeness check: {'passed' if complete else 'gaps detected'}.")

        # Guardrail activity
        guard_events = [
            e
            for e in trace.events
            if e.stage in ("guard_pre_ok", "guard_post_ok", "guard_pre_block", "guard_post_block")
        ]
        if guard_events:
            blocks = sum(1 for e in guard_events if "block" in e.stage)
            parts.append(
                f"Guardrails evaluated: {len(guard_events)} check(s), " f"{blocks} intervention(s)."
            )

        # Scores
        score = response.get("score")
        if score is not None:
            parts.append(f"Confidence score: {score:.2f}.")

        return Explanation(
            level=ExplanationLevel.DETAILED,
            narrative=" ".join(parts),
            agents_involved=agents,
            decisions=decisions,
            provenance=provenance,
            metrics=self._extract_metrics(trace, response),
        )

    # ── Level 3: Full (developer-facing) ─────────────────────────────

    def _full_from_trace(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """Developer-facing explanation: complete trace dump."""
        agents = self._extract_agents(trace, response)
        decisions = self._extract_decisions(trace, response)
        provenance = self._extract_provenance(trace, response)

        # Full event log
        events = []
        base_ts = trace.started_ts_ms
        for event in trace.events:
            events.append(
                {
                    "delta_ms": event.ts_ms - base_ts,
                    "stage": event.stage,
                    "data": event.data,
                }
            )

        narrative = (
            f"Full execution trace for request {trace.request_id}. "
            f"{len(trace.events)} events over {self._total_latency(trace)} ms. "
            f"Agents: {', '.join(agents) if agents else 'none'}."
        )

        metrics = self._extract_metrics(trace, response)
        metrics["event_log"] = events

        return Explanation(
            level=ExplanationLevel.FULL,
            narrative=narrative,
            agents_involved=agents,
            decisions=decisions,
            provenance=provenance,
            metrics=metrics,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_agents(trace: Trace, response: Dict[str, Any]) -> List[str]:
        """Extract unique agent IDs from trace and response."""
        agents: List[str] = []
        seen: set = set()

        # From response
        aid = response.get("agent_id")
        if aid and aid not in seen:
            agents.append(aid)
            seen.add(aid)

        # From subtask results (AOP)
        for st in response.get("subtask_results", []):
            aid = st.get("agent_id")
            if aid and aid not in seen:
                agents.append(aid)
                seen.add(aid)

        # From trace events
        for event in trace.events:
            for key in ("agent_id", "primary", "selected_agent"):
                aid = event.data.get(key)
                if aid and aid not in seen:
                    agents.append(aid)
                    seen.add(aid)
            # AOP execution results
            for r in event.data.get("results", []):
                if isinstance(r, dict):
                    aid = r.get("agent")
                    if aid and aid not in seen:
                        agents.append(aid)
                        seen.add(aid)

        return agents

    @staticmethod
    def _extract_decisions(trace: Trace, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key decision points from trace events."""
        decision_stages = {
            "orchestration_pattern",
            "route",
            "intent_inferred",
            "aop_decompose",
            "aop_solvability",
            "aop_completeness",
            "aop_redecompose",
            "aop_task_selected",
            "guard_pre_ok",
            "guard_pre_block",
            "guard_post_ok",
            "guard_post_block",
            "select",
            "rag_delegation",
        }
        decisions = []
        for event in trace.events:
            if event.stage in decision_stages:
                decisions.append(
                    {
                        "stage": event.stage,
                        **event.data,
                    }
                )
        return decisions

    @staticmethod
    def _extract_provenance(trace: Trace, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data provenance: sources, citations, policies used."""
        prov: List[Dict[str, Any]] = []

        # Router plan provenance
        router_plan = response.get("router_plan")
        if isinstance(router_plan, dict):
            prov.append(
                {
                    "source": "router",
                    "strategy": router_plan.get("strategy"),
                    "primary_agent": router_plan.get("primary"),
                    "candidates": router_plan.get("candidates", []),
                }
            )

        # Solvability provenance
        solv = response.get("solvability")
        if isinstance(solv, dict):
            prov.append(
                {
                    "source": "solvability_estimator",
                    "assignments": solv.get("assignments", {}),
                    "scores": solv.get("assignment_scores", {}),
                }
            )

        # Completeness provenance
        comp = response.get("completeness")
        if isinstance(comp, dict):
            prov.append(
                {
                    "source": "completeness_detector",
                    "complete": comp.get("complete"),
                    "coverage_ratio": comp.get("coverage_ratio"),
                    "reasoning": comp.get("reasoning", ""),
                }
            )

        # Subtask-level provenance (AOP)
        for st in response.get("subtask_results", []):
            result = st.get("result")
            if isinstance(result, dict):
                # RAG citations
                citations = result.get("grounded_citations") or result.get("citations")
                if citations:
                    prov.append(
                        {
                            "source": f"agent:{st.get('agent_id', '?')}",
                            "citations": citations,
                        }
                    )

        return prov

    @staticmethod
    def _extract_metrics(trace: Trace, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantitative metrics from trace and response."""
        base_ts = trace.started_ts_ms
        total_ms = 0
        if trace.events:
            total_ms = trace.events[-1].ts_ms - base_ts

        metrics: Dict[str, Any] = {
            "total_latency_ms": total_ms,
            "event_count": len(trace.events),
        }

        score = response.get("score")
        if score is not None:
            metrics["confidence_score"] = score

        solv = response.get("solvability")
        if isinstance(solv, dict):
            scores = solv.get("assignment_scores", {})
            if scores:
                metrics["mean_solvability"] = sum(scores.values()) / len(scores)

        comp = response.get("completeness")
        if isinstance(comp, dict):
            metrics["coverage_ratio"] = comp.get("coverage_ratio")

        subtasks = response.get("subtask_results", [])
        if subtasks:
            metrics["subtask_count"] = len(subtasks)
            metrics["subtask_success_rate"] = sum(1 for s in subtasks if s.get("success")) / len(
                subtasks
            )

        return metrics

    @staticmethod
    def _total_latency(trace: Trace) -> int:
        """Total wall-clock time from trace."""
        if not trace.events:
            return 0
        return trace.events[-1].ts_ms - trace.started_ts_ms
