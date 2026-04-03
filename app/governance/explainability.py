# app/governance/explainability.py
"""
Multi-Level Explainability Engine — IEEE 2894-2024 Alignment

Generates explanations at three granularity levels from existing
runtime traces, WITHOUT any LLM calls:

  SUMMARY   — User-facing: plain language, no jargon, no internal IDs
  DETAILED  — Auditor: routing rationale, policies applied, governance checks
  FULL      — Developer: complete trace with react steps and raw data

Maps directly to IEEE 2894-2024 requirements for explanation completeness,
accuracy, and audience-appropriateness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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


# ── Human-readable labels ────────────────────────────────────────────

_DOMAIN_LABELS = {
    "refunds": "Refunds Specialist",
    "complaints": "Complaints Handler",
    "faq": "FAQ & Knowledge Base",
    "general": "General Assistant",
}

_PATTERN_LABELS = {
    "direct": "single-agent routing",
    "hierarchical_delegation": "multi-agent task decomposition",
    "fsm_workflow": "structured workflow",
    "aop_task_menu": "task planning",
    "aop_task_result": "task execution",
    "aop_plan_declined": "task plan (declined by user)",
}


def _agent_label(agent_id: str, response: Optional[Dict[str, Any]] = None) -> str:
    """Convert an internal agent_id to a human-readable label."""
    # If this is the primary agent, use the response's domain
    if response and response.get("agent_id") == agent_id:
        domain = response.get("domain", "")
        if domain and domain in _DOMAIN_LABELS:
            return _DOMAIN_LABELS[domain]
    # Try to extract domain from agent_id pattern like "refunds_agent_v1"
    for key, label in _DOMAIN_LABELS.items():
        if key in agent_id:
            return label
    return agent_id.replace("_", " ").replace("v1", "").strip().title()


def _primary_reason(response: Dict[str, Any]) -> Optional[str]:
    """Get the router's reason for selecting the primary agent."""
    plan = response.get("router_plan")
    if not isinstance(plan, dict):
        return None
    primary = plan.get("primary", "")
    for c in plan.get("candidates", []):
        if isinstance(c, dict) and c.get("id") == primary:
            return c.get("reason")
    return None


class ExplainabilityEngine:
    """Generate multi-level explanations from runtime traces."""

    def generate(
        self, trace: Trace, response: Dict[str, Any], level: ExplanationLevel
    ) -> Explanation:
        """Generate an explanation at the specified level."""
        if level == ExplanationLevel.SUMMARY:
            return self._summary(trace, response)
        elif level == ExplanationLevel.DETAILED:
            return self._detailed(trace, response)
        else:
            return self._full(trace, response)

    def generate_all_levels(
        self, trace: Trace, response: Dict[str, Any]
    ) -> Dict[str, Explanation]:
        """Generate explanations at all three levels."""
        return {
            level.value: self.generate(trace, response, level)
            for level in ExplanationLevel
        }

    # ── Level 1: Summary (user-facing, IEEE 2894-R2) ─────────────────

    def _summary(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """User-facing: plain language, no technical jargon or internal IDs."""
        agent_id = response.get("agent_id", "")
        label = _agent_label(agent_id, response)
        pattern = response.get("orchestration_pattern", "direct")
        needs_input = response.get("needs_input", False)

        parts: List[str] = []

        # What happened
        if pattern == "hierarchical_delegation":
            subtasks = response.get("subtask_results", [])
            parts.append(
                f"Your request required multiple steps. The system broke it into "
                f"{len(subtasks)} part(s) and assigned each to a specialist."
            )
        elif pattern == "fsm_workflow":
            state = response.get("current_state", "processing")
            parts.append(
                f"Your request is being handled through a step-by-step process. "
                f"Current status: {state}."
            )
        else:
            parts.append(f"Your request was handled by our {label}.")

        # Why this agent
        reason = _primary_reason(response)
        if reason:
            # Simplify the router reason for users — take first sentence
            first_sentence = reason.split(".")[0].strip()
            # Remove technical prefixes like "Best match for an ACTIONABLE..."
            for prefix in ("Best match for ", "Strong intent fit", "High intent fit"):
                if first_sentence.startswith(prefix):
                    first_sentence = first_sentence[len(prefix) :].lstrip(":— ")
                    break
            if first_sentence:
                parts.append(f"This specialist was selected because: {first_sentence}.")

        # What the agent did
        if needs_input:
            parts.append(
                "The system needs additional information from you before it can proceed."
            )
        elif response.get("knowledge_retrieved"):
            parts.append("The answer was found by searching our knowledge base.")

        # Policies (simplified)
        policies = response.get("policies_applied", [])
        if policies:
            parts.append(
                f"The system followed {len(policies)} internal policy rule(s) "
                f"while processing your request."
            )

        # AI disclosure (IEEE 3152-R1)
        parts.append("This response was generated by an AI system.")

        return Explanation(
            level=ExplanationLevel.SUMMARY,
            narrative=" ".join(parts),
            agents_involved=[agent_id] if agent_id else [],
            metrics={"response_time_ms": self._total_latency(trace)},
        )

    # ── Level 2: Detailed (auditor-facing, IEEE 2894-R3) ─────────────

    def _detailed(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """Auditor-facing: routing decisions, policies, governance checks."""
        agent_id = response.get("agent_id", "")
        label = _agent_label(agent_id, response)
        agents = self._extract_agents(trace, response)
        decisions = self._extract_decisions(trace, response)
        provenance = self._extract_provenance(trace, response)

        parts: List[str] = []

        # 1. Routing decision with rationale (IEEE 2894-R5)
        plan = response.get("router_plan")
        if isinstance(plan, dict):
            primary = plan.get("primary", "?")
            strategy = plan.get("strategy", "single")
            score = response.get("score")
            parts.append(
                f"ROUTING: The system evaluated {len(plan.get('candidates', []))} "
                f"candidate agent(s) using strategy '{strategy}'. "
                f"Selected: {_agent_label(primary, response)} "
                f"(confidence: {score:.0%})."
                if score is not None
                else f"ROUTING: Selected {_agent_label(primary, response)} "
                f"using '{strategy}' strategy."
            )
            # Include runner-up for audit context
            candidates = plan.get("candidates", [])
            if len(candidates) > 1:
                runner_up = candidates[1] if isinstance(candidates[1], dict) else {}
                ru_id = runner_up.get("id", "?")
                ru_score = runner_up.get("score", 0)
                parts.append(
                    f"Runner-up: {_agent_label(ru_id, response)} "
                    f"(score: {ru_score:.0%}). "
                    f"Reason for primary selection: {candidates[0].get('reason', 'N/A')}"
                    if isinstance(candidates[0], dict)
                    else ""
                )

        # 2. Policies applied (IEEE 2894-R5 rationale)
        policies = response.get("policies_applied", [])
        if policies:
            parts.append(f"POLICIES APPLIED ({len(policies)}):")
            for i, pol in enumerate(policies[:5], 1):  # Cap at 5 for readability
                pol_text = pol if isinstance(pol, str) else str(pol)
                parts.append(f"  {i}. {pol_text[:150]}")
            if len(policies) > 5:
                parts.append(f"  ... and {len(policies) - 5} more.")

        # 3. Knowledge and data sources (IEEE 2894-R4 provenance)
        ks = response.get("knowledge_sources", [])
        ps = response.get("policy_sources", [])
        if ks:
            parts.append(f"KNOWLEDGE SOURCES: {len(ks)} source(s) consulted.")
            for src in ks[:3]:
                if isinstance(src, dict):
                    parts.append(f"  - {src.get('name', src.get('source', 'unknown'))}")
        if ps:
            parts.append(f"POLICY SOURCES: {len(ps)} policy document(s) referenced.")
            for src in ps[:3]:
                if isinstance(src, dict):
                    parts.append(f"  - {src.get('name', 'unnamed policy')}")

        # 4. Tools used
        tools = response.get("tools_used", [])
        if tools:
            parts.append(f"TOOLS INVOKED: {', '.join(tools)}.")

        # 5. Guardrail activity
        guard_events = [
            e
            for e in trace.events
            if e.stage
            in (
                "guard_pre_ok",
                "guard_post_ok",
                "guard_pre_block",
                "guard_post_block",
            )
        ]
        if guard_events:
            blocks = sum(1 for e in guard_events if "block" in e.stage)
            parts.append(
                f"GOVERNANCE: {len(guard_events)} guardrail check(s) executed, "
                f"{blocks} intervention(s)."
            )

        # 6. AI disclosure (IEEE 3152-R1/R2)
        parts.append(
            f"AI DISCLOSURE: This response was generated by an AI system. "
            f"Processing agent: {label} ({agent_id}). "
            f"Total agents involved: {len(agents)}."
        )

        return Explanation(
            level=ExplanationLevel.DETAILED,
            narrative="\n".join(parts),
            agents_involved=agents,
            decisions=decisions,
            provenance=provenance,
            metrics=self._extract_metrics(trace, response),
        )

    # ── Level 3: Full (developer-facing) ─────────────────────────────

    def _full(self, trace: Trace, response: Dict[str, Any]) -> Explanation:
        """Developer-facing: complete trace with react steps and raw data."""
        agents = self._extract_agents(trace, response)
        decisions = self._extract_decisions(trace, response)
        provenance = self._extract_provenance(trace, response)

        parts: List[str] = []

        # Header
        parts.append(f"Request ID: {trace.request_id}")
        parts.append(f"Query: {trace.query!r}")
        parts.append(f"Agent: {response.get('agent_id', 'N/A')}")
        parts.append(f"Domain: {response.get('domain', 'N/A')}")
        parts.append(f"Intent: {response.get('intent', 'N/A')}")
        parts.append(f"Score: {response.get('score', 'N/A')}")
        parts.append(f"Wall time: {self._total_latency(trace)} ms")
        parts.append("")

        # Router plan (full detail)
        plan = response.get("router_plan")
        if isinstance(plan, dict):
            parts.append("=== ROUTER PLAN ===")
            parts.append(f"Strategy: {plan.get('strategy')}")
            parts.append(f"Primary: {plan.get('primary')}")
            for c in plan.get("candidates", []):
                if isinstance(c, dict):
                    parts.append(
                        f"  [{c.get('id')}] score={c.get('score')} "
                        f"reason={c.get('reason', '')[:200]}"
                    )
            parts.append("")

        # React trace (agent's internal reasoning)
        react = response.get("react_trace")
        if react:
            parts.append("=== AGENT REASONING (react_trace) ===")
            if isinstance(react, str):
                try:
                    react = json.loads(react)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(react, list):
                for step in react:
                    if isinstance(step, dict):
                        parts.append(
                            f"Step {step.get('step', '?')}: "
                            f"{step.get('thought', step.get('action', ''))[:300]}"
                        )
            else:
                parts.append(str(react)[:500])
            parts.append("")

        # Policies
        policies = response.get("policies_applied", [])
        if policies:
            parts.append("=== POLICIES APPLIED ===")
            for pol in policies:
                parts.append(f"  - {pol}")
            parts.append("")

        # Tools
        tools = response.get("tools_used", [])
        if tools:
            parts.append(f"=== TOOLS USED === {', '.join(tools)}")
            parts.append("")

        # Slots / extracted entities
        slots = response.get("slots", {})
        if slots:
            parts.append("=== EXTRACTED SLOTS ===")
            for k, v in slots.items():
                parts.append(f"  {k}: {v}")
            parts.append("")

        # Trace events
        if trace.events:
            parts.append("=== TRACE EVENTS ===")
            base_ts = trace.started_ts_ms
            for event in trace.events:
                delta = event.ts_ms - base_ts
                data_str = (
                    ", ".join(f"{k}={v!r}" for k, v in event.data.items())
                    if event.data
                    else ""
                )
                parts.append(f"  +{delta:>6d}ms  [{event.stage}]  {data_str}")

        narrative = "\n".join(parts)

        metrics = self._extract_metrics(trace, response)
        metrics["event_log"] = [
            {
                "stage": event.stage,
                "ts_ms": event.ts_ms,
                "delta_ms": event.ts_ms - trace.started_ts_ms,
                **event.data,
            }
            for event in trace.events
        ]

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
    def _extract_decisions(
        trace: Trace, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
    def _extract_provenance(
        trace: Trace, response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
                # Domain agent knowledge sources
                ks = result.get("knowledge_sources")
                if isinstance(ks, list) and ks:
                    prov.append(
                        {
                            "source": f"agent:{st.get('agent_id', '?')}",
                            "type": "domain_agent_knowledge",
                            "knowledge_sources": ks,
                        }
                    )

        # Direct-route domain agent provenance
        ks = response.get("knowledge_sources")
        if isinstance(ks, list) and ks:
            prov.append(
                {
                    "source": f"agent:{response.get('agent_id', '?')}",
                    "type": "domain_agent_knowledge",
                    "knowledge_sources": ks,
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
            metrics["subtask_success_rate"] = sum(
                1 for s in subtasks if s.get("success")
            ) / len(subtasks)

        return metrics

    @staticmethod
    def _total_latency(trace: Trace) -> int:
        """Total wall-clock time from trace."""
        if not trace.events:
            return 0
        return trace.events[-1].ts_ms - trace.started_ts_ms
