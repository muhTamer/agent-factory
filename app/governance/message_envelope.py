# app/governance/message_envelope.py
"""
Universal Message Format (UMF) — IEEE P3394 Alignment

Wraps existing ad-hoc Dict responses in a standardized message envelope
with required fields for inter-agent communication:
  - sender / receiver identity
  - message type and intent
  - conversation context
  - provenance and transparency metadata

The envelope is stored in the audit trail alongside the trace.
It does NOT replace the existing Dict response format — it wraps it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.runtime.trace import Trace


@dataclass
class AgentIdentity:
    """Identity block for an agent or user, per IEEE 3152-2024."""

    agent_id: str
    agent_type: str  # "faq_rag", "workflow_runner", "aop_coordinator", "user"
    is_human: bool = False
    version: str = "1.0"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "is_human": self.is_human,
            "version": self.version,
            "description": self.description,
        }


# Standard identity for the human user
USER_IDENTITY = AgentIdentity(
    agent_id="user",
    agent_type="human",
    is_human=True,
    version="n/a",
    description="End user (customer)",
)

# Standard identity for the meta-agent system
SYSTEM_IDENTITY = AgentIdentity(
    agent_id="meta_agent_factory",
    agent_type="orchestrator",
    is_human=False,
    version="1.0",
    description="Meta-Agent Factory orchestration system",
)


@dataclass
class MessageEnvelope:
    """
    Standardized message envelope aligned with IEEE P3394.

    Wraps any payload (query or response) with metadata required for
    auditability, transparency, and inter-agent protocol compliance.
    """

    message_id: str
    conversation_id: str  # thread_id
    timestamp_ms: int
    sender: AgentIdentity
    receiver: AgentIdentity
    message_type: str  # "query", "response", "delegation", "explanation"
    intent: str  # inferred or declared intent
    payload: Dict[str, Any]  # the actual message data
    provenance: Dict[str, Any] = field(default_factory=dict)
    ai_generated: bool = True
    # Agents involved in producing this message (for multi-agent responses)
    agents_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "timestamp_ms": self.timestamp_ms,
            "sender": self.sender.to_dict(),
            "receiver": self.receiver.to_dict(),
            "message_type": self.message_type,
            "intent": self.intent,
            "payload": self.payload,
            "provenance": self.provenance,
            "ai_generated": self.ai_generated,
            "agents_chain": self.agents_chain,
        }


def _extract_agents_chain(trace: Trace) -> List[str]:
    """Extract the sequence of agents involved from trace events."""
    agents: List[str] = []
    seen: set = set()
    for event in trace.events:
        data = event.data
        # Various events record agent IDs differently
        for key in ("agent_id", "primary", "selected_agent"):
            aid = data.get(key)
            if aid and aid not in seen:
                agents.append(aid)
                seen.add(aid)
        # AOP execution records multiple agents
        results = data.get("results")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    aid = r.get("agent")
                    if aid and aid not in seen:
                        agents.append(aid)
                        seen.add(aid)
        # Solvability assignments
        assignments = data.get("assignments")
        if isinstance(assignments, dict):
            for aid in assignments.values():
                if aid and aid not in seen:
                    agents.append(str(aid))
                    seen.add(str(aid))
    return agents


def _extract_provenance(trace: Trace, response: Dict[str, Any]) -> Dict[str, Any]:
    """Build provenance metadata from trace and response."""
    prov: Dict[str, Any] = {
        "request_id": trace.request_id,
        "trace_event_count": len(trace.events),
        "pipeline_stages": [e.stage for e in trace.events],
    }

    # Orchestration pattern
    pattern = response.get("orchestration_pattern")
    if pattern:
        prov["orchestration_pattern"] = pattern

    # Solvability scores
    solv = response.get("solvability")
    if isinstance(solv, dict):
        prov["solvability_scores"] = solv.get("assignment_scores", {})

    # Completeness
    comp = response.get("completeness")
    if isinstance(comp, dict):
        prov["completeness"] = {
            "complete": comp.get("complete"),
            "coverage_ratio": comp.get("coverage_ratio"),
        }

    # Router plan
    router_plan = response.get("router_plan")
    if isinstance(router_plan, dict):
        prov["routing"] = {
            "primary": router_plan.get("primary"),
            "strategy": router_plan.get("strategy"),
        }

    return prov


def wrap_response(
    response: Dict[str, Any],
    trace: Trace,
    context: Dict[str, Any],
) -> MessageEnvelope:
    """Wrap a spine response in a UMF-compliant MessageEnvelope.

    Args:
        response: The raw response dict from handle_chat.
        trace: The Trace object for this request.
        context: The request context (contains thread_id, intent, etc.).

    Returns:
        A fully populated MessageEnvelope.
    """
    # Determine the responding agent's identity
    agent_id = response.get("agent_id", "unknown")
    agent_type = "orchestrator"
    if response.get("orchestration_pattern") == "hierarchical_delegation":
        agent_type = "aop_coordinator"
    elif response.get("workflow_id"):
        agent_type = "workflow_runner"
    elif response.get("router_plan"):
        agent_type = "direct_router"

    sender = AgentIdentity(
        agent_id=agent_id,
        agent_type=agent_type,
        is_human=False,
    )

    intent = context.get("intent", "unknown")
    thread_id = str(context.get("thread_id", "default"))

    return MessageEnvelope(
        message_id=str(uuid.uuid4()),
        conversation_id=thread_id,
        timestamp_ms=int(time.time() * 1000),
        sender=sender,
        receiver=USER_IDENTITY,
        message_type="response",
        intent=intent,
        payload=response,
        provenance=_extract_provenance(trace, response),
        ai_generated=True,
        agents_chain=_extract_agents_chain(trace),
    )
