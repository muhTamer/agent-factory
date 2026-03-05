# app/governance/ieee_compliance.py
"""
IEEE Standards Compliance Checker — RQ2 Core Module

Defines requirements for each IEEE standard referenced in the thesis
and checks whether messages, traces, and responses comply:

  IEEE P3394      — Universal Message Format for multi-agent systems
  IEEE 2894-2024  — Guide for Explainable AI
  IEEE 3152-2024  — Transparent Human/Machine Agency

Produces a ComplianceReport with per-requirement pass/fail, evidence,
and an aggregate compliance rate.  No LLM calls — purely structural checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.runtime.trace import Trace


# ── Requirement definitions ──────────────────────────────────────────


@dataclass(frozen=True)
class IEEERequirement:
    """A single auditable requirement from an IEEE standard."""

    standard: str  # "P3394", "2894-2024", "3152-2024"
    requirement_id: str  # e.g. "P3394-R1"
    description: str
    category: str  # "message_format", "explainability", "transparency"
    severity: str  # "must", "should", "may"


@dataclass
class ComplianceResult:
    """Result of checking one requirement."""

    requirement: IEEERequirement
    compliant: bool
    evidence: str = ""  # what was found
    gap: str = ""  # what's missing (if non-compliant)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "standard": self.requirement.standard,
            "requirement_id": self.requirement.requirement_id,
            "description": self.requirement.description,
            "category": self.requirement.category,
            "severity": self.requirement.severity,
            "compliant": self.compliant,
            "evidence": self.evidence,
            "gap": self.gap,
        }


@dataclass
class ComplianceReport:
    """Aggregate compliance report across all checked requirements."""

    results: List[ComplianceResult] = field(default_factory=list)

    @property
    def compliance_rate(self) -> float:
        """Overall compliance rate (0.0 - 1.0)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.compliant) / len(self.results)

    @property
    def by_standard(self) -> Dict[str, float]:
        """Compliance rate per IEEE standard."""
        standards: Dict[str, List[bool]] = {}
        for r in self.results:
            standards.setdefault(r.requirement.standard, []).append(r.compliant)
        return {std: sum(vals) / len(vals) if vals else 0.0 for std, vals in standards.items()}

    @property
    def by_severity(self) -> Dict[str, float]:
        """Compliance rate by severity level."""
        severities: Dict[str, List[bool]] = {}
        for r in self.results:
            severities.setdefault(r.requirement.severity, []).append(r.compliant)
        return {sev: sum(vals) / len(vals) if vals else 0.0 for sev, vals in severities.items()}

    def non_compliant(self) -> List[ComplianceResult]:
        """Return only non-compliant results."""
        return [r for r in self.results if not r.compliant]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compliance_rate": round(self.compliance_rate, 4),
            "by_standard": {k: round(v, 4) for k, v in self.by_standard.items()},
            "by_severity": {k: round(v, 4) for k, v in self.by_severity.items()},
            "total_requirements": len(self.results),
            "compliant_count": sum(1 for r in self.results if r.compliant),
            "non_compliant_count": sum(1 for r in self.results if not r.compliant),
            "results": [r.to_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Render a human-readable compliance report."""
        lines = [
            "# IEEE Standards Compliance Report",
            "",
            f"**Overall Compliance:** {self.compliance_rate:.0%}",
            "",
            "## Compliance by Standard",
            "",
            "| Standard | Compliance |",
            "|----------|-----------|",
        ]
        for std, rate in self.by_standard.items():
            lines.append(f"| IEEE {std} | {rate:.0%} |")

        lines.extend(
            [
                "",
                "## Requirement Details",
                "",
                "| ID | Standard | Severity | Status | Description |",
                "|----|----------|----------|--------|-------------|",
            ]
        )
        for r in self.results:
            status = "PASS" if r.compliant else "FAIL"
            lines.append(
                f"| {r.requirement.requirement_id} | {r.requirement.standard} "
                f"| {r.requirement.severity} | {status} "
                f"| {r.requirement.description} |"
            )

        # Non-compliant details
        nc = self.non_compliant()
        if nc:
            lines.extend(["", "## Gaps", ""])
            for r in nc:
                lines.append(f"- **{r.requirement.requirement_id}**: {r.gap}")

        return "\n".join(lines)


# ── IEEE P3394 Requirements (Universal Message Format) ───────────────

P3394_REQUIREMENTS = [
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R1",
        description="Message contains sender identification",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R2",
        description="Message contains receiver identification",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R3",
        description="Message contains timestamp",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R4",
        description="Message contains message type classification",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R5",
        description="Message contains intent declaration",
        category="message_format",
        severity="should",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R6",
        description="Message contains conversation context identifier",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R7",
        description="Message contains unique message identifier",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R8",
        description="Message payload is structured (JSON-serializable)",
        category="message_format",
        severity="must",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R9",
        description="Provenance metadata is attached to message",
        category="message_format",
        severity="should",
    ),
    IEEERequirement(
        standard="P3394",
        requirement_id="P3394-R10",
        description="Agent chain (delegation path) is recorded",
        category="message_format",
        severity="should",
    ),
]

# ── IEEE 2894-2024 Requirements (Explainable AI Guide) ───────────────

IEEE_2894_REQUIREMENTS = [
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R1",
        description="System provides explanation for its output",
        category="explainability",
        severity="must",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R2",
        description="Explanation is available at summary level (user-appropriate)",
        category="explainability",
        severity="must",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R3",
        description="Explanation is available at detailed level (auditor-appropriate)",
        category="explainability",
        severity="should",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R4",
        description="Explanation includes provenance (data sources, citations)",
        category="explainability",
        severity="must",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R5",
        description="Explanation includes decision rationale",
        category="explainability",
        severity="must",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R6",
        description="Explanation includes confidence/uncertainty information",
        category="explainability",
        severity="should",
    ),
    IEEERequirement(
        standard="2894-2024",
        requirement_id="2894-R7",
        description="Explanation is traceable to specific processing steps",
        category="explainability",
        severity="must",
    ),
]

# ── IEEE 3152-2024 Requirements (Transparent Agency) ─────────────────

IEEE_3152_REQUIREMENTS = [
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R1",
        description="Response discloses AI-generated nature",
        category="transparency",
        severity="must",
    ),
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R2",
        description="Agent identity is disclosed in response",
        category="transparency",
        severity="must",
    ),
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R3",
        description="Human/machine agency boundary is clear",
        category="transparency",
        severity="must",
    ),
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R4",
        description="System capabilities and limitations are discoverable",
        category="transparency",
        severity="should",
    ),
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R5",
        description="Audit trail is maintained for all agent actions",
        category="transparency",
        severity="must",
    ),
    IEEERequirement(
        standard="3152-2024",
        requirement_id="3152-R6",
        description="Escalation to human agent is supported",
        category="transparency",
        severity="should",
    ),
]


# ── Compliance Checker ───────────────────────────────────────────────


class IEEEComplianceChecker:
    """Check compliance against IEEE P3394, 2894-2024, and 3152-2024."""

    def check_p3394(self, message: Dict[str, Any]) -> List[ComplianceResult]:
        """Check IEEE P3394 (Universal Message Format) compliance.

        Args:
            message: A MessageEnvelope.to_dict() or raw response dict.
        """
        results = []

        # R1: Sender identification
        sender = message.get("sender")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[0],
                compliant=isinstance(sender, dict) and bool(sender.get("agent_id")),
                evidence=f"sender={sender}" if sender else "",
                gap="" if sender else "No sender identification in message",
            )
        )

        # R2: Receiver identification
        receiver = message.get("receiver")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[1],
                compliant=isinstance(receiver, dict) and bool(receiver.get("agent_id")),
                evidence=f"receiver={receiver}" if receiver else "",
                gap="" if receiver else "No receiver identification in message",
            )
        )

        # R3: Timestamp
        ts = message.get("timestamp_ms")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[2],
                compliant=isinstance(ts, (int, float)) and ts > 0,
                evidence=f"timestamp_ms={ts}" if ts else "",
                gap="" if ts else "No timestamp in message",
            )
        )

        # R4: Message type
        msg_type = message.get("message_type")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[3],
                compliant=isinstance(msg_type, str) and len(msg_type) > 0,
                evidence=f"message_type={msg_type}" if msg_type else "",
                gap="" if msg_type else "No message type classification",
            )
        )

        # R5: Intent
        intent = message.get("intent")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[4],
                compliant=isinstance(intent, str) and len(intent) > 0,
                evidence=f"intent={intent}" if intent else "",
                gap="" if intent else "No intent declaration",
            )
        )

        # R6: Conversation ID
        conv_id = message.get("conversation_id")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[5],
                compliant=isinstance(conv_id, str) and len(conv_id) > 0,
                evidence=f"conversation_id={conv_id}" if conv_id else "",
                gap="" if conv_id else "No conversation context identifier",
            )
        )

        # R7: Message ID
        msg_id = message.get("message_id")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[6],
                compliant=isinstance(msg_id, str) and len(msg_id) > 0,
                evidence=f"message_id={msg_id}" if msg_id else "",
                gap="" if msg_id else "No unique message identifier",
            )
        )

        # R8: Structured payload
        payload = message.get("payload")
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[7],
                compliant=isinstance(payload, dict),
                evidence=f"payload type={type(payload).__name__}" if payload else "",
                gap="" if isinstance(payload, dict) else "Payload is not structured",
            )
        )

        # R9: Provenance
        prov = message.get("provenance")
        has_prov = isinstance(prov, dict) and len(prov) > 0
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[8],
                compliant=has_prov,
                evidence=f"provenance keys={list(prov.keys())}" if has_prov else "",
                gap="" if has_prov else "No provenance metadata attached",
            )
        )

        # R10: Agent chain
        chain = message.get("agents_chain")
        has_chain = isinstance(chain, list) and len(chain) > 0
        results.append(
            ComplianceResult(
                requirement=P3394_REQUIREMENTS[9],
                compliant=has_chain,
                evidence=f"agents_chain={chain}" if has_chain else "",
                gap="" if has_chain else "No agent delegation chain recorded",
            )
        )

        return results

    def check_2894(
        self,
        trace: Trace,
        explanations: Optional[Dict[str, Any]] = None,
    ) -> List[ComplianceResult]:
        """Check IEEE 2894-2024 (Explainable AI Guide) compliance.

        Args:
            trace: The runtime Trace object.
            explanations: Dict of ExplanationLevel -> Explanation.to_dict(),
                          or None if no explanations were generated.
        """
        results = []
        expl = explanations or {}

        # R1: Explanation exists
        has_any = len(expl) > 0
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[0],
                compliant=has_any,
                evidence=f"explanation levels={list(expl.keys())}" if has_any else "",
                gap="" if has_any else "No explanation provided for system output",
            )
        )

        # R2: Summary level
        summary = expl.get("summary")
        has_summary = isinstance(summary, dict) and bool(summary.get("narrative"))
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[1],
                compliant=has_summary,
                evidence=summary.get("narrative", "")[:100] if has_summary else "",
                gap="" if has_summary else "No summary-level explanation for end users",
            )
        )

        # R3: Detailed level
        detailed = expl.get("detailed")
        has_detailed = isinstance(detailed, dict) and bool(detailed.get("narrative"))
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[2],
                compliant=has_detailed,
                evidence="detailed explanation present" if has_detailed else "",
                gap="" if has_detailed else "No detailed-level explanation for auditors",
            )
        )

        # R4: Provenance
        has_provenance = False
        for level_data in expl.values():
            if isinstance(level_data, dict):
                prov = level_data.get("provenance")
                if isinstance(prov, list) and len(prov) > 0:
                    has_provenance = True
                    break
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[3],
                compliant=has_provenance,
                evidence="provenance data present" if has_provenance else "",
                gap=(
                    ""
                    if has_provenance
                    else "No provenance (data sources/citations) in explanation"
                ),
            )
        )

        # R5: Decision rationale
        has_decisions = False
        for level_data in expl.values():
            if isinstance(level_data, dict):
                decisions = level_data.get("decisions")
                if isinstance(decisions, list) and len(decisions) > 0:
                    has_decisions = True
                    break
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[4],
                compliant=has_decisions,
                evidence="decision rationale present" if has_decisions else "",
                gap="" if has_decisions else "No decision rationale in explanation",
            )
        )

        # R6: Confidence/uncertainty
        has_confidence = False
        for level_data in expl.values():
            if isinstance(level_data, dict):
                metrics = level_data.get("metrics", {})
                if isinstance(metrics, dict) and (
                    "confidence_score" in metrics
                    or "mean_solvability" in metrics
                    or "coverage_ratio" in metrics
                ):
                    has_confidence = True
                    break
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[5],
                compliant=has_confidence,
                evidence="confidence metrics present" if has_confidence else "",
                gap="" if has_confidence else "No confidence/uncertainty information",
            )
        )

        # R7: Traceable to processing steps
        has_trace_link = len(trace.events) > 0
        results.append(
            ComplianceResult(
                requirement=IEEE_2894_REQUIREMENTS[6],
                compliant=has_trace_link,
                evidence=f"{len(trace.events)} trace events recorded" if has_trace_link else "",
                gap="" if has_trace_link else "No trace events linking to processing steps",
            )
        )

        return results

    def check_3152(
        self,
        response: Dict[str, Any],
        envelope: Optional[Dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> List[ComplianceResult]:
        """Check IEEE 3152-2024 (Transparent Agency) compliance.

        Args:
            response: The raw response dict.
            envelope: The MessageEnvelope.to_dict(), if available.
            trace: The runtime Trace, if available.
        """
        results = []
        env = envelope or {}

        # R1: AI-generated disclosure
        ai_flag = env.get("ai_generated")
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[0],
                compliant=ai_flag is True,
                evidence="ai_generated=True" if ai_flag else "",
                gap="" if ai_flag else "Response does not disclose AI-generated nature",
            )
        )

        # R2: Agent identity disclosed
        agent_id = response.get("agent_id") or (
            env.get("sender", {}).get("agent_id") if env else None
        )
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[1],
                compliant=bool(agent_id),
                evidence=f"agent_id={agent_id}" if agent_id else "",
                gap="" if agent_id else "No agent identity disclosed in response",
            )
        )

        # R3: Human/machine boundary clear
        sender = env.get("sender", {})
        has_boundary = isinstance(sender, dict) and "is_human" in sender and "agent_type" in sender
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[2],
                compliant=has_boundary,
                evidence=f"sender.is_human={sender.get('is_human')}" if has_boundary else "",
                gap="" if has_boundary else "Human/machine agency boundary not explicit",
            )
        )

        # R4: Capabilities discoverable
        router_plan = response.get("router_plan")
        has_caps = isinstance(router_plan, dict) and bool(router_plan.get("candidates"))
        # Also check if AOP subtask results expose agent capabilities
        if not has_caps:
            has_caps = bool(response.get("subtask_results"))
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[3],
                compliant=has_caps,
                evidence="capabilities visible via routing/AOP" if has_caps else "",
                gap="" if has_caps else "System capabilities not discoverable",
            )
        )

        # R5: Audit trail maintained
        has_audit = trace is not None and len(trace.events) > 0
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[4],
                compliant=has_audit,
                evidence=f"{len(trace.events)} trace events" if has_audit else "",
                gap="" if has_audit else "No audit trail for agent actions",
            )
        )

        # R6: Human escalation supported
        # Check if the system has escalation mechanisms
        has_escalation = False
        if trace:
            for event in trace.events:
                if "escalat" in event.stage.lower() or "hitl" in event.stage.lower():
                    has_escalation = True
                    break
                if isinstance(event.data.get("pattern"), str):
                    if "escalat" in event.data["pattern"].lower():
                        has_escalation = True
                        break
        # Also check if guardrails can block (which implies escalation path)
        if not has_escalation:
            has_escalation = response.get("orchestration_pattern") == "hitl_escalation"
        # Check registry for handoff agent as evidence of capability
        if not has_escalation and trace:
            for event in trace.events:
                candidates = event.data.get("candidates")
                if isinstance(candidates, list):
                    for c in candidates:
                        if isinstance(c, dict) and "handoff" in str(c.get("id", "")).lower():
                            has_escalation = True
                            break
        results.append(
            ComplianceResult(
                requirement=IEEE_3152_REQUIREMENTS[5],
                compliant=has_escalation,
                evidence="escalation mechanism present" if has_escalation else "",
                gap="" if has_escalation else "No human escalation path evident",
            )
        )

        return results

    def check_all(
        self,
        message: Dict[str, Any],
        trace: Trace,
        response: Dict[str, Any],
        explanations: Optional[Dict[str, Any]] = None,
        envelope: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReport:
        """Run all IEEE compliance checks and return aggregate report.

        Args:
            message: The UMF envelope dict (or raw response if no envelope).
            trace: The runtime Trace.
            response: The raw response dict.
            explanations: Multi-level explanations (from ExplainabilityEngine).
            envelope: The envelope dict (same as message if using UMF).
        """
        report = ComplianceReport()
        report.results.extend(self.check_p3394(message))
        report.results.extend(self.check_2894(trace, explanations))
        report.results.extend(self.check_3152(response, envelope or message, trace))
        return report
