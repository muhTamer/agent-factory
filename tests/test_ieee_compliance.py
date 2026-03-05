# tests/test_ieee_compliance.py
"""Tests for IEEE compliance checker, message envelope, and compliance report."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.governance.ieee_compliance import (  # noqa: E402
    ComplianceReport,
    ComplianceResult,
    IEEEComplianceChecker,
    IEEERequirement,
)
from app.governance.message_envelope import (  # noqa: E402
    AgentIdentity,
    MessageEnvelope,
    wrap_response,
)
from app.runtime.trace import Trace  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────


def _make_trace(query: str = "What is the refund policy?") -> Trace:
    trace = Trace.start(query=query, request_id="test-req-001")
    trace.add("request_received")
    trace.add("orchestration_pattern", pattern="direct")
    trace.add(
        "route",
        primary="faq_agent",
        strategy="single",
        candidates=[
            {"id": "faq_agent", "score": 0.9, "reason": "keyword match"},
        ],
    )
    trace.add("intent_inferred", intent="faq_lookup")
    trace.add("guard_pre_ok")
    trace.add("execute", agent_id="faq_agent")
    trace.add("select", selected_agent="faq_agent", score=0.9)
    trace.add("response_ready")
    trace.add("guard_post_ok")
    return trace


def _make_response() -> dict:
    return {
        "text": "Our refund policy allows returns within 30 days.",
        "answer": "Our refund policy allows returns within 30 days.",
        "agent_id": "faq_agent",
        "score": 0.9,
        "request_id": "test-req-001",
        "router_plan": {
            "primary": "faq_agent",
            "strategy": "single",
            "candidates": [{"id": "faq_agent", "score": 0.9, "reason": "keyword match"}],
        },
    }


def _make_context() -> dict:
    return {
        "thread_id": "test-thread-001",
        "intent": "faq_lookup",
        "original_query": "What is the refund policy?",
    }


def _make_envelope() -> dict:
    trace = _make_trace()
    response = _make_response()
    context = _make_context()
    envelope = wrap_response(response, trace, context)
    return envelope.to_dict()


# ── Message Envelope Tests ───────────────────────────────────────────


class TestMessageEnvelope:
    def test_wrap_response_creates_valid_envelope(self):
        trace = _make_trace()
        response = _make_response()
        context = _make_context()
        envelope = wrap_response(response, trace, context)

        assert isinstance(envelope, MessageEnvelope)
        assert envelope.conversation_id == "test-thread-001"
        assert envelope.message_type == "response"
        assert envelope.intent == "faq_lookup"
        assert envelope.ai_generated is True
        assert isinstance(envelope.payload, dict)
        assert envelope.sender.agent_id == "faq_agent"
        assert envelope.receiver.is_human is True

    def test_envelope_to_dict_has_required_fields(self):
        d = _make_envelope()
        required = [
            "message_id",
            "conversation_id",
            "timestamp_ms",
            "sender",
            "receiver",
            "message_type",
            "intent",
            "payload",
            "provenance",
            "ai_generated",
            "agents_chain",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_envelope_provenance_includes_routing(self):
        d = _make_envelope()
        prov = d["provenance"]
        assert "request_id" in prov
        assert "pipeline_stages" in prov
        assert "routing" in prov
        assert prov["routing"]["primary"] == "faq_agent"

    def test_agents_chain_extracted(self):
        d = _make_envelope()
        chain = d["agents_chain"]
        assert isinstance(chain, list)
        assert "faq_agent" in chain

    def test_agent_identity_serialization(self):
        identity = AgentIdentity(
            agent_id="test_agent",
            agent_type="faq_rag",
            is_human=False,
            version="2.0",
        )
        d = identity.to_dict()
        assert d["agent_id"] == "test_agent"
        assert d["is_human"] is False


# ── IEEE P3394 Tests ─────────────────────────────────────────────────


class TestP3394Compliance:
    def test_full_envelope_is_compliant(self):
        checker = IEEEComplianceChecker()
        envelope = _make_envelope()
        results = checker.check_p3394(envelope)

        assert len(results) == 10  # 10 requirements
        for r in results:
            assert r.compliant, f"{r.requirement.requirement_id} failed: {r.gap}"

    def test_raw_response_is_not_compliant(self):
        checker = IEEEComplianceChecker()
        raw = _make_response()  # No envelope fields
        results = checker.check_p3394(raw)

        non_compliant = [r for r in results if not r.compliant]
        # Raw response lacks sender, receiver, timestamp, message_type, etc.
        assert len(non_compliant) >= 5

    def test_partial_envelope_reports_gaps(self):
        checker = IEEEComplianceChecker()
        partial = {
            "message_id": "abc",
            "conversation_id": "thread-1",
            "timestamp_ms": 1234567890,
            # missing sender, receiver, message_type, intent
            "payload": {"text": "hello"},
        }
        results = checker.check_p3394(partial)
        gaps = [r for r in results if not r.compliant]
        assert any(r.requirement.requirement_id == "P3394-R1" for r in gaps)  # sender


# ── IEEE 2894-2024 Tests ─────────────────────────────────────────────


class TestIEEE2894Compliance:
    def test_with_full_explanations(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        response = _make_response()

        from app.governance.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()
        explanations = {
            k: v.to_dict() for k, v in engine.generate_all_levels(trace, response).items()
        }

        results = checker.check_2894(trace, explanations)
        assert len(results) == 7  # 7 requirements

        # With full explanations, most should pass
        compliant = [r for r in results if r.compliant]
        assert len(compliant) >= 5

    def test_without_explanations(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        results = checker.check_2894(trace, explanations=None)

        # Without explanations, R1-R6 should fail (R7 may pass if trace has events)
        non_compliant = [r for r in results if not r.compliant]
        assert len(non_compliant) >= 5

    def test_trace_events_satisfy_traceability(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        results = checker.check_2894(trace, explanations={})

        # R7 should pass — trace has events
        r7 = [r for r in results if r.requirement.requirement_id == "2894-R7"]
        assert len(r7) == 1
        assert r7[0].compliant


# ── IEEE 3152-2024 Tests ─────────────────────────────────────────────


class TestIEEE3152Compliance:
    def test_with_envelope(self):
        checker = IEEEComplianceChecker()
        response = _make_response()
        envelope = _make_envelope()
        trace = _make_trace()

        results = checker.check_3152(response, envelope, trace)
        assert len(results) == 6  # 6 requirements

        # With envelope, ai_generated and agent identity should pass
        r1 = [r for r in results if r.requirement.requirement_id == "3152-R1"][0]
        assert r1.compliant  # ai_generated=True

        r2 = [r for r in results if r.requirement.requirement_id == "3152-R2"][0]
        assert r2.compliant  # agent_id present

    def test_without_envelope(self):
        checker = IEEEComplianceChecker()
        response = _make_response()
        results = checker.check_3152(response, envelope=None, trace=None)

        r1 = [r for r in results if r.requirement.requirement_id == "3152-R1"][0]
        assert not r1.compliant  # no ai_generated flag

    def test_audit_trail_requirement(self):
        checker = IEEEComplianceChecker()
        response = _make_response()
        trace = _make_trace()
        results = checker.check_3152(response, trace=trace)

        r5 = [r for r in results if r.requirement.requirement_id == "3152-R5"][0]
        assert r5.compliant  # trace has events


# ── Compliance Report Tests ──────────────────────────────────────────


class TestComplianceReport:
    def test_full_check_all_produces_report(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        response = _make_response()
        envelope = _make_envelope()

        from app.governance.explainability import ExplainabilityEngine

        engine = ExplainabilityEngine()
        explanations = {
            k: v.to_dict() for k, v in engine.generate_all_levels(trace, response).items()
        }

        report = checker.check_all(
            message=envelope,
            trace=trace,
            response=response,
            explanations=explanations,
            envelope=envelope,
        )

        assert isinstance(report, ComplianceReport)
        assert len(report.results) == 23  # 10 + 7 + 6
        assert 0.0 <= report.compliance_rate <= 1.0

        # By standard
        by_std = report.by_standard
        assert "P3394" in by_std
        assert "2894-2024" in by_std
        assert "3152-2024" in by_std

    def test_report_to_dict(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        response = _make_response()
        envelope = _make_envelope()
        report = checker.check_all(envelope, trace, response, envelope=envelope)

        d = report.to_dict()
        assert "compliance_rate" in d
        assert "by_standard" in d
        assert "results" in d
        assert isinstance(d["results"], list)

    def test_report_to_markdown(self):
        checker = IEEEComplianceChecker()
        trace = _make_trace()
        response = _make_response()
        envelope = _make_envelope()
        report = checker.check_all(envelope, trace, response, envelope=envelope)

        md = report.to_markdown()
        assert "# IEEE Standards Compliance Report" in md
        assert "P3394" in md
        assert "2894-2024" in md
        assert "3152-2024" in md

    def test_non_compliant_filter(self):
        report = ComplianceReport(
            results=[
                ComplianceResult(
                    requirement=IEEERequirement("P3394", "R1", "test", "cat", "must"),
                    compliant=True,
                ),
                ComplianceResult(
                    requirement=IEEERequirement("P3394", "R2", "test", "cat", "must"),
                    compliant=False,
                    gap="missing",
                ),
            ]
        )
        nc = report.non_compliant()
        assert len(nc) == 1
        assert nc[0].requirement.requirement_id == "R2"

    def test_by_severity(self):
        report = ComplianceReport(
            results=[
                ComplianceResult(
                    requirement=IEEERequirement("P3394", "R1", "test", "cat", "must"),
                    compliant=True,
                ),
                ComplianceResult(
                    requirement=IEEERequirement("P3394", "R2", "test", "cat", "should"),
                    compliant=False,
                ),
            ]
        )
        by_sev = report.by_severity
        assert by_sev["must"] == 1.0
        assert by_sev["should"] == 0.0
