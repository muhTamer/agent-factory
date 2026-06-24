# tests/test_explainability.py
"""Tests for multi-level explainability engine and PII redactor."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.governance.explainability import (  # noqa: E402
    Explanation,
    ExplainabilityEngine,
    ExplanationLevel,
)
from app.governance.pii_redactor import PIIRedactor, RedactionRecord  # noqa: E402
from app.runtime.trace import Trace  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────


def _trace_direct() -> Trace:
    """A simple direct-routing trace."""
    trace = Trace.start(query="What is the refund policy?", request_id="test-001")
    trace.add("request_received")
    trace.add("orchestration_pattern", pattern="direct")
    trace.add(
        "route",
        primary="faq_agent",
        strategy="single",
        candidates=[
            {"id": "faq_agent", "score": 0.85, "reason": "keyword match"},
        ],
    )
    trace.add("intent_inferred", intent="faq_lookup")
    trace.add("guard_pre_ok")
    trace.add("execute", agent_id="faq_agent")
    trace.add("select", selected_agent="faq_agent", score=0.85)
    trace.add("response_ready")
    trace.add("guard_post_ok")
    return trace


def _response_direct() -> dict:
    return {
        "text": "Our refund policy allows returns within 30 days.",
        "answer": "Our refund policy allows returns within 30 days.",
        "agent_id": "faq_agent",
        "score": 0.85,
        "request_id": "test-001",
        "router_plan": {
            "primary": "faq_agent",
            "strategy": "single",
            "candidates": [{"id": "faq_agent", "score": 0.85}],
        },
    }


def _trace_aop() -> Trace:
    """An AOP hierarchical delegation trace."""
    trace = Trace.start(
        query="Tell me about account types and refund policy", request_id="test-002"
    )
    trace.add("request_received")
    trace.add("orchestration_pattern", pattern="hierarchical_delegation")
    trace.add("aop_decompose", subtasks=["account types", "refund policy"])
    trace.add(
        "aop_solvability",
        assignments={
            "account types": "faq_agent",
            "refund policy": "faq_agent",
        },
    )
    trace.add("aop_completeness", complete=True, missing=[])
    trace.add(
        "aop_execute",
        results=[
            {"subtask": "account types", "agent": "faq_agent", "success": True},
            {"subtask": "refund policy", "agent": "faq_agent", "success": True},
        ],
    )
    trace.add("guard_post_ok")
    return trace


def _response_aop() -> dict:
    return {
        "text": "Account types and refund information.",
        "answer": "Account types and refund information.",
        "score": 0.88,
        "orchestration_pattern": "hierarchical_delegation",
        "subtask_results": [
            {
                "subtask": "account types",
                "agent_id": "faq_agent",
                "success": True,
                "solvability_score": 0.9,
                "latency_ms": 150,
                "result": {"answer": "We offer Current and Savings accounts."},
            },
            {
                "subtask": "refund policy",
                "agent_id": "faq_agent",
                "success": True,
                "solvability_score": 0.85,
                "latency_ms": 120,
                "result": {"answer": "Returns within 30 days."},
            },
        ],
        "completeness": {
            "complete": True,
            "missing": [],
            "coverage_ratio": 1.0,
            "reasoning": "All aspects covered.",
        },
        "solvability": {
            "assignments": {"account types": "faq_agent", "refund policy": "faq_agent"},
            "assignment_scores": {"account types": 0.9, "refund policy": 0.85},
        },
    }


# ── Explainability Engine Tests ──────────────────────────────────────


class TestExplainabilityEngine:
    def test_generate_summary_direct(self):
        engine = ExplainabilityEngine()
        trace = _trace_direct()
        response = _response_direct()
        expl = engine.generate(trace, response, ExplanationLevel.SUMMARY)

        assert isinstance(expl, Explanation)
        assert expl.level == ExplanationLevel.SUMMARY
        assert len(expl.narrative) > 0
        assert "faq_agent" in expl.agents_involved or "FAQ" in expl.narrative

    def test_generate_summary_aop(self):
        engine = ExplainabilityEngine()
        trace = _trace_aop()
        response = _response_aop()
        expl = engine.generate(trace, response, ExplanationLevel.SUMMARY)

        assert (
            "2 part" in expl.narrative
            or "2 specialist" in expl.narrative
            or "decomposition" in expl.narrative
        )

    def test_generate_detailed(self):
        engine = ExplainabilityEngine()
        trace = _trace_direct()
        response = _response_direct()
        expl = engine.generate(trace, response, ExplanationLevel.DETAILED)

        assert expl.level == ExplanationLevel.DETAILED
        assert len(expl.decisions) > 0
        assert len(expl.narrative) > 0
        # Should mention routing
        assert any(d["stage"] == "route" for d in expl.decisions)

    def test_generate_detailed_aop(self):
        engine = ExplainabilityEngine()
        trace = _trace_aop()
        response = _response_aop()
        expl = engine.generate(trace, response, ExplanationLevel.DETAILED)

        assert len(expl.provenance) > 0
        assert expl.metrics.get("coverage_ratio") == 1.0

    def test_generate_full(self):
        engine = ExplainabilityEngine()
        trace = _trace_direct()
        response = _response_direct()
        expl = engine.generate(trace, response, ExplanationLevel.FULL)

        assert expl.level == ExplanationLevel.FULL
        assert "event_count" in expl.metrics
        assert expl.metrics["event_count"] == len(trace.events)

    def test_generate_all_levels(self):
        engine = ExplainabilityEngine()
        trace = _trace_direct()
        response = _response_direct()
        all_expl = engine.generate_all_levels(trace, response)

        assert "summary" in all_expl
        assert "detailed" in all_expl
        assert "full" in all_expl
        assert all_expl["summary"].level == ExplanationLevel.SUMMARY
        assert all_expl["detailed"].level == ExplanationLevel.DETAILED
        assert all_expl["full"].level == ExplanationLevel.FULL

    def test_explanation_to_dict(self):
        engine = ExplainabilityEngine()
        trace = _trace_direct()
        response = _response_direct()
        expl = engine.generate(trace, response, ExplanationLevel.SUMMARY)
        d = expl.to_dict()

        assert d["level"] == "summary"
        assert isinstance(d["narrative"], str)
        assert isinstance(d["agents_involved"], list)
        assert isinstance(d["metrics"], dict)

    def test_extract_agents_from_response(self):
        engine = ExplainabilityEngine()
        trace = _trace_aop()
        response = _response_aop()
        agents = engine._extract_agents(trace, response)

        assert "faq_agent" in agents

    def test_extract_metrics_aop(self):
        engine = ExplainabilityEngine()
        trace = _trace_aop()
        response = _response_aop()
        metrics = engine._extract_metrics(trace, response)

        assert metrics["subtask_count"] == 2
        assert metrics["subtask_success_rate"] == 1.0
        assert metrics["mean_solvability"] > 0
        assert metrics["coverage_ratio"] == 1.0


# ── PII Redactor Tests ───────────────────────────────────────────────


class TestPIIRedactor:
    def setup_method(self):
        self.redactor = PIIRedactor()

    def test_redact_email(self):
        text = "Contact john.doe@example.com for info"
        redacted, records = self.redactor.redact(text)
        assert "[EMAIL_REDACTED]" in redacted
        assert "john.doe@example.com" not in redacted
        assert len(records) == 1
        assert records[0].pii_type == "email"

    def test_redact_phone_international(self):
        text = "Call us at +45 12345678 for support"
        redacted, records = self.redactor.redact(text)
        assert "[PHONE_REDACTED]" in redacted
        assert "12345678" not in redacted

    def test_redact_credit_card_with_luhn(self):
        # 4111111111111111 is a valid Luhn test card
        text = "Card number: 4111 1111 1111 1111"
        redacted, records = self.redactor.redact(text)
        assert "[CARD_REDACTED]" in redacted
        assert "4111" not in redacted
        assert any(r.pii_type == "credit_card" for r in records)

    def test_no_false_positive_on_invalid_card(self):
        # Random number that fails Luhn
        text = "Reference: 1234 5678 9012 3456"
        redacted, records = self.redactor.redact(text)
        card_records = [r for r in records if r.pii_type == "credit_card"]
        assert len(card_records) == 0

    def test_redact_national_id_with_dashes(self):
        text = "SSN: 123-45-6789"
        redacted, records = self.redactor.redact(text)
        assert "[ID_REDACTED]" in redacted
        assert "123-45-6789" not in redacted

    def test_no_false_positive_on_plain_number(self):
        text = "Order 123456789 is ready"
        redacted, records = self.redactor.redact(text)
        id_records = [r for r in records if r.pii_type == "national_id"]
        assert len(id_records) == 0

    def test_redact_multiple_pii(self):
        text = "Email john@test.com, call +45 12345678"
        redacted, records = self.redactor.redact(text)
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted
        assert len(records) >= 2

    def test_redact_dict_shallow(self):
        data = {
            "answer": "Contact john@test.com",
            "score": 0.9,
        }
        redacted, records = self.redactor.redact_dict(data)
        assert "[EMAIL_REDACTED]" in redacted["answer"]
        assert redacted["score"] == 0.9  # non-string untouched
        assert len(records) == 1

    def test_redact_dict_nested(self):
        data = {
            "result": {
                "text": "Call +45 12345678",
                "meta": {"email": "user@test.com"},
            },
        }
        redacted, records = self.redactor.redact_dict(data)
        assert "[PHONE_REDACTED]" in redacted["result"]["text"]
        assert "[EMAIL_REDACTED]" in redacted["result"]["meta"]["email"]
        assert len(records) >= 2

    def test_redact_dict_with_list(self):
        data = {
            "messages": ["Contact john@example.com", "No PII here"],
        }
        redacted, records = self.redactor.redact_dict(data)
        assert "[EMAIL_REDACTED]" in redacted["messages"][0]
        assert redacted["messages"][1] == "No PII here"

    def test_no_pii_returns_unchanged(self):
        text = "What is the refund policy?"
        redacted, records = self.redactor.redact(text)
        assert redacted == text
        assert len(records) == 0

    def test_redaction_record_to_dict(self):
        record = RedactionRecord(
            pii_type="email",
            original_snippet="j***m",
            position=5,
            replacement="[EMAIL_REDACTED]",
        )
        d = record.to_dict()
        assert d["pii_type"] == "email"
        assert d["replacement"] == "[EMAIL_REDACTED]"

    def test_iban_redaction(self):
        text = "IBAN: DK50 0040 0440 1162 43"
        redacted, records = self.redactor.redact(text)
        assert "[IBAN_REDACTED]" in redacted
        assert any(r.pii_type == "iban" for r in records)
