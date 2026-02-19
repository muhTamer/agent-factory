# tests/test_rag_fsm.py
"""Tests for the RAG internal FSM with solvability estimation."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List

from app.runtime.memory import ConversationMemory
from app.runtime.rag_fsm import (
    RAGFSMConfig,
    RAGFiniteStateMachine,
    RAGState,
    _tok,
)


# ── Test data: small FAQ corpus ──────────────────────────────────────

_TEST_FAQS = [
    {
        "q": "What is the refund policy?",
        "a": "You can get a refund within 30 days.",
        "source": "faq.csv",
    },
    {
        "q": "How do I reset my password?",
        "a": "Go to Settings > Security > Reset Password.",
        "source": "faq.csv",
    },
    {
        "q": "What payment methods do you accept?",
        "a": "We accept Visa, Mastercard, and PayPal.",
        "source": "faq.csv",
    },
    {
        "q": "How do I contact support?",
        "a": "Email support@example.com or call 1-800-HELP.",
        "source": "faq.csv",
    },
    {
        "q": "What are your business hours?",
        "a": "Monday to Friday, 9am to 5pm.",
        "source": "faq.csv",
    },
]


def _build_index(faqs: List[Dict[str, Any]]):
    """Build TF-IDF index from test FAQs (same logic as generated agent)."""
    _WORD = re.compile(r"[A-Za-z0-9]+")

    texts = [f"Q: {f['q']} A: {f['a']}" for f in faqs]

    # Build IDF
    df: Dict[str, int] = {}
    for text in texts:
        seen = set(_tok(text))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(texts))
    idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

    # Build vecs
    vecs: List[Dict[str, float]] = []
    for text in texts:
        tf: Dict[str, int] = {}
        for t in _tok(text):
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
        vecs.append(vec)

    return idf, vecs, texts


def _make_fsm(
    config: RAGFSMConfig | None = None,
    memory: ConversationMemory | None = None,
) -> RAGFiniteStateMachine:
    """Create an FSM with the test FAQ corpus."""
    idf, vecs, texts = _build_index(_TEST_FAQS)
    return RAGFiniteStateMachine(
        agent_id="test_faq",
        faqs=_TEST_FAQS,
        idf=idf,
        vecs=vecs,
        texts=texts,
        config=config,
        memory=memory,
    )


# ── Solvability Estimation Tests ────────────────────────────────────


class TestSolvabilityEstimation:
    def test_high_overlap_query_scores_above_threshold(self):
        fsm = _make_fsm()
        solv = fsm.estimate_solvability("What is the refund policy?")
        assert solv.tfidf_score > 0.3
        assert solv.confidence > 0.25
        assert solv.should_delegate is False

    def test_low_overlap_query_scores_below_threshold(self):
        fsm = _make_fsm()
        solv = fsm.estimate_solvability("xyzzy foobar baz quantum computing")
        assert solv.tfidf_score < 0.15
        assert solv.confidence < 0.25
        assert solv.should_delegate is True

    def test_coverage_ratio_reflects_vocabulary_overlap(self):
        fsm = _make_fsm()
        # "refund policy" — both tokens in corpus
        solv = fsm.estimate_solvability("refund policy")
        assert solv.coverage_ratio > 0.5

        # Nonsense tokens — none in corpus
        solv2 = fsm.estimate_solvability("xyzzy foobar")
        assert solv2.coverage_ratio == 0.0

    def test_empty_query_returns_zero_confidence(self):
        fsm = _make_fsm()
        solv = fsm.estimate_solvability("")
        assert solv.confidence == 0.0
        assert solv.should_delegate is True

    def test_signals_combined_formula(self):
        """Verify the weighting: 0.5*tfidf + 0.3*coverage + 0.2*top_k_avg."""
        fsm = _make_fsm()
        solv = fsm.estimate_solvability("What is the refund policy?")
        expected = 0.5 * solv.tfidf_score + 0.3 * solv.coverage_ratio + 0.2 * solv.top_k_avg
        assert abs(solv.confidence - expected) < 0.01

    def test_reasoning_string_populated(self):
        fsm = _make_fsm()
        solv = fsm.estimate_solvability("refund")
        assert "tfidf=" in solv.reasoning
        assert "coverage=" in solv.reasoning
        assert "confidence=" in solv.reasoning


# ── FSM State Transition Tests ──────────────────────────────────────


class TestFSMStateTransitions:
    def test_initial_state_is_analyze(self):
        fsm = _make_fsm()
        assert fsm.state == RAGState.ANALYZE

    def test_high_solvability_goes_to_respond(self):
        """Clear FAQ query should go ANALYZE -> RETRIEVE -> RESPOND."""
        fsm = _make_fsm()
        result = fsm.step("What is the refund policy?")
        assert result.state == RAGState.RESPOND
        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.score > 0.0

    def test_low_solvability_goes_to_delegate(self):
        """Out-of-scope query should go ANALYZE -> DELEGATE."""
        fsm = _make_fsm()
        result = fsm.step("Tell me about quantum physics dark matter singularity")
        assert result.state == RAGState.DELEGATE
        assert result.delegation_reason is not None

    def test_vague_short_query_goes_to_clarify(self):
        """Very short, vague query should trigger clarification."""
        config = RAGFSMConfig(
            solvability_threshold=0.25,
            min_query_tokens=3,  # "um" is only 1 token, not in corpus
        )
        fsm = _make_fsm(config=config)
        result = fsm.step("um")
        assert result.state == RAGState.CLARIFY
        assert result.clarification_question is not None

    def test_clarify_returns_clarification_question(self):
        config = RAGFSMConfig(min_query_tokens=3)
        fsm = _make_fsm(config=config)
        result = fsm.step("um")
        assert result.state == RAGState.CLARIFY
        assert (
            "more details" in result.clarification_question.lower()
            or "specific" in result.clarification_question.lower()
        )

    def test_clarify_with_better_query_goes_to_respond(self):
        """After clarification, a good query should go to RESPOND."""
        config = RAGFSMConfig(min_query_tokens=3)
        fsm = _make_fsm(config=config)

        # First step: vague -> CLARIFY
        result1 = fsm.step("um")
        assert result1.state == RAGState.CLARIFY

        # Second step: clear query -> ANALYZE -> RETRIEVE -> RESPOND
        result2 = fsm.step("What is the refund policy?")
        assert result2.state == RAGState.RESPOND
        assert result2.answer is not None

    def test_max_clarifications_then_delegate(self):
        """After max clarification rounds, should delegate."""
        config = RAGFSMConfig(
            min_query_tokens=10,  # Force clarification for short queries
            max_clarifications=2,
            solvability_threshold=0.99,  # Force delegation for anything
        )
        fsm = _make_fsm(config=config)

        # Round 1: clarify
        r1 = fsm.step("x")
        assert r1.state == RAGState.CLARIFY

        # Round 2: still vague -> clarify again
        r2 = fsm.step("y")
        assert r2.state == RAGState.CLARIFY

        # Round 3: max reached -> delegate
        r3 = fsm.step("z")
        assert r3.state == RAGState.DELEGATE

    def test_retrieve_returns_answer_and_citations(self):
        fsm = _make_fsm()
        result = fsm.step("How do I reset my password?")
        assert result.state == RAGState.RESPOND
        assert "password" in result.answer.lower() or "settings" in result.answer.lower()
        assert len(result.citations) > 0
        assert "score" in result.citations[0]

    def test_delegation_result_has_target(self):
        config = RAGFSMConfig(delegation_target="refund_agent")
        fsm = _make_fsm(config=config)
        result = fsm.step("xyzzy foobar completely unrelated quantum")
        assert result.state == RAGState.DELEGATE
        assert result.delegation_target == "refund_agent"

    def test_solvability_attached_to_result(self):
        fsm = _make_fsm()
        result = fsm.step("What is the refund policy?")
        assert result.solvability is not None
        assert result.solvability.confidence > 0

    def test_reset_returns_to_analyze(self):
        fsm = _make_fsm()
        fsm.step("What is the refund policy?")
        assert fsm.state == RAGState.RESPOND
        fsm.reset()
        assert fsm.state == RAGState.ANALYZE


# ── Query Expansion Tests ───────────────────────────────────────────


class TestQueryExpansion:
    def test_expand_query_appends_prior_context(self):
        mem = ConversationMemory()
        mem.record_turn(
            "t1",
            query="What is the refund policy?",
            response={"answer": "30-day window"},
        )
        fsm = _make_fsm(memory=mem)
        expanded = fsm._expand_query("how long", "t1")
        # Should append keywords from prior turn
        assert len(expanded) > len("how long")
        assert "refund" in expanded.lower() or "policy" in expanded.lower()

    def test_expand_query_without_memory_returns_original(self):
        fsm = _make_fsm(memory=None)
        assert fsm._expand_query("test query", "t1") == "test query"

    def test_expand_query_empty_history_returns_original(self):
        mem = ConversationMemory()
        fsm = _make_fsm(memory=mem)
        assert fsm._expand_query("test query", "t1") == "test query"


# ── Integration Tests ───────────────────────────────────────────────


class TestIntegration:
    def test_full_happy_path_analyze_retrieve_respond(self):
        fsm = _make_fsm()
        result = fsm.step("What payment methods do you accept?")
        assert result.state == RAGState.RESPOND
        assert result.answer is not None
        assert result.score > 0.0
        assert result.solvability is not None
        assert result.solvability.should_delegate is False

    def test_full_delegation_path(self):
        config = RAGFSMConfig(solvability_threshold=0.99)  # Force delegation
        fsm = _make_fsm(config=config)
        result = fsm.step("This is a completely unrelated question about astrophysics")
        assert result.state == RAGState.DELEGATE

    def test_clarification_then_answer_path(self):
        config = RAGFSMConfig(min_query_tokens=3)
        mem = ConversationMemory()
        fsm = _make_fsm(config=config, memory=mem)

        # Step 1: vague query
        r1 = fsm.step("hi", "t1")
        assert r1.state == RAGState.CLARIFY

        # Step 2: better query
        r2 = fsm.step("What are your business hours?", "t1")
        assert r2.state == RAGState.RESPOND
        assert (
            "monday" in r2.answer.lower()
            or "9am" in r2.answer.lower()
            or "friday" in r2.answer.lower()
        )
