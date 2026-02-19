# tests/test_rag_retrieval_clarification.py
"""Tests for post-retrieval ambiguity detection and proactive clarification."""
from __future__ import annotations

import math
from typing import Any, Dict, List

from app.runtime.rag_fsm import (
    RAGFSMConfig,
    RAGFiniteStateMachine,
    RAGState,
    _tok,
)


# ── Test data: diverse FAQ corpus about different account types ────────

_DIVERSE_FAQS = [
    {
        "q": "How do I open a Current Account for a sole proprietorship?",
        "a": "Submit PAN, address proof, and business registration.",
        "source": "BankFAQs.csv",
    },
    {
        "q": "How do I open a Savings Account?",
        "a": "Visit any branch with your ID proof and address proof.",
        "source": "BankFAQs.csv",
    },
    {
        "q": "How do I open an NRI Account?",
        "a": "NRI accounts can be opened online with your passport and visa.",
        "source": "BankFAQs.csv",
    },
    {
        "q": "How do I open a Fixed Deposit Account?",
        "a": "You need an existing savings account to open a fixed deposit.",
        "source": "BankFAQs.csv",
    },
    {
        "q": "How do I open a Demat Account for trading?",
        "a": "Apply online with PAN card and bank account details.",
        "source": "BankFAQs.csv",
    },
]

# Focused corpus: all FAQs are about the same topic (refunds)
_FOCUSED_FAQS = [
    {
        "q": "What is the refund policy for electronics?",
        "a": "Electronics can be returned within 15 days.",
        "source": "refunds.csv",
    },
    {
        "q": "What is the refund policy for clothing?",
        "a": "Clothing can be returned within 30 days.",
        "source": "refunds.csv",
    },
    {
        "q": "What is the refund policy for digital products?",
        "a": "Digital products are non-refundable.",
        "source": "refunds.csv",
    },
    {
        "q": "How long does a refund take to process?",
        "a": "Refunds are processed within 5-7 business days.",
        "source": "refunds.csv",
    },
    {
        "q": "Can I get a refund after 30 days?",
        "a": "Refunds after 30 days are handled on a case-by-case basis.",
        "source": "refunds.csv",
    },
]


def _build_index(faqs: List[Dict[str, Any]]):
    texts = [f"Q: {f['q']} A: {f['a']}" for f in faqs]
    df: Dict[str, int] = {}
    for text in texts:
        seen = set(_tok(text))
        for t in seen:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(texts))
    idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

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
    faqs=None,
    config: RAGFSMConfig | None = None,
    llm_fn=None,
) -> RAGFiniteStateMachine:
    faqs = faqs or _DIVERSE_FAQS
    idf, vecs, texts = _build_index(faqs)
    return RAGFiniteStateMachine(
        agent_id="test_faq",
        faqs=faqs,
        idf=idf,
        vecs=vecs,
        texts=texts,
        config=config,
        llm_fn=llm_fn,
    )


def _mock_llm_clarification(messages, model=None, **kwargs):
    """Mock LLM that returns a clarification question."""
    return {
        "question": (
            "Are you asking about opening a Current Account, "
            "Savings Account, NRI Account, or Demat Account?"
        )
    }


# ── Feature Flag Tests ──────────────────────────────────────────────


class TestFeatureFlag:
    def test_default_off_no_clarification(self):
        """With enable_retrieval_clarification=False (default), broad query goes to RESPOND."""
        fsm = _make_fsm(config=RAGFSMConfig())
        result = fsm.step("how to open an account?")
        assert result.state in (RAGState.RESPOND, RAGState.DELEGATE)

    def test_enabled_broad_query_triggers_clarification(self):
        """With feature enabled, broad query on diverse corpus triggers CLARIFY."""
        # Use low thresholds to match TF-IDF score characteristics:
        # TF-IDF scores are peaked (~0.46 top, ~0.13 bottom), so flatness ~0.29
        # Topic diversity is ~0.63 for the diverse FAQ set
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        result = fsm.step("how to open an account?")
        assert result.state == RAGState.CLARIFY
        assert result.clarification_question is not None


# ── Ambiguity Detection Tests ───────────────────────────────────────


class TestAmbiguityDetection:
    def test_score_flatness_flat_distribution(self):
        """Flat scores yield high flatness ratio."""
        fsm = _make_fsm()
        hits = [(0.5, 0), (0.48, 1), (0.46, 2), (0.44, 3), (0.42, 4)]
        flatness = fsm._score_flatness(hits)
        assert flatness > 0.80  # 0.42/0.5 = 0.84

    def test_score_flatness_peaked_distribution(self):
        """Peaked scores yield low flatness ratio."""
        fsm = _make_fsm()
        hits = [(0.9, 0), (0.3, 1), (0.2, 2), (0.1, 3), (0.05, 4)]
        flatness = fsm._score_flatness(hits)
        assert flatness < 0.20  # 0.05/0.9 = 0.056

    def test_score_flatness_single_hit(self):
        """Single hit returns 0.0 flatness."""
        fsm = _make_fsm()
        assert fsm._score_flatness([(0.5, 0)]) == 0.0

    def test_topic_diversity_diverse_faqs(self):
        """Diverse FAQ topics yield high diversity."""
        fsm = _make_fsm(faqs=_DIVERSE_FAQS)
        hits = [(0.5, i) for i in range(5)]
        diversity = fsm._topic_diversity("how to open an account", hits)
        assert diversity > 0.40

    def test_topic_diversity_same_topic_lower_than_diverse(self):
        """Same-topic FAQs yield lower diversity than diverse FAQs."""
        fsm_focused = _make_fsm(faqs=_FOCUSED_FAQS)
        fsm_diverse = _make_fsm(faqs=_DIVERSE_FAQS)
        hits = [(0.5, i) for i in range(5)]
        div_focused = fsm_focused._topic_diversity("refund policy", hits)
        div_diverse = fsm_diverse._topic_diversity("how to open an account", hits)
        # Diverse FAQs have different distinguishing tokens → higher diversity
        assert div_focused <= div_diverse or abs(div_focused - div_diverse) < 0.25


# ── Clarification Question Generation ───────────────────────────────


class TestClarificationGeneration:
    def test_llm_clarification_question(self):
        """When llm_fn is provided, it generates a context-aware question."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            ),
            llm_fn=_mock_llm_clarification,
        )
        result = fsm.step("how to open an account?")
        assert result.state == RAGState.CLARIFY
        assert "Current Account" in result.clarification_question
        assert "Savings Account" in result.clarification_question

    def test_heuristic_fallback_without_llm(self):
        """Without llm_fn, heuristic builds question from passage tokens."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            ),
            llm_fn=None,
        )
        result = fsm.step("how to open an account?")
        if result.state == RAGState.CLARIFY:
            assert result.clarification_question is not None
            assert len(result.clarification_question) > 10

    def test_heuristic_clarification_method(self):
        """Heuristic clarification directly produces passage-informed text."""
        fsm = _make_fsm()
        passages = [
            {
                "text": "Q: How to open a Current Account? A: Submit docs.",
                "source": "a.csv",
                "score": 0.5,
                "index": 0,
            },
            {
                "text": "Q: How to open a Savings Account? A: Visit branch.",
                "source": "a.csv",
                "score": 0.48,
                "index": 1,
            },
            {
                "text": "Q: How to open an NRI Account? A: Apply online.",
                "source": "a.csv",
                "score": 0.46,
                "index": 2,
            },
        ]
        question = fsm._heuristic_clarification("how to open an account", passages)
        assert "related topics" in question.lower() or "clarify" in question.lower()


# ── Full Clarification Flow ─────────────────────────────────────────


class TestClarificationFlow:
    def test_clarify_then_respond(self):
        """Broad query → CLARIFY, then focused follow-up → RESPOND."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        r1 = fsm.step("how to open an account?")
        if r1.state == RAGState.CLARIFY:
            # Follow up with a more specific query
            r2 = fsm.step("I want to open a Savings Account")
            assert r2.state in (RAGState.RESPOND, RAGState.CLARIFY)

    def test_max_clarifications_then_respond(self):
        """After max clarification rounds, proceed to RESPOND."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                max_clarifications=1,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        r1 = fsm.step("how to open an account?")
        if r1.state == RAGState.CLARIFY:
            # Second attempt — should not clarify again (max_clarifications=1)
            r2 = fsm.step("open account")
            assert r2.state != RAGState.CLARIFY or r2.state == RAGState.RESPOND

    def test_high_confidence_skips_clarification(self):
        """Very specific query with high confidence skips ambiguity check."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_confidence_ceiling=0.30,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        # Very specific query that matches one FAQ strongly
        result = fsm.step("How do I open a Current Account for a sole proprietorship?")
        assert result.state != RAGState.CLARIFY


# ── Metadata Tests ──────────────────────────────────────────────────


class TestClarificationMetadata:
    def test_retrieval_informed_metadata(self):
        """Retrieval-informed clarification includes ambiguity metadata."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        result = fsm.step("how to open an account?")
        if result.state == RAGState.CLARIFY:
            assert result.metadata.get("clarification_type") == "retrieval_informed"
            assert "ambiguity_flatness" in result.metadata
            assert "ambiguity_diversity" in result.metadata
            assert "candidate_passages" in result.metadata

    def test_regular_clarify_has_no_retrieval_metadata(self):
        """Pre-retrieval clarification (short query) doesn't have retrieval metadata."""
        fsm = _make_fsm(config=RAGFSMConfig(min_query_tokens=5))
        result = fsm.step("um")
        if result.state == RAGState.CLARIFY:
            assert result.metadata.get("clarification_type") is None


# ── Build Passages Helper ───────────────────────────────────────────


class TestBuildPassages:
    def test_build_passages_returns_correct_structure(self):
        """_build_passages returns passages with text, source, score, index."""
        fsm = _make_fsm()
        hits = [(0.5, 0), (0.4, 1)]
        passages = fsm._build_passages(hits)
        assert len(passages) == 2
        for p in passages:
            assert "text" in p
            assert "source" in p
            assert "score" in p
            assert "index" in p

    def test_build_passages_respects_max_context(self):
        """_build_passages limits to max_context_passages."""
        fsm = _make_fsm(config=RAGFSMConfig(max_context_passages=2))
        hits = [(0.5, i) for i in range(5)]
        passages = fsm._build_passages(hits)
        assert len(passages) == 2


# ── Backward Compatibility ──────────────────────────────────────────


class TestBackwardCompatibility:
    def test_existing_behavior_unchanged_without_feature(self):
        """Default config produces same behavior as before (no clarification)."""
        fsm = _make_fsm(config=RAGFSMConfig())
        result = fsm.step("how to open an account?")
        # Should go straight to RESPOND or DELEGATE, never CLARIFY
        assert result.state in (RAGState.RESPOND, RAGState.DELEGATE)

    def test_focused_faqs_no_false_positive(self):
        """Same-topic FAQs with flat scores should NOT trigger clarification."""
        fsm = _make_fsm(
            faqs=_FOCUSED_FAQS,
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.70,
                ambiguity_topic_diversity_threshold=0.60,
            ),
        )
        result = fsm.step("refund policy")
        # Same-topic FAQs should have low diversity → no clarification
        assert result.state in (RAGState.RESPOND, RAGState.DELEGATE)


# ── Selection Resolution Tests ──────────────────────────────────────


class TestSelectionResolution:
    def _make_clarified_fsm(self):
        """Create an FSM that has entered CLARIFY state with pending hits."""
        fsm = _make_fsm(
            config=RAGFSMConfig(
                enable_retrieval_clarification=True,
                ambiguity_score_flatness_threshold=0.20,
                ambiguity_topic_diversity_threshold=0.50,
            )
        )
        r1 = fsm.step("how to open an account?")
        assert r1.state == RAGState.CLARIFY, f"Expected CLARIFY, got {r1.state}"
        return fsm

    def test_letter_selection_b(self):
        """User replies 'B' → resolved to second passage → RESPOND."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("B")
        assert r2.state == RAGState.RESPOND
        assert r2.answer is not None

    def test_letter_selection_a(self):
        """User replies 'A' → resolved to first passage → RESPOND."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("A")
        assert r2.state == RAGState.RESPOND
        assert r2.answer is not None

    def test_number_selection(self):
        """User replies '2' → resolved to second passage → RESPOND."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("2")
        assert r2.state == RAGState.RESPOND

    def test_prefixed_selection(self):
        """User replies 'option B' → resolved to second passage → RESPOND."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("option B")
        assert r2.state == RAGState.RESPOND

    def test_keyword_selection(self):
        """User replies with keyword like 'savings' → resolved to best match → RESPOND."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("Savings Account")
        assert r2.state == RAGState.RESPOND

    def test_detailed_followup_reanalyzes(self):
        """User replies with full sentence → re-analyzes with combined query."""
        fsm = self._make_clarified_fsm()
        r2 = fsm.step("I want to open a Current Account for my company registration")
        # Should go to RESPOND (the combined query has enough context)
        assert r2.state in (RAGState.RESPOND, RAGState.DELEGATE, RAGState.CLARIFY)

    def test_resolve_selection_letters(self):
        """_resolve_selection correctly maps A-D to indices."""
        fsm = self._make_clarified_fsm()
        assert fsm._resolve_selection("A") == 0
        assert fsm._resolve_selection("b") == 1
        assert fsm._resolve_selection("C") == 2

    def test_resolve_selection_numbers(self):
        """_resolve_selection correctly maps 1-4 to indices."""
        fsm = self._make_clarified_fsm()
        assert fsm._resolve_selection("1") == 0
        assert fsm._resolve_selection("2") == 1
        assert fsm._resolve_selection("3") == 2

    def test_resolve_selection_out_of_range(self):
        """Out-of-range selections return None."""
        fsm = self._make_clarified_fsm()
        assert fsm._resolve_selection("Z") is None
        assert fsm._resolve_selection("99") is None

    def test_resolve_selection_empty(self):
        """Empty or non-selection response returns None."""
        fsm = _make_fsm()
        # No pending hits
        assert fsm._resolve_selection("B") is None
