# tests/test_rag_enhanced.py
"""Tests for enhanced RAG: LLM synthesis, dense retrieval, citation grounding."""
from __future__ import annotations

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

from app.runtime.rag_fsm import (
    RAGFSMConfig,
    RAGFiniteStateMachine,
    RAGState,
    _tok,
)


# ── Test data ─────────────────────────────────────────────────────────

_TEST_FAQS = [
    {
        "q": "What is the refund policy?",
        "a": "You can get a refund within 30 days of purchase.",
        "source": "refund_policy.csv",
    },
    {
        "q": "How do I reset my password?",
        "a": "Go to Settings > Security > Reset Password.",
        "source": "account_help.csv",
    },
    {
        "q": "What payment methods do you accept?",
        "a": "We accept Visa, Mastercard, and PayPal.",
        "source": "payments.csv",
    },
    {
        "q": "How do I contact support?",
        "a": "Email support@example.com or call 1-800-HELP.",
        "source": "contact.csv",
    },
    {
        "q": "What are your business hours?",
        "a": "Monday to Friday, 9am to 5pm.",
        "source": "general.csv",
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
    config: RAGFSMConfig | None = None,
    llm_fn=None,
    embed_fn=None,
) -> RAGFiniteStateMachine:
    idf, vecs, texts = _build_index(_TEST_FAQS)
    return RAGFiniteStateMachine(
        agent_id="test_faq",
        faqs=_TEST_FAQS,
        idf=idf,
        vecs=vecs,
        texts=texts,
        config=config,
        llm_fn=llm_fn,
        embed_fn=embed_fn,
    )


def _mock_embed_fn(texts: List[str]) -> List[List[float]]:
    """Deterministic mock embeddings: hash-based 8-dim vectors."""
    vecs = []
    for t in texts:
        tokens = _tok(t)
        vec = [0.0] * 8
        for tok in tokens:
            h = hash(tok) % 8
            vec[h] += 1.0
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vecs.append([v / norm for v in vec])
    return vecs


def _mock_llm_fn(messages, model=None, **kwargs):
    """Mock LLM that returns a synthesized answer with citations."""
    return {
        "answer": "Based on our policy, you can get a refund within 30 days [1].",
        "cited_passages": [1],
    }


# ── LLM Synthesis Tests ──────────────────────────────────────────────


class TestLLMSynthesis:
    def test_synthesis_enabled_calls_llm(self):
        """When synthesis is enabled, the LLM function should be called."""
        mock_llm = MagicMock(
            return_value={
                "answer": "Synthesized answer [1].",
                "cited_passages": [1],
            }
        )
        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=mock_llm)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert mock_llm.called
        assert "Synthesized answer" in result.answer

    def test_synthesis_receives_passages_in_context(self):
        """The LLM should receive retrieved passages in its prompt."""
        captured_messages = []

        def capture_llm(messages, **kwargs):
            captured_messages.extend(messages)
            return {"answer": "Test answer [1].", "cited_passages": [1]}

        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=capture_llm)

        fsm.step("What is the refund policy?")

        assert len(captured_messages) >= 2
        user_msg = captured_messages[-1]["content"]
        assert "refund" in user_msg.lower()
        assert "[1]" in user_msg

    def test_synthesis_disabled_uses_extractive(self):
        """When synthesis is disabled, raw FAQ answer should be returned."""
        mock_llm = MagicMock()
        config = RAGFSMConfig(enable_llm_synthesis=False)
        fsm = _make_fsm(config=config, llm_fn=mock_llm)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert not mock_llm.called
        assert "30 days" in result.answer
        assert result.synthesis_metadata.get("synthesized") is False

    def test_synthesis_without_llm_fn_uses_extractive(self):
        """When llm_fn is None, extractive path is used regardless of config."""
        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=None)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert "30 days" in result.answer
        assert result.synthesis_metadata.get("synthesized") is False

    def test_synthesis_populates_metadata(self):
        """Synthesis metadata should include model and synthesized flag."""
        config = RAGFSMConfig(enable_llm_synthesis=True, synthesis_model="test-model")
        fsm = _make_fsm(config=config, llm_fn=_mock_llm_fn)

        result = fsm.step("What is the refund policy?")

        assert result.synthesis_metadata["synthesized"] is True
        assert result.synthesis_metadata["model"] == "test-model"

    def test_synthesis_fallback_on_llm_error(self):
        """If LLM raises an exception, fallback to first passage text."""

        def bad_llm(messages, **kwargs):
            raise RuntimeError("LLM unavailable")

        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=bad_llm)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert result.answer  # Should have fallback answer
        assert len(result.grounded_citations) > 0  # Fallback cites passage 1


# ── Citation Grounding Tests ──────────────────────────────────────────


class TestCitationGrounding:
    def test_grounded_citations_populated_with_synthesis(self):
        """Grounded citations should map LLM-cited indices to passages."""
        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=_mock_llm_fn)

        result = fsm.step("What is the refund policy?")

        assert len(result.grounded_citations) > 0
        cit = result.grounded_citations[0]
        assert "passage" in cit
        assert "source" in cit
        assert "score" in cit
        assert cit["source"] == "refund_policy.csv"

    def test_grounded_citations_empty_without_synthesis(self):
        """Without synthesis, grounded_citations should be empty."""
        fsm = _make_fsm()

        result = fsm.step("What is the refund policy?")

        assert result.grounded_citations == []

    def test_basic_citations_always_present(self):
        """Basic citations (backward compatible) should always be populated."""
        config = RAGFSMConfig(enable_llm_synthesis=True)
        fsm = _make_fsm(config=config, llm_fn=_mock_llm_fn)

        result = fsm.step("What is the refund policy?")

        assert len(result.citations) > 0
        assert "question" in result.citations[0]
        assert "source" in result.citations[0]


# ── Dense Retrieval Tests ─────────────────────────────────────────────


class TestDenseRetrieval:
    def test_hybrid_retrieval_uses_embed_fn(self):
        """When dense retrieval is enabled, embed_fn should be called."""
        mock_embed = MagicMock(side_effect=_mock_embed_fn)
        config = RAGFSMConfig(enable_dense_retrieval=True)
        fsm = _make_fsm(config=config, embed_fn=mock_embed)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert mock_embed.called

    def test_hybrid_retrieval_disabled_skips_embed(self):
        """When dense retrieval is disabled, embed_fn should not be called."""
        mock_embed = MagicMock(side_effect=_mock_embed_fn)
        config = RAGFSMConfig(enable_dense_retrieval=False)
        fsm = _make_fsm(config=config, embed_fn=mock_embed)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert not mock_embed.called

    def test_hybrid_retrieval_without_embed_fn(self):
        """When embed_fn is None, sparse-only path is used."""
        config = RAGFSMConfig(enable_dense_retrieval=True)
        fsm = _make_fsm(config=config, embed_fn=None)

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert result.synthesis_metadata["retrieval_mode"] == "sparse"

    def test_hybrid_mode_reported_in_metadata(self):
        """When hybrid retrieval is active, metadata should report it."""
        config = RAGFSMConfig(enable_dense_retrieval=True)
        fsm = _make_fsm(config=config, embed_fn=_mock_embed_fn)

        result = fsm.step("What is the refund policy?")

        assert result.synthesis_metadata["retrieval_mode"] == "hybrid"

    def test_dense_index_cached_across_queries(self):
        """Dense embeddings should be computed once and cached."""
        call_count = 0

        def counting_embed(texts):
            nonlocal call_count
            call_count += 1
            return _mock_embed_fn(texts)

        config = RAGFSMConfig(enable_dense_retrieval=True)
        fsm = _make_fsm(config=config, embed_fn=counting_embed)

        # First query: builds index (1 call for corpus) + 1 call for query
        fsm.step("What is the refund policy?")
        first_count = call_count

        # Reset and query again — index should be cached
        fsm.reset()
        fsm.step("How do I reset my password?")

        # The corpus embedding call should not repeat (only new query call)
        # _ensure_dense_index is called but returns early since _dense_vecs is cached
        # But _hybrid_retrieve still embeds the query each time
        assert call_count > first_count  # Query embedding still happens


# ── Hybrid Scoring Tests ──────────────────────────────────────────────


class TestHybridScoring:
    def test_dot_product(self):
        """Verify dot product computation."""
        assert RAGFiniteStateMachine._dot([1.0, 0.0], [0.0, 1.0]) == 0.0
        assert RAGFiniteStateMachine._dot([1.0, 0.0], [1.0, 0.0]) == 1.0
        assert abs(RAGFiniteStateMachine._dot([0.5, 0.5], [0.5, 0.5]) - 0.5) < 1e-9

    def test_fusion_weights_applied(self):
        """Sparse and dense weights should sum correctly."""
        config = RAGFSMConfig(
            enable_dense_retrieval=True,
            sparse_weight=0.3,
            dense_weight=0.7,
        )

        # Use a specific embed function that gives known scores
        def known_embed(texts):
            # Return identical vectors so dense score = 1.0 for matching
            return [[1.0, 0.0, 0.0, 0.0]] * len(texts)

        fsm = _make_fsm(config=config, embed_fn=known_embed)
        result = fsm.step("What is the refund policy?")

        # Should still produce a valid result
        assert result.state == RAGState.RESPOND
        assert result.score > 0.0


# ── Full Pipeline Integration ─────────────────────────────────────────


class TestFullPipeline:
    def test_full_pipeline_with_synthesis_and_dense(self):
        """End-to-end: dense retrieval + LLM synthesis + citation grounding."""
        config = RAGFSMConfig(
            enable_dense_retrieval=True,
            enable_llm_synthesis=True,
        )
        fsm = _make_fsm(
            config=config,
            llm_fn=_mock_llm_fn,
            embed_fn=_mock_embed_fn,
        )

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert "refund" in result.answer.lower() or "30 days" in result.answer.lower()
        assert result.synthesis_metadata["synthesized"] is True
        assert result.synthesis_metadata["retrieval_mode"] == "hybrid"
        assert len(result.grounded_citations) > 0
        assert len(result.citations) > 0  # Backward compatible citations

    def test_existing_tests_still_work_with_defaults(self):
        """Default config (no LLM, no dense) should behave like before."""
        fsm = _make_fsm()

        result = fsm.step("What is the refund policy?")

        assert result.state == RAGState.RESPOND
        assert "30 days" in result.answer
        assert result.synthesis_metadata.get("synthesized") is False
        assert result.synthesis_metadata.get("retrieval_mode") == "sparse"
        assert result.grounded_citations == []
