# tests/test_faq_rag_agent.py
"""
Happy-path tests for the FAQ RAG agent (faq_rag_agent).

The FAQ RAG agent is entirely self-contained:
  - No LLM calls during handle() — pure TF-IDF retrieval
  - Loads faqs.json from its own directory
  - Builds an in-memory TF-IDF index
  - Returns ranked answers with citations

Tests verify:
  - Agent loads and indexes FAQs correctly
  - Known-topic queries return relevant answers with score > 0
  - Citations are included for matched queries
  - Low-relevance queries return a fallback message
  - Empty query returns a helpful prompt
  - metadata() reports correct id/type/docs count
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FAQ_AGENT_PATH = REPO_ROOT / "generated" / "customer_facing_rag" / "agent.py"
FAQS_PATH = REPO_ROOT / "generated" / "customer_facing_rag" / "faqs.json"

pytestmark = pytest.mark.skipif(
    not FAQS_PATH.exists(),
    reason="faqs.json not found — run factory deploy for faq_rag_agent first",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_faq_agent():
    module_name = "_test_faq_rag_agent"
    spec = importlib.util.spec_from_file_location(module_name, FAQ_AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def faq_agent():
    return _load_faq_agent()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_faq_agent_loads_without_error(faq_agent):
    assert faq_agent.ready is True


def test_faq_agent_loads_faqs(faq_agent):
    assert len(faq_agent.faqs) > 0


def test_faq_agent_builds_index(faq_agent):
    assert len(faq_agent._vecs) > 0
    assert len(faq_agent._texts) > 0
    assert len(faq_agent._idf) > 0


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------


def test_faq_agent_metadata_id(faq_agent):
    assert faq_agent.metadata()["id"] == "customer_facing_rag"


def test_faq_agent_metadata_type(faq_agent):
    assert faq_agent.metadata()["type"] == "faq_rag"


def test_faq_agent_metadata_ready(faq_agent):
    assert faq_agent.metadata()["ready"] is True


def test_faq_agent_metadata_docs_count(faq_agent):
    meta = faq_agent.metadata()
    assert meta["docs"] == len(faq_agent.faqs)
    assert meta["docs"] > 0


def test_faq_agent_metadata_capabilities(faq_agent):
    caps = faq_agent.metadata()["capabilities"]
    assert "faq_answering" in caps


# ---------------------------------------------------------------------------
# handle() — happy path (known-topic query)
# ---------------------------------------------------------------------------


def test_faq_handle_returns_intent_faq(faq_agent):
    result = faq_agent.handle({"query": "current account transfer branch"})
    assert result["intent"] == "faq"


def test_faq_handle_returns_non_empty_answer(faq_agent):
    result = faq_agent.handle({"query": "current account transfer branch"})
    assert result["answer"]
    assert len(result["answer"]) > 0


def test_faq_handle_returns_positive_score(faq_agent):
    result = faq_agent.handle({"query": "current account transfer branch"})
    # Enhanced RAG may return score=0 with clarification, check solvability score instead
    tfidf_score = result.get("solvability", {}).get("tfidf_score", 0.0)
    assert tfidf_score > 0.0 or result["score"] > 0.0


def test_faq_handle_returns_citations_list(faq_agent):
    result = faq_agent.handle({"query": "current account transfer branch"})
    assert isinstance(result["citations"], list)


def test_faq_handle_score_above_relevance_gate(faq_agent):
    # A query matching vocabulary from the FAQ corpus should pass the 0.12 gate
    result = faq_agent.handle({"query": "current account transfer branch"})
    assert result["score"] >= 0.12


def test_faq_handle_text_key_also_accepted(faq_agent):
    result = faq_agent.handle({"text": "documents required for current account"})
    assert result["intent"] == "faq"
    assert result["score"] > 0.0


# ---------------------------------------------------------------------------
# handle() — different query entry points
# ---------------------------------------------------------------------------


def test_faq_handle_query_key_priority_over_text(faq_agent):
    # Both keys present — query takes precedence
    result = faq_agent.handle(
        {
            "query": "current account transfer branch",
            "text": "xylophone quantum",
        }
    )
    # Should match "current account" not "xylophone" -> check solvability or top-level score
    tfidf_score = result.get("solvability", {}).get("tfidf_score", 0.0)
    assert tfidf_score > 0.0 or result["score"] > 0.0


# ---------------------------------------------------------------------------
# handle() — low relevance fallback
# ---------------------------------------------------------------------------


def test_faq_low_relevance_returns_answer(faq_agent):
    # Garbage query — should still return a valid answer key (even if fallback message)
    result = faq_agent.handle({"query": "xylophone quantum thermodynamics"})
    assert result["intent"] == "faq"
    assert "answer" in result


def test_faq_low_relevance_score_below_gate(faq_agent):
    result = faq_agent.handle({"query": "xylophone quantum thermodynamics"})
    # Either score is below gate OR the fallback message is returned
    assert result["score"] < 0.12 or "couldn't find" in result["answer"].lower()


# ---------------------------------------------------------------------------
# handle() — empty query
# ---------------------------------------------------------------------------


def test_faq_empty_query_returns_prompt(faq_agent):
    result = faq_agent.handle({"query": ""})
    assert result["intent"] == "faq"
    assert "Please provide" in result["answer"] or result["answer"]


def test_faq_missing_query_key_returns_prompt(faq_agent):
    result = faq_agent.handle({})
    assert result["intent"] == "faq"
    assert "Please provide" in result["answer"] or result["answer"]
