# tests/test_faq_rag_agent.py
"""
Happy-path tests for the FAQ domain agent (faq_agent).

The FAQ agent is a ReACT domain agent backed by BankFAQs.csv.
It retrieves FAQ entries via TF-IDF (+ optional dense retrieval) and
responds with customer-facing answers.

Requires: generated/faq_agent/ artifacts from a factory deploy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = REPO_ROOT / "generated" / "faq_agent"
AGENT_PATH = AGENT_DIR / "agent.py"
CORPUS_PATH = AGENT_DIR / "corpus.json"

pytestmark = pytest.mark.skipif(
    not AGENT_PATH.exists(),
    reason="faq_agent artifacts not found — run factory deploy first",
)


# ---------------------------------------------------------------------------
# Fixture: load the generated agent once per module
# ---------------------------------------------------------------------------
def _load_agent():
    spec = importlib.util.spec_from_file_location("_test_faq_agent", AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_faq_agent"] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


@pytest.fixture(scope="module")
def faq_agent():
    return _load_agent()


# ---------------------------------------------------------------------------
# Loading & configuration
# ---------------------------------------------------------------------------
class TestFAQAgentLoading:
    def test_loads_without_error(self, faq_agent):
        assert faq_agent.ready is True

    def test_has_corpus(self):
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        assert len(corpus) > 0, "Corpus should have FAQ entries"

    def test_metadata_id(self, faq_agent):
        meta = faq_agent.metadata()
        assert meta["id"] == "faq_agent"

    def test_metadata_ready(self, faq_agent):
        meta = faq_agent.metadata()
        assert meta["ready"] is True

    def test_metadata_domain(self, faq_agent):
        meta = faq_agent.metadata()
        assert meta["domain"] == "faq"

    def test_metadata_has_capabilities(self, faq_agent):
        meta = faq_agent.metadata()
        caps = meta.get("capabilities", [])
        assert len(caps) > 0, "Metadata should list capabilities"


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------
class TestFAQAgentHandle:
    def test_returns_non_empty_answer(self, faq_agent):
        result = faq_agent.handle({"query": "How do I transfer money?"})
        assert result.get("answer"), f"Expected non-empty answer, got: {result}"

    def test_returns_domain_faq(self, faq_agent):
        result = faq_agent.handle({"query": "What is a fixed deposit?"})
        assert result.get("domain") == "faq"

    def test_returns_agent_id(self, faq_agent):
        result = faq_agent.handle({"query": "How do I open an account?"})
        assert result.get("agent_id") == "faq_agent"

    def test_has_react_trace(self, faq_agent):
        result = faq_agent.handle({"query": "What are the savings account features?"})
        assert "react_trace" in result
        assert len(result["react_trace"]) >= 1

    def test_knowledge_retrieved(self, faq_agent):
        result = faq_agent.handle({"query": "How do I reset my password?"})
        assert result.get("knowledge_retrieved") is True

    def test_text_key_accepted(self, faq_agent):
        result = faq_agent.handle({"text": "What is a debit card?"})
        assert result.get("answer"), "Should accept 'text' key"

    def test_empty_query_returns_prompt(self, faq_agent):
        result = faq_agent.handle({"query": ""})
        assert result.get("answer"), "Should return a prompt for empty query"
