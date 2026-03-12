# tests/test_domain_agent_builder.py
"""Tests for the Domain Agent builder (app.shared.domain_agent.build_agent)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.shared.domain_agent import build_agent

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def gen_dir(tmp_path: Path) -> Path:
    return tmp_path / "generated" / "test_agent"


@pytest.fixture
def sample_kb(tmp_path: Path) -> Path:
    """Create a sample CSV knowledge base file."""
    kb = tmp_path / "faq.csv"
    kb.write_text(
        "question,answer\n"
        "What is the refund policy?,Refunds within 30 days.\n"
        "How to reset password?,Go to Settings > Security.\n",
        encoding="utf-8",
    )
    return kb


@pytest.fixture
def sample_policy(tmp_path: Path) -> Path:
    """Create a sample YAML policy file."""
    pol = tmp_path / "refund_policy.yaml"
    pol.write_text(
        "policy:\n"
        "  name: refund_eligibility\n"
        "  rules:\n"
        "    - condition: days_since_purchase <= 30\n"
        "      result: eligible\n",
        encoding="utf-8",
    )
    return pol


# ── Tests ────────────────────────────────────────────────────────────


class TestBuildAgent:
    """Test build_agent generates the expected files."""

    def test_generates_config_corpus_agent(self, gen_dir: Path, sample_kb: Path):
        """Build agent with knowledge sources → generates 3 files."""
        inputs = {
            "domain": "refunds",
            "goal": "Help customers with refund requests",
            "knowledge_sources": [str(sample_kb)],
            "available_tools": ["lookup_payment", "initiate_refund"],
            "policies": ["Refunds within 30 days only"],
        }

        result = build_agent("test_refund_agent", inputs, gen_dir)

        assert result == gen_dir
        assert (gen_dir / "config.json").exists()
        assert (gen_dir / "corpus.json").exists()
        assert (gen_dir / "agent.py").exists()

    def test_config_contents(self, gen_dir: Path, sample_kb: Path):
        """Verify config.json has correct structure."""
        inputs = {
            "domain": "refunds",
            "goal": "Help customers with refund requests",
            "knowledge_sources": [str(sample_kb)],
            "available_tools": ["lookup_payment"],
            "policies": ["Verify identity first"],
            "max_steps": 3,
        }

        build_agent("refund_agent", inputs, gen_dir)

        cfg = json.loads((gen_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["id"] == "refund_agent"
        assert cfg["domain"] == "refunds"
        assert cfg["goal"] == "Help customers with refund requests"
        assert cfg["available_tools"] == ["lookup_payment"]
        assert cfg["policies"] == ["Verify identity first"]
        assert cfg["max_steps"] == 3

    def test_corpus_from_csv(self, gen_dir: Path, sample_kb: Path):
        """Verify corpus.json is populated from CSV knowledge source."""
        inputs = {
            "domain": "faq",
            "goal": "Answer FAQs",
            "knowledge_sources": [str(sample_kb)],
        }

        build_agent("faq_agent", inputs, gen_dir)

        corpus = json.loads((gen_dir / "corpus.json").read_text(encoding="utf-8"))
        assert isinstance(corpus, list)
        assert len(corpus) > 0
        # Each item should have text, source, kind, meta
        for item in corpus:
            assert "text" in item
            assert "source" in item
            assert "kind" in item

    def test_empty_knowledge_sources(self, gen_dir: Path):
        """Build agent with no knowledge sources → empty corpus."""
        inputs = {
            "domain": "general",
            "goal": "General assistant",
            "knowledge_sources": [],
            "available_tools": [],
        }

        build_agent("empty_agent", inputs, gen_dir)

        corpus = json.loads((gen_dir / "corpus.json").read_text(encoding="utf-8"))
        assert corpus == []

    def test_agent_py_is_valid_python(self, gen_dir: Path):
        """Generated agent.py should be syntactically valid Python."""
        inputs = {
            "domain": "orders",
            "goal": "Help with order tracking",
        }

        build_agent("order_agent", inputs, gen_dir)

        agent_src = (gen_dir / "agent.py").read_text(encoding="utf-8")
        # Should compile without syntax errors
        compile(agent_src, "agent.py", "exec")

    def test_agent_py_contains_class(self, gen_dir: Path):
        """Generated agent.py should contain an Agent class implementing IAgent."""
        inputs = {
            "domain": "accounts",
            "goal": "Help with account management",
        }

        build_agent("account_agent", inputs, gen_dir)

        agent_src = (gen_dir / "agent.py").read_text(encoding="utf-8")
        assert "class Agent(IAgent):" in agent_src
        assert "def load(" in agent_src
        assert "def handle(" in agent_src
        assert "def metadata(" in agent_src
        assert "DomainAgentEngine" in agent_src

    def test_defaults_applied(self, gen_dir: Path):
        """Missing optional inputs get sensible defaults."""
        inputs = {
            "domain": "general",
            "goal": "Help customers",
        }

        build_agent("default_agent", inputs, gen_dir)

        cfg = json.loads((gen_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["max_steps"] == 5
        assert cfg["model"] == "gpt-5-mini"
        assert cfg["available_tools"] == []
        assert cfg["policies"] == []
        assert cfg["knowledge_sources"] == []

    def test_backward_compat_docs_key(self, gen_dir: Path, sample_kb: Path):
        """'docs' key should work as alias for 'knowledge_sources'."""
        inputs = {
            "domain": "faq",
            "goal": "FAQ bot",
            "docs": [str(sample_kb)],
        }

        build_agent("compat_agent", inputs, gen_dir)

        corpus = json.loads((gen_dir / "corpus.json").read_text(encoding="utf-8"))
        assert len(corpus) > 0

    def test_string_inputs_coerced_to_list(self, gen_dir: Path, sample_kb: Path):
        """String inputs for list fields should be auto-wrapped."""
        inputs = {
            "domain": "faq",
            "goal": "FAQ bot",
            "knowledge_sources": str(sample_kb),
            "available_tools": "lookup_payment",
            "policies": "Be polite",
        }

        build_agent("coerce_agent", inputs, gen_dir)

        cfg = json.loads((gen_dir / "config.json").read_text(encoding="utf-8"))
        assert isinstance(cfg["available_tools"], list)
        assert cfg["available_tools"] == ["lookup_payment"]
        assert isinstance(cfg["policies"], list)
        assert cfg["policies"] == ["Be polite"]


class TestBuildAgentMetadata:
    """Test the metadata() output from generated agents."""

    def test_agent_py_metadata_fields(self, gen_dir: Path):
        """Generated agent.py metadata() should include domain agent fields."""
        inputs = {
            "domain": "refunds",
            "goal": "Handle refund requests",
            "available_tools": ["lookup_payment", "initiate_refund"],
        }

        build_agent("meta_agent", inputs, gen_dir)

        agent_src = (gen_dir / "agent.py").read_text(encoding="utf-8")
        assert '"type": "domain_agent"' in agent_src
        assert '"agent_kind": "domain_agent"' in agent_src
        assert "domain_agent" in agent_src
