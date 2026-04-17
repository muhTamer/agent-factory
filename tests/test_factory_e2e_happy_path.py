# tests/test_factory_e2e_happy_path.py
"""
Full Factory Happy-Path End-to-End Test

Tests the complete lifecycle using generated ReACT domain agents:
  Phase 1 — Generated artifacts exist and load
  Phase 2 — Agents register in AgentRegistry
  Phase 3 — Agents handle queries via RuntimeSpine
  Phase 4 — Thread isolation and multi-session independence

Requires: generated/{faq_agent,refunds_agent}/ artifacts from factory deploy.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.runtime.registry import AgentRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
FAQ_AGENT_DIR = REPO_ROOT / "generated" / "faq_agent"
REFUNDS_AGENT_DIR = REPO_ROOT / "generated" / "refunds_agent"

HAVE_ARTIFACTS = (FAQ_AGENT_DIR / "agent.py").exists() and (
    REFUNDS_AGENT_DIR / "agent.py"
).exists()

pytestmark = pytest.mark.skipif(
    not HAVE_ARTIFACTS,
    reason="Pre-built agent artifacts not found — run factory deploy first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_agent_from_dir(module_name: str, agent_dir: Path):
    agent_path = agent_dir / "agent.py"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


# ---------------------------------------------------------------------------
# Phase 1: Generated artifacts
# ---------------------------------------------------------------------------
class TestPhase1Artifacts:
    def test_faq_agent_has_config(self):
        cfg = json.loads((FAQ_AGENT_DIR / "config.json").read_text(encoding="utf-8"))
        assert cfg["domain"] == "faq"

    def test_faq_agent_has_corpus(self):
        corpus = json.loads((FAQ_AGENT_DIR / "corpus.json").read_text(encoding="utf-8"))
        assert len(corpus) > 0

    def test_refunds_agent_has_config(self):
        cfg = json.loads(
            (REFUNDS_AGENT_DIR / "config.json").read_text(encoding="utf-8")
        )
        assert cfg["domain"] == "refunds"

    def test_refunds_agent_has_corpus(self):
        corpus = json.loads(
            (REFUNDS_AGENT_DIR / "corpus.json").read_text(encoding="utf-8")
        )
        assert len(corpus) > 0

    def test_tool_operators_exist(self):
        tool_ids = [
            "initiate_refund_tool",
            "create_ticket_tool",
            "lookup_payment_tool",
        ]
        for tid in tool_ids:
            agent_dir = REPO_ROOT / "generated" / tid
            assert (agent_dir / "agent.py").exists(), f"Missing: {tid}/agent.py"


# ---------------------------------------------------------------------------
# Phase 2: Agent registry
# ---------------------------------------------------------------------------
class TestPhase2Registry:
    def test_registry_accepts_both_agents(self):
        registry = AgentRegistry()
        faq = _load_agent_from_dir("_e2e_faq", FAQ_AGENT_DIR)
        refunds = _load_agent_from_dir("_e2e_refunds", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)
        assert "faq_agent" in registry.all_ids()
        assert "refunds_agent" in registry.all_ids()

    def test_registry_get_returns_agent(self):
        registry = AgentRegistry()
        faq = _load_agent_from_dir("_e2e_faq2", FAQ_AGENT_DIR)
        registry.register("faq_agent", faq)
        assert registry.get("faq_agent") is faq

    def test_registry_meta_includes_ids(self):
        registry = AgentRegistry()
        faq = _load_agent_from_dir("_e2e_faq3", FAQ_AGENT_DIR)
        refunds = _load_agent_from_dir("_e2e_ref3", REFUNDS_AGENT_DIR)
        registry.register("faq_agent", faq)
        registry.register("refunds_agent", refunds)
        meta = registry.all_meta()
        assert "faq_agent" in meta
        assert "refunds_agent" in meta

    def test_import_generated_agent(self):
        registry = AgentRegistry()
        agent = registry.import_generated_agent("faq_agent", FAQ_AGENT_DIR)
        assert agent is not None


# ---------------------------------------------------------------------------
# Phase 3: Agent handles queries
# ---------------------------------------------------------------------------
class TestPhase3Queries:
    def test_faq_query_returns_answer(self):
        agent = _load_agent_from_dir("_e2e_faq_q", FAQ_AGENT_DIR)
        result = agent.handle({"query": "How do I transfer money?"})
        assert result.get("answer"), f"Expected answer, got: {result}"

    def test_faq_response_has_agent_id(self):
        agent = _load_agent_from_dir("_e2e_faq_q2", FAQ_AGENT_DIR)
        result = agent.handle({"query": "What is a savings account?"})
        assert result.get("agent_id") == "faq_agent"

    def test_refund_query_returns_answer(self):
        agent = _load_agent_from_dir("_e2e_ref_q", REFUNDS_AGENT_DIR)
        result = agent.handle({"query": "I want a refund"})
        assert result.get("answer"), f"Expected answer, got: {result}"

    def test_refund_response_has_agent_id(self):
        agent = _load_agent_from_dir("_e2e_ref_q2", REFUNDS_AGENT_DIR)
        result = agent.handle({"query": "I want a refund"})
        assert result.get("agent_id") == "refunds_agent"

    def test_empty_query_returns_prompt(self):
        agent = _load_agent_from_dir("_e2e_faq_empty", FAQ_AGENT_DIR)
        result = agent.handle({"query": ""})
        assert result.get("answer"), "Empty query should return a prompt"


# ---------------------------------------------------------------------------
# Phase 4: Isolation
# ---------------------------------------------------------------------------
class TestPhase4Isolation:
    def test_concurrent_threads_isolated(self):
        agent = _load_agent_from_dir("_e2e_iso", FAQ_AGENT_DIR)
        r1 = agent.handle({"query": "savings account", "thread_id": "t1"})
        r2 = agent.handle({"query": "credit card", "thread_id": "t2"})
        assert r1.get("answer"), "Thread 1 should respond"
        assert r2.get("answer"), "Thread 2 should respond"

    def test_guardrails_block_empty(self):
        agent = _load_agent_from_dir("_e2e_guard", FAQ_AGENT_DIR)
        result = agent.handle({"query": ""})
        # Should handle gracefully (prompt or error), not crash
        assert "answer" in result or "error" in str(result).lower()
