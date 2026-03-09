# tests/test_e2e_scenarios.py
"""
End-to-end integration scenarios that hit the real LLM.

Test 1: FAQ refund policy — clarify → select → answer
Test 2: Multi-intent (documents + refund) — decompose → FAQ → refund action

Run:  pytest tests/test_e2e_scenarios.py -m integration -v
"""
from __future__ import annotations

import pytest
from typing import Dict, Any, List

from app.runtime.domain_agent_engine import DomainAgentEngine, DomainAgentConfig
from app.shared.rag import load_corpus, build_index
from app.llm_client import chat_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_real_engine(
    corpus_files: List[str],
    agent_id: str = "test_agent",
    domain: str = "faq",
    goal: str = "Answer customer questions from the FAQ knowledge base",
    policies: List[str] | None = None,
    tools: Dict[str, Any] | None = None,
    max_steps: int = 8,
) -> DomainAgentEngine:
    """Build a DomainAgentEngine with real LLM and real corpus."""
    corpus = load_corpus(corpus_files)
    index = build_index(corpus)

    # Try to load embeddings for hybrid retrieval
    embed_fn = None
    dense_vecs = None
    enable_dense = False
    try:
        from app.runtime.embeddings import get_embed_fn

        embed_fn = get_embed_fn()
        texts = [item.text for item in corpus]
        dense_vecs = embed_fn(texts)
        enable_dense = True
    except Exception:
        pass

    config = DomainAgentConfig(
        agent_id=agent_id,
        domain=domain,
        goal=goal,
        policies=policies or [],
        max_steps=max_steps,
        enable_dense_retrieval=enable_dense,
    )
    return DomainAgentEngine(
        config=config,
        index=index,
        tools=tools or {},
        llm_fn=chat_json,
        embed_fn=embed_fn,
        dense_vecs=dense_vecs,
    )


def _stub_tool(name: str, response: Dict[str, Any]):
    """Create a minimal stub tool."""
    from unittest.mock import MagicMock

    tool = MagicMock()
    tool.execute.return_value = response
    tool.describe.return_value = {"description": f"{name} tool"}
    return tool


# ===========================================================================
# TEST 1: FAQ Refund Policy — Clarify → Select → Answer
# ===========================================================================


@pytest.mark.integration
class TestFAQRefundPolicy:
    """
    User asks 'What is the refund policy?'
    → Agent retrieves refund-related FAQs
    → Agent asks clarifying question with options (travel, home, forex, etc.)
    → User selects an option
    → Agent provides the correct FAQ info
    → knowledge_sources populated in every response
    """

    def test_turn1_asks_clarifying_question(self):
        """Turn 1: 'What is the refund policy?' → clarifying question with options."""
        engine = _build_real_engine(
            corpus_files=["data/BankFAQs.csv"],
            agent_id="faq_agent",
            domain="faq",
            goal="Answer customer FAQ questions accurately from the knowledge base",
        )

        r1 = engine.handle("What is the refund policy?", thread_id="t1")

        # Must ask for clarification (multiple refund types in FAQs)
        assert (
            r1.get("needs_input") is True
        ), f"Should ask clarifying question, got: {r1.get('answer', '')[:200]}"

        # Must have retrieved knowledge
        assert r1.get("knowledge_retrieved") is True, "Should have retrieved knowledge"

        # knowledge_sources must be populated
        ks = r1.get("knowledge_sources", [])
        assert len(ks) > 0, f"knowledge_sources should be populated, got: {ks}"
        assert any(
            "BankFAQs" in s for src in ks for s in src.get("sources", [])
        ), f"Should reference BankFAQs.csv, got sources: {ks}"

        # Answer should mention at least 2 options (different refund types)
        answer = r1.get("answer", "").lower()
        option_keywords = ["insurance", "forex", "card", "cancel", "travel", "home"]
        matches = sum(1 for kw in option_keywords if kw in answer)
        assert (
            matches >= 2
        ), f"Clarifying question should mention multiple options, got: {r1.get('answer', '')[:300]}"

        print(f"\n[Turn 1 PASS] Agent asked: {r1['answer'][:200]}")

    def test_full_clarify_then_answer_flow(self):
        """Full flow: ask → clarify → user selects → specific answer."""
        engine = _build_real_engine(
            corpus_files=["data/BankFAQs.csv"],
            agent_id="faq_agent",
            domain="faq",
            goal="Answer customer FAQ questions accurately from the knowledge base",
        )

        # Turn 1: "What is the refund policy?"
        r1 = engine.handle("What is the refund policy?", thread_id="t2")

        # Must ask for clarification (multiple refund types in FAQs)
        assert r1.get("needs_input") is True, (
            f"Turn 1 MUST ask clarifying question when FAQs cover multiple "
            f"refund types (travel, home, forex). Got direct answer: "
            f"{r1.get('answer', '')[:200]}"
        )
        assert r1.get("knowledge_retrieved") is True, "Should have retrieved knowledge"

        # knowledge_sources must be populated
        ks1 = r1.get("knowledge_sources", [])
        assert len(ks1) > 0, f"Turn 1 knowledge_sources should be populated, got: {ks1}"

        print(f"\n[Turn 1] Agent asked: {r1['answer'][:200]}")

        # Turn 2: user selects home insurance
        r2 = engine.handle("Home insurance", thread_id="t2")

        # Must have retrieved knowledge again for specific topic
        assert r2.get("knowledge_retrieved") is True, "Turn 2 should retrieve knowledge"

        # knowledge_sources must be populated
        ks2 = r2.get("knowledge_sources", [])
        assert len(ks2) > 0, f"Turn 2 knowledge_sources should be populated, got: {ks2}"

        # Answer should contain home-insurance-specific FAQ info
        answer = r2.get("answer", "").lower()
        home_keywords = [
            "home",
            "insurance",
            "cancel",
            "refund",
            "premium",
            "policy",
            "sold",
            "transfer",
            "ownership",
            "pro-rata",
            "pro rata",
        ]
        matches = sum(1 for kw in home_keywords if kw in answer)
        assert (
            matches >= 2
        ), f"Answer should reference home insurance refund info, got: {r2.get('answer', '')[:300]}"

        # Should NOT leak internal info
        for forbidden in ["PSD2", "AMLD5", "GDPR", "AML/KYC", "novapay", "policy_id"]:
            assert (
                forbidden.lower() not in answer
            ), f"Answer must not contain internal term '{forbidden}': {r2.get('answer', '')[:300]}"

        print(f"\n[Turn 2] Agent answered: {r2['answer'][:300]}")

    def test_no_internal_policy_leakage_in_faq(self):
        """FAQ agent using BankFAQs should never reference internal policy docs."""
        engine = _build_real_engine(
            corpus_files=["data/BankFAQs.csv"],
            agent_id="faq_agent",
            domain="faq",
            goal="Answer customer FAQ questions accurately from the knowledge base",
        )

        r1 = engine.handle("What is the refund policy?", thread_id="t3")
        answer = r1.get("answer", "").lower()

        for term in ["refunds_policy.yaml", "psd2", "amld5", "aml", "kyc", "novapay"]:
            assert (
                term not in answer
            ), f"FAQ answer must not contain '{term}': {r1.get('answer', '')[:300]}"


# ===========================================================================
# TEST 2: Multi-Intent — Documents + Refund Action
# ===========================================================================


@pytest.mark.integration
class TestMultiIntentDocumentsAndRefund:
    """
    User asks about documents for opening account AND wants a refund.
    Tests the full multi-intent flow with two domain agents.
    """

    def test_faq_agent_handles_account_documents(self):
        """FAQ agent retrieves account opening document requirements."""
        engine = _build_real_engine(
            corpus_files=["data/BankFAQs.csv"],
            agent_id="faq_agent",
            domain="faq",
            goal="Answer customer FAQ questions accurately from the knowledge base",
        )

        r1 = engine.handle(
            "What are the required documents for opening an account?",
            thread_id="docs1",
        )

        # Should either give info or ask clarifying question (which account type?)
        answer = r1.get("answer", "").lower()
        assert r1.get("knowledge_retrieved") is True, "Should retrieve knowledge"

        # knowledge_sources populated
        ks = r1.get("knowledge_sources", [])
        assert len(ks) > 0, f"knowledge_sources should be populated, got: {ks}"

        # If asking clarification, should mention account types
        if r1.get("needs_input"):
            account_keywords = ["savings", "current", "fixed", "deposit", "account"]
            matches = sum(1 for kw in account_keywords if kw in answer)
            assert matches >= 1, f"Clarification should mention account types, got: {answer[:200]}"
            print(f"\n[Docs Turn 1] Agent asked: {r1['answer'][:200]}")
        else:
            # Direct answer — should mention documents
            doc_keywords = ["document", "id", "proof", "photo", "address", "pan", "kyc"]
            matches = sum(1 for kw in doc_keywords if kw in answer)
            assert matches >= 1, f"Answer should mention documents, got: {answer[:200]}"
            print(f"\n[Docs Turn 1] Agent answered: {r1['answer'][:200]}")

    def test_refund_agent_gradual_questioning(self):
        """Refund agent gathers info gradually for refund action."""
        engine = _build_real_engine(
            corpus_files=["data/refunds_policy.yaml"],
            agent_id="refunds_agent",
            domain="refunds",
            goal="Help customers with refund and reversal requests",
            tools={
                "lookup_payment": _stub_tool(
                    "lookup_payment",
                    {
                        "found": True,
                        "amount": 49.99,
                        "currency": "EUR",
                        "status": "completed",
                        "date": "2025-01-15",
                        "eligible": True,
                    },
                ),
                "verify_identity": _stub_tool(
                    "verify_identity",
                    {
                        "verified": True,
                        "name": "John Doe",
                    },
                ),
                "initiate_refund": _stub_tool(
                    "initiate_refund",
                    {
                        "refund_id": "REF-98765",
                        "status": "initiated",
                        "amount": 49.99,
                    },
                ),
            },
        )

        # Turn 1: User wants a refund (action request)
        r1 = engine.handle("I want to issue a refund", thread_id="ref1")

        # Should ask for transaction reference (first step per policy)
        assert (
            r1.get("needs_input") is True
        ), f"Should ask for transaction details, got: {r1.get('answer', '')[:200]}"
        answer1 = r1.get("answer", "").lower()
        assert any(
            kw in answer1 for kw in ["transaction", "order", "reference", "id"]
        ), f"Should ask for transaction reference, got: {r1.get('answer', '')[:200]}"

        # No internal leakage
        for term in ["psd2", "amld5", "novapay", "policy_id", "refunds_policy"]:
            assert term not in answer1, f"Must not contain '{term}': {answer1[:200]}"

        print(f"\n[Refund Turn 1] Agent asked: {r1['answer'][:200]}")

        # Turn 2: User provides transaction ID
        r2 = engine.handle("Order #12345", thread_id="ref1")

        # Agent should look up the payment (call_tool) and then ask for more info or process
        answer2 = r2.get("answer", "").lower()
        tools_used = r2.get("tools_used", [])

        # Should have used at least lookup_payment
        assert len(tools_used) >= 1 or r2.get("knowledge_retrieved"), (
            f"Should call tools or retrieve knowledge. Tools: {tools_used}, "
            f"answer: {answer2[:200]}"
        )

        print(f"\n[Refund Turn 2] Tools: {tools_used}, Answer: {r2['answer'][:200]}")

        # Verify knowledge_sources is always populated when knowledge was retrieved
        for turn_name, resp in [("Turn 1", r1), ("Turn 2", r2)]:
            if resp.get("knowledge_retrieved"):
                ks = resp.get("knowledge_sources", [])
                assert (
                    len(ks) > 0
                ), f"{turn_name}: knowledge_sources should be populated when knowledge_retrieved=True"

    def test_explainability_sources_in_every_response(self):
        """Every response that retrieves knowledge must have knowledge_sources."""
        engine = _build_real_engine(
            corpus_files=["data/BankFAQs.csv"],
            agent_id="faq_agent",
            domain="faq",
            goal="Answer customer FAQ questions accurately from the knowledge base",
        )

        r1 = engine.handle("What is the refund policy?", thread_id="expl1")

        # react_trace should show retrieve_knowledge action
        trace = r1.get("react_trace", [])
        retrieve_steps = [s for s in trace if s.get("action") == "retrieve_knowledge"]
        assert len(retrieve_steps) > 0, (
            f"Should have retrieve_knowledge in trace, got actions: "
            f"{[s.get('action') for s in trace]}"
        )

        # knowledge_sources must have query, sources, and passages
        ks = r1.get("knowledge_sources", [])
        assert len(ks) > 0, "knowledge_sources must be populated"
        for entry in ks:
            assert "query" in entry, f"knowledge_source entry needs 'query': {entry}"
            assert "sources" in entry, f"knowledge_source entry needs 'sources': {entry}"
            assert len(entry["sources"]) > 0, f"sources list should not be empty: {entry}"
            assert "passages" in entry, f"knowledge_source entry needs 'passages': {entry}"
            assert len(entry["passages"]) > 0, f"passages list should not be empty: {entry}"

        print(f"\n[Explainability PASS] Sources: {ks[0].get('sources')}")
        print(f"  Query: {ks[0].get('query')}")
        print(f"  Passages: {len(ks[0].get('passages', []))} entries")
