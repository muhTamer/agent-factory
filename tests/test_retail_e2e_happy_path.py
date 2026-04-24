# tests/test_retail_e2e_happy_path.py
"""
Retail Quickstart Full Happy-Path End-to-End Test

This test simulates the complete lifecycle for the retail vertical:

  PHASE 1  -- Document Analysis (DUA)
               Upload retail docs (RetailFAQs.csv, retail_refunds_policy.yaml,
               retail_complaints_policy.yaml) -> factory infers capabilities

  PHASE 2  -- Agent Suggestions
               ConciergeAgent analyzes docs and proposes 3 agents:
               FAQ agent, Refund agent, Complaint agent

  PHASE 3  -- Deploy (Load Agents into Registry)
               Agents are loaded into AgentRegistry and verified

  PHASE 4  -- Multi-Agent Routing via Spine
               FAQ, refund, and complaint intents route to correct agents

  PHASE 5  -- Refund Agent with Tool Actions
               Full ReAct workflow: lookup_order -> check_return_eligibility
               -> initiate_return -> respond

  PHASE 6  -- Complaint Agent with Tool Actions
               Full ReAct workflow: lookup_order -> create_complaint_record
               -> compute_compensation -> respond

No real LLM calls are made. All routing and workflows are mocked.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from app.runtime.domain_agent_engine import (
    DomainAgentConfig,
    DomainAgentEngine,
)
from app.runtime.routing import Candidate, RoutePlan
from app.shared.rag import CorpusItem, build_index

# ---------------------------------------------------------------------------
# Repo constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RETAIL_FAQ_CSV = DATA_DIR / "RetailFAQs.csv"
RETAIL_REFUND_POLICY = DATA_DIR / "retail_refunds_policy.yaml"
RETAIL_COMPLAINT_POLICY = DATA_DIR / "retail_complaints_policy.yaml"

HAVE_RETAIL_DATA = (
    RETAIL_FAQ_CSV.exists()
    and RETAIL_REFUND_POLICY.exists()
    and RETAIL_COMPLAINT_POLICY.exists()
)

pytestmark = pytest.mark.skipif(
    not HAVE_RETAIL_DATA,
    reason="Retail data files not found -- run retail quickstart setup first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_items(texts: List[str], source: str = "policy.yaml") -> List[CorpusItem]:
    """Build corpus items from plain text chunks."""
    return [CorpusItem(text=t, source=source, kind="policy", meta={}) for t in texts]


def _mock_tool(name: str, response: Dict[str, Any]) -> MagicMock:
    """Create a mock ITool that returns a fixed response."""
    tool = MagicMock()
    tool.execute.return_value = response
    tool.describe.return_value = {"description": f"{name} tool"}
    return tool


class ScenarioLLM:
    """
    Mock LLM that inspects the prompt to decide what ReAct action to take.
    The decision_fn receives (system_prompt, user_content, call_index)
    and returns a ReAct JSON dict.
    """

    def __init__(self, decision_fn):
        self._fn = decision_fn
        self.calls: List[List[Dict[str, str]]] = []

    def __call__(self, messages, model=None, temperature=None, **kwargs):
        self.calls.append(messages)
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return self._fn(system, user, len(self.calls) - 1)


def _build_engine(
    corpus_texts: List[str],
    corpus_source: str = "policy.yaml",
    tools: Dict[str, Any] | None = None,
    policies: List[str] | None = None,
    llm_fn=None,
    max_steps: int = 8,
    domain: str = "retail",
    goal: str = "Help customers with retail inquiries",
) -> DomainAgentEngine:
    items = _corpus_items(corpus_texts, source=corpus_source)
    index = build_index(items)
    config = DomainAgentConfig(
        agent_id=f"retail_{domain}_agent",
        domain=domain,
        goal=goal,
        policies=policies or [],
        max_steps=max_steps,
    )
    return DomainAgentEngine(
        config=config,
        index=index,
        tools=tools or {},
        llm_fn=llm_fn,
    )


def _load_faq_corpus() -> List[str]:
    """Load FAQ questions and answers from RetailFAQs.csv as corpus texts."""
    texts = []
    with open(RETAIL_FAQ_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("Question", "")
            a = row.get("Answer", "")
            texts.append(f"Q: {q}\nA: {a}")
    return texts


def _load_policy_corpus(path: Path) -> List[str]:
    """Load a YAML policy file and split into paragraph-level chunks."""
    raw = path.read_text(encoding="utf-8")
    chunks = []
    current = []
    for line in raw.splitlines():
        if line.strip() == "" and current:
            chunks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if len(c.strip()) > 20]


class _MockRouter:
    """Routes queries deterministically based on keywords -- no LLM needed."""

    def __init__(self, routing_map: Dict[str, str]):
        self.routing_map = routing_map
        self._all_ids = list(dict.fromkeys(routing_map.values()))

    def route(self, query: str) -> RoutePlan:
        q_lower = query.lower()
        for keyword, agent_id in self.routing_map.items():
            if keyword in q_lower:
                return RoutePlan(
                    primary=agent_id,
                    strategy="single",
                    candidates=[
                        Candidate(id=agent_id, score=1.0, reason=f"matched '{keyword}'")
                    ],
                )
        fallback = self._all_ids[0]
        return RoutePlan(
            primary=fallback,
            strategy="single",
            candidates=[Candidate(id=fallback, score=0.5, reason="fallback")],
        )


class _WrappedDomainAgent:
    """Wraps a DomainAgentEngine to satisfy the IAgent protocol for registry use."""

    def __init__(self, agent_id: str, engine: DomainAgentEngine):
        self._id = agent_id
        self._engine = engine
        self.ready = True

    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        text = request.get("text") or request.get("query", "")
        ctx = request.get("context", {})
        thread_id = ctx.get("thread_id", "default")
        result = self._engine.handle(query=text, thread_id=thread_id, context=ctx)
        result["agent_id"] = self._id
        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "type": "domain_agent",
            "ready": self.ready,
            "capabilities": ["multi_turn", "tool_use", "knowledge_retrieval"],
        }


# ---------------------------------------------------------------------------
# PHASE 1 -- Document Analysis (DUA)
# ---------------------------------------------------------------------------


class TestPhase1DocumentAnalysis:
    """
    Upload retail docs and choose domain.
    Factory infers capabilities from the uploaded files.
    """

    def test_infer_capabilities_from_retail_files(self, tmp_path):
        """
        Files containing 'faq' and 'policy' keywords trigger respective capabilities.
        Uses heuristic-only logic -- no LLM.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        # Copy real retail files to tmp workspace
        shutil.copy2(RETAIL_FAQ_CSV, tmp_path / "RetailFAQs.csv")
        shutil.copy2(RETAIL_REFUND_POLICY, tmp_path / "retail_refunds_policy.yaml")
        shutil.copy2(
            RETAIL_COMPLAINT_POLICY, tmp_path / "retail_complaints_policy.yaml"
        )

        caps = dua.infer_capabilities(list(tmp_path.glob("*")))

        assert isinstance(caps, list)
        assert len(caps) > 0
        assert "faq" in caps

    def test_build_requirements_produces_valid_structure(self, tmp_path):
        """
        build_requirements() returns a dict with expected top-level keys for retail.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        shutil.copy2(RETAIL_FAQ_CSV, tmp_path / "RetailFAQs.csv")
        files = list(tmp_path.glob("*"))

        req = dua.build_requirements("retail", ["faq"], files)

        assert req["vertical"] == "retail"
        assert "capabilities" in req
        assert "faq" in req["capabilities"]
        assert "entities" in req
        assert "workflows" in req

    def test_detect_signals_llm_returns_advisory_when_mocked(self, monkeypatch):
        """
        The LLM advisory interface contract works for retail domain signals.
        """
        import importlib

        if "app.dua_v0" in sys.modules:
            del sys.modules["app.dua_v0"]
        dua = importlib.import_module("app.dua_v0")

        monkeypatch.setattr(
            dua,
            "detect_signals_llm",
            lambda filenames: {
                "primary": "retail",
                "scores": {"retail": 0.95, "fintech": 0.05},
                "explanation": "Retail FAQ, refund and complaint policy keywords found",
            },
        )

        advisory = dua.detect_signals_llm(
            [
                "RetailFAQs.csv",
                "retail_refunds_policy.yaml",
                "retail_complaints_policy.yaml",
            ]
        )
        assert advisory["primary"] == "retail"
        assert advisory["scores"]["retail"] > 0.5


# ---------------------------------------------------------------------------
# PHASE 2 -- Agent Suggestions (ConciergeAgent plan)
# ---------------------------------------------------------------------------


class TestPhase2AgentSuggestions:
    """
    ConciergeAgent analyzes retail docs and proposes agents.
    Verifies the plan includes FAQ, refund, and complaint agents.
    """

    def test_concierge_generates_plan_for_retail_docs(self, tmp_path):
        """
        ConciergeAgent.handle_event(upload_docs) returns a plan with agents.
        """
        from app.concierge.concierge_agent import ConciergeAgent

        # Copy retail files into workspace
        shutil.copy2(RETAIL_FAQ_CSV, tmp_path / "RetailFAQs.csv")
        shutil.copy2(RETAIL_REFUND_POLICY, tmp_path / "retail_refunds_policy.yaml")
        shutil.copy2(
            RETAIL_COMPLAINT_POLICY, tmp_path / "retail_complaints_policy.yaml"
        )

        agent = ConciergeAgent(
            vertical="retail",
            data_dir=str(tmp_path),
            llm_client=None,
        )

        result = agent.handle_event({"type": "upload_docs", "use_llm": False})

        assert result["type"] == "factory_plan_preview"
        assert "plan" in result
        plan = result["plan"]
        assert plan["vertical"] == "retail"
        assert len(plan["agents"]) >= 1, "Plan should propose at least one agent"

    def test_plan_contains_faq_agent(self, tmp_path):
        """Plan should include at least a FAQ/RAG agent."""
        from app.concierge.concierge_agent import ConciergeAgent

        shutil.copy2(RETAIL_FAQ_CSV, tmp_path / "RetailFAQs.csv")
        shutil.copy2(RETAIL_REFUND_POLICY, tmp_path / "retail_refunds_policy.yaml")
        shutil.copy2(
            RETAIL_COMPLAINT_POLICY, tmp_path / "retail_complaints_policy.yaml"
        )

        agent = ConciergeAgent(vertical="retail", data_dir=str(tmp_path))
        result = agent.handle_event({"type": "upload_docs", "use_llm": False})

        plan = result["plan"]
        agent_ids = [a["id"] for a in plan["agents"]]
        has_faq = any("rag" in aid or "faq" in aid for aid in agent_ids)
        assert has_faq, f"Plan should include a FAQ/RAG agent, got: {agent_ids}"

    def test_plan_text_summary_mentions_retail(self, tmp_path):
        """Text summary should reference the retail domain."""
        from app.concierge.concierge_agent import ConciergeAgent

        shutil.copy2(RETAIL_FAQ_CSV, tmp_path / "RetailFAQs.csv")
        shutil.copy2(RETAIL_REFUND_POLICY, tmp_path / "retail_refunds_policy.yaml")
        shutil.copy2(
            RETAIL_COMPLAINT_POLICY, tmp_path / "retail_complaints_policy.yaml"
        )

        agent = ConciergeAgent(vertical="retail", data_dir=str(tmp_path))
        result = agent.handle_event({"type": "upload_docs", "use_llm": False})

        assert "retail" in result["text"].lower()


# ---------------------------------------------------------------------------
# PHASE 3 -- Deploy (Load Agents into Registry)
# ---------------------------------------------------------------------------


def _make_faq_engine():
    """Build a FAQ domain agent engine with real retail FAQ corpus."""
    corpus = _load_faq_corpus()[:20]  # first 20 FAQs for speed
    llm = ScenarioLLM(
        lambda sys, usr, idx: {
            "thought": "I found the answer in the knowledge base.",
            "action": "respond",
            "action_input": {
                "answer": "Based on our FAQ, I can help you with that. "
                "Please refer to our customer service for further details."
            },
        }
    )
    return _build_engine(
        corpus_texts=corpus,
        corpus_source="RetailFAQs.csv",
        domain="faq",
        goal="Answer customer questions using the retail FAQ knowledge base",
        llm_fn=llm,
    )


def _make_refund_engine():
    """Build a refund domain agent engine with retail refund policy corpus."""
    corpus = _load_policy_corpus(RETAIL_REFUND_POLICY)[:15]
    llm = ScenarioLLM(
        lambda sys, usr, idx: (
            {
                "thought": "I should retrieve the refund policy.",
                "action": "retrieve_knowledge",
                "action_input": {"query": "return refund eligibility"},
            }
            if idx == 0
            else {
                "thought": "Policy retrieved. Responding to customer.",
                "action": "respond",
                "action_input": {
                    "answer": "Based on our return policy, you may return unopened items "
                    "within 365 days for a full refund."
                },
            }
        )
    )
    return _build_engine(
        corpus_texts=corpus,
        corpus_source="retail_refunds_policy.yaml",
        domain="refunds",
        goal="Help customers with return and refund requests following the retail return policy",
        llm_fn=llm,
    )


def _make_complaint_engine():
    """Build a complaint domain agent engine with retail complaint policy corpus."""
    corpus = _load_policy_corpus(RETAIL_COMPLAINT_POLICY)[:15]
    llm = ScenarioLLM(
        lambda sys, usr, idx: (
            {
                "thought": "I should retrieve the complaint policy.",
                "action": "retrieve_knowledge",
                "action_input": {"query": "complaint resolution damaged product"},
            }
            if idx == 0
            else {
                "thought": "Policy retrieved. Responding to customer.",
                "action": "respond",
                "action_input": {
                    "answer": "I am sorry about the damage. We will arrange a replacement "
                    "or refund. Your complaint has been recorded."
                },
            }
        )
    )
    return _build_engine(
        corpus_texts=corpus,
        corpus_source="retail_complaints_policy.yaml",
        domain="complaints",
        goal="Help customers resolve complaints about products, deliveries, and services",
        llm_fn=llm,
    )


class TestPhase3Deploy:
    """
    Agents are loaded into the runtime registry and verified.
    """

    def test_registry_accepts_three_retail_agents(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()

        faq = _WrappedDomainAgent("retail_faq_agent", _make_faq_engine())
        refund = _WrappedDomainAgent("retail_refund_agent", _make_refund_engine())
        complaint = _WrappedDomainAgent(
            "retail_complaint_agent", _make_complaint_engine()
        )

        registry.register("retail_faq_agent", faq)
        registry.register("retail_refund_agent", refund)
        registry.register("retail_complaint_agent", complaint)

        assert "retail_faq_agent" in registry.all_ids()
        assert "retail_refund_agent" in registry.all_ids()
        assert "retail_complaint_agent" in registry.all_ids()

    def test_registry_get_returns_correct_agent(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()
        faq = _WrappedDomainAgent("retail_faq_agent", _make_faq_engine())
        registry.register("retail_faq_agent", faq)

        retrieved = registry.get("retail_faq_agent")
        assert retrieved is faq

    def test_registry_all_meta_shows_ready(self):
        from app.runtime.registry import AgentRegistry

        registry = AgentRegistry()
        faq = _WrappedDomainAgent("retail_faq_agent", _make_faq_engine())
        refund = _WrappedDomainAgent("retail_refund_agent", _make_refund_engine())
        complaint = _WrappedDomainAgent(
            "retail_complaint_agent", _make_complaint_engine()
        )

        registry.register("retail_faq_agent", faq)
        registry.register("retail_refund_agent", refund)
        registry.register("retail_complaint_agent", complaint)

        meta = registry.all_meta()
        assert meta["retail_faq_agent"]["ready"] is True
        assert meta["retail_refund_agent"]["ready"] is True
        assert meta["retail_complaint_agent"]["ready"] is True


# ---------------------------------------------------------------------------
# PHASE 4 -- Multi-Agent Routing via Spine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def retail_spine():
    """
    Build a fully wired RuntimeSpine with retail agents and a mock router.
    """
    from app.runtime.registry import AgentRegistry
    from app.runtime.spine import RuntimeSpine
    from app.runtime.guardrails import NoOpGuardrails

    registry = AgentRegistry()

    faq = _WrappedDomainAgent("retail_faq_agent", _make_faq_engine())
    refund = _WrappedDomainAgent("retail_refund_agent", _make_refund_engine())
    complaint = _WrappedDomainAgent("retail_complaint_agent", _make_complaint_engine())

    registry.register("retail_faq_agent", faq)
    registry.register("retail_refund_agent", refund)
    registry.register("retail_complaint_agent", complaint)

    router = _MockRouter(
        {
            # Refund keywords
            "return": "retail_refund_agent",
            "refund": "retail_refund_agent",
            "exchange": "retail_refund_agent",
            # Complaint keywords
            "damaged": "retail_complaint_agent",
            "complaint": "retail_complaint_agent",
            "missing": "retail_complaint_agent",
            "warranty": "retail_complaint_agent",
            "broken": "retail_complaint_agent",
            "wrong item": "retail_complaint_agent",
            # FAQ keywords (fallback)
            "delivery": "retail_faq_agent",
            "order": "retail_faq_agent",
            "payment": "retail_faq_agent",
            "faq": "retail_faq_agent",
            "family": "retail_faq_agent",
            "gift": "retail_faq_agent",
            "track": "retail_faq_agent",
            "shipping": "retail_faq_agent",
            "assembly": "retail_faq_agent",
        }
    )

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
    )
    return spine


class TestPhase4MultiAgentRouting:
    """
    Customer interacts with the deployed retail system via the spine.
    Shows that FAQ, refund, and complaint agents are correctly routed.
    """

    # -- FAQ Agent Routing --

    def test_faq_delivery_query_routes_to_faq_agent(self, retail_spine):
        result = retail_spine.handle_chat(
            "What delivery options do you offer?",
            context={"thread_id": "retail-e2e-faq-001"},
        )
        assert result.get("agent_id") == "retail_faq_agent"

    def test_faq_query_returns_non_empty_answer(self, retail_spine):
        result = retail_spine.handle_chat(
            "How can I track my order?",
            context={"thread_id": "retail-e2e-faq-002"},
        )
        answer = result.get("answer", "")
        assert answer, "FAQ agent should return a non-empty answer"

    def test_faq_payment_query_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "What payment methods do you accept?",
            context={"thread_id": "retail-e2e-faq-003"},
        )
        assert result.get("agent_id") == "retail_faq_agent"

    def test_faq_gift_card_query_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "How do gift cards work?",
            context={"thread_id": "retail-e2e-faq-004"},
        )
        assert result.get("agent_id") == "retail_faq_agent"

    def test_faq_assembly_query_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "Do you offer assembly services?",
            context={"thread_id": "retail-e2e-faq-005"},
        )
        assert result.get("agent_id") == "retail_faq_agent"

    def test_faq_response_has_score(self, retail_spine):
        result = retail_spine.handle_chat(
            "What shipping options are available?",
            context={"thread_id": "retail-e2e-faq-006"},
        )
        assert "score" in result
        assert isinstance(result["score"], (int, float))

    # -- Refund Agent Routing --

    def test_refund_query_routes_to_refund_agent(self, retail_spine):
        result = retail_spine.handle_chat(
            "I want to return a product I bought last week",
            context={"thread_id": "retail-e2e-refund-001"},
        )
        assert result.get("agent_id") == "retail_refund_agent"

    def test_refund_exchange_query_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "Can I exchange my mattress?",
            context={"thread_id": "retail-e2e-refund-002"},
        )
        assert result.get("agent_id") == "retail_refund_agent"

    # -- Complaint Agent Routing --

    def test_complaint_damaged_query_routes_to_complaint_agent(self, retail_spine):
        result = retail_spine.handle_chat(
            "My product arrived damaged",
            context={"thread_id": "retail-e2e-complaint-001"},
        )
        assert result.get("agent_id") == "retail_complaint_agent"

    def test_complaint_missing_parts_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "I am missing screws from my furniture kit",
            context={"thread_id": "retail-e2e-complaint-002"},
        )
        assert result.get("agent_id") == "retail_complaint_agent"

    def test_complaint_warranty_query_routes_correctly(self, retail_spine):
        result = retail_spine.handle_chat(
            "I need to make a warranty claim",
            context={"thread_id": "retail-e2e-complaint-003"},
        )
        assert result.get("agent_id") == "retail_complaint_agent"

    # -- Router Plan in Response --

    def test_router_plan_in_response(self, retail_spine):
        result = retail_spine.handle_chat(
            "What delivery options are available?",
            context={"thread_id": "retail-e2e-plan-001"},
        )
        plan = result.get("router_plan", {})
        assert plan.get("primary") == "retail_faq_agent"

    # -- Session Isolation --

    def test_concurrent_sessions_are_isolated(self, retail_spine):
        """Two threads with different intents should not bleed into each other."""
        from app.runtime.spine import THREAD_CTX

        tid_faq = "retail-e2e-iso-faq-001"
        tid_complaint = "retail-e2e-iso-complaint-001"
        THREAD_CTX.pop(tid_faq, None)
        THREAD_CTX.pop(tid_complaint, None)

        r_faq = retail_spine.handle_chat(
            "What delivery options do you offer?",
            context={"thread_id": tid_faq},
        )
        r_complaint = retail_spine.handle_chat(
            "My product arrived damaged and I want to file a complaint",
            context={"thread_id": tid_complaint},
        )

        assert r_faq.get("agent_id") == "retail_faq_agent"
        assert r_complaint.get("agent_id") == "retail_complaint_agent"

    def test_empty_query_returns_error(self, retail_spine):
        result = retail_spine.handle_chat(
            "", context={"thread_id": "retail-e2e-empty-001"}
        )
        assert "error" in result

    def test_multiple_faq_sessions_independent(self, retail_spine):
        """Multiple parallel FAQ sessions dont share state."""
        agents_used = []
        for i in range(3):
            result = retail_spine.handle_chat(
                "What are the delivery options?",
                context={"thread_id": f"retail-e2e-multi-faq-{i}"},
            )
            agents_used.append(result.get("agent_id"))

        assert all(a == "retail_faq_agent" for a in agents_used)


# ---------------------------------------------------------------------------
# PHASE 5 -- Refund Agent with Tool Actions
# ---------------------------------------------------------------------------


class TestPhase5RefundAgentToolActions:
    """
    Refund agent performs a full ReAct workflow with retail tools:
    retrieve_knowledge -> lookup_order -> check_return_eligibility -> initiate_return -> respond
    """

    def _refund_happy_llm(self, system: str, user: str, call_idx: int):
        """Index-based LLM that progresses through a refund happy-path workflow."""
        steps = [
            {
                "thought": "I should check the knowledge base for the return policy.",
                "action": "retrieve_knowledge",
                "action_input": {"query": "return refund eligibility window"},
            },
            {
                "thought": "Policy retrieved. I need to look up the customer order.",
                "action": "call_tool",
                "action_input": {
                    "tool": "lookup_order",
                    "args": {"order_id": "ORD-12345"},
                },
            },
            {
                "thought": "Order found and within 365-day window. Checking eligibility.",
                "action": "call_tool",
                "action_input": {
                    "tool": "check_return_eligibility",
                    "args": {
                        "order_id": "ORD-12345",
                        "item_id": "ITEM-001",
                        "product_condition": "unopened",
                    },
                },
            },
            {
                "thought": "Item is eligible. Initiating the return.",
                "action": "call_tool",
                "action_input": {
                    "tool": "initiate_return",
                    "args": {
                        "order_id": "ORD-12345",
                        "item_id": "ITEM-001",
                        "reason": "Changed mind",
                    },
                },
            },
            {
                "thought": "Return initiated successfully. Informing the customer.",
                "action": "respond",
                "action_input": {
                    "answer": "Your return has been approved. Return ID: RET-ORD-12345-ITEM-001. "
                    "A refund of $89.99 will be processed to your original payment method "
                    "within 3-10 business days."
                },
            },
        ]
        return steps[min(call_idx, len(steps) - 1)]

    def test_refund_full_workflow_with_tools(self):
        """Engine processes retrieve -> lookup -> eligibility -> initiate -> respond."""
        llm = ScenarioLLM(self._refund_happy_llm)
        engine = _build_engine(
            corpus_texts=[
                "Unopened products may be returned within 365 days with proof of purchase for a full refund.",
                "Opened products may be returned within 180 days provided the item is clean and complete.",
                "Refunds are made to the original payment method within 3-10 business days.",
                "Plants, cut fabric, custom countertops, and as-is products are not eligible for return.",
                "Use lookup_order to find the order, check_return_eligibility to verify, initiate_return to process.",
            ],
            corpus_source="retail_refunds_policy.yaml",
            tools={
                "lookup_order": _mock_tool(
                    "lookup_order",
                    {
                        "order_id": "ORD-12345",
                        "order_found": True,
                        "order_date": "2026-03-01",
                        "order_age_days": 30,
                        "order_total": 89.99,
                        "items": [
                            {
                                "item_id": "ITEM-001",
                                "name": "KALLAX Shelf",
                                "price": 89.99,
                            }
                        ],
                    },
                ),
                "check_return_eligibility": _mock_tool(
                    "check_return_eligibility",
                    {
                        "eligible": True,
                        "return_window_days": 365,
                        "days_remaining": 335,
                        "refund_amount": 89.99,
                    },
                ),
                "initiate_return": _mock_tool(
                    "initiate_return",
                    {
                        "return_id": "RET-ORD-12345-ITEM-001",
                        "return_status": "approved",
                        "refund_amount": 89.99,
                        "estimated_refund_timeline": "3-10 business days",
                    },
                ),
            },
            domain="refunds",
            goal="Help customers with return and refund requests",
            llm_fn=llm,
        )
        result = engine.handle("I want to return order ORD-12345, item ITEM-001")

        # Answer is present
        assert result["answer"], "Refund agent must return an answer"

        # All tools were used
        assert "lookup_order" in result["tools_used"]
        assert "check_return_eligibility" in result["tools_used"]
        assert "initiate_return" in result["tools_used"]

        # Knowledge was retrieved
        assert result["knowledge_retrieved"] is True

        # At least 4 steps: retrieve + 3 tools + respond
        assert result["step_count"] >= 4

        # No escalation on happy path
        assert result.get("escalation") is not True

    def test_refund_tools_accumulate_in_slots(self):
        """Tool results should accumulate in the engine slots."""
        llm = ScenarioLLM(self._refund_happy_llm)
        engine = _build_engine(
            corpus_texts=["Return items within 365 days for full refund."],
            tools={
                "lookup_order": _mock_tool(
                    "lookup_order", {"order_found": True, "order_age_days": 30}
                ),
                "check_return_eligibility": _mock_tool(
                    "check_return_eligibility",
                    {"eligible": True, "refund_amount": 89.99},
                ),
                "initiate_return": _mock_tool(
                    "initiate_return", {"return_status": "approved"}
                ),
            },
            domain="refunds",
            goal="Process customer returns",
            llm_fn=llm,
        )
        result = engine.handle("Return my order ORD-12345")

        slots = result.get("slots", {})
        assert slots.get("order_found") is True
        assert slots.get("eligible") is True
        assert slots.get("return_status") == "approved"

    def test_refund_react_trace_contains_expected_actions(self):
        """The ReAct trace should contain retrieve, call_tool, and respond actions."""
        llm = ScenarioLLM(self._refund_happy_llm)
        engine = _build_engine(
            corpus_texts=["Return policy: 365 days unopened, 180 days opened."],
            tools={
                "lookup_order": _mock_tool("lookup_order", {"order_found": True}),
                "check_return_eligibility": _mock_tool(
                    "check_return_eligibility", {"eligible": True}
                ),
                "initiate_return": _mock_tool(
                    "initiate_return", {"return_status": "approved"}
                ),
            },
            domain="refunds",
            goal="Process customer returns",
            llm_fn=llm,
        )
        result = engine.handle("Return ORD-12345")

        actions = [s["action"] for s in result["react_trace"]]
        assert "retrieve_knowledge" in actions
        assert "call_tool" in actions
        assert "respond" in actions


# ---------------------------------------------------------------------------
# PHASE 6 -- Complaint Agent with Tool Actions
# ---------------------------------------------------------------------------


class TestPhase6ComplaintAgentToolActions:
    """
    Complaint agent performs a full ReAct workflow with retail tools:
    retrieve_knowledge -> lookup_order -> create_complaint_record
    -> compute_compensation -> respond
    """

    def _complaint_happy_llm(self, system: str, user: str, call_idx: int):
        """Index-based LLM for complaint happy-path workflow."""
        steps = [
            {
                "thought": "I should check the knowledge base for the complaint handling policy.",
                "action": "retrieve_knowledge",
                "action_input": {"query": "complaint resolution damaged product"},
            },
            {
                "thought": "Policy retrieved. I need to look up the customer order.",
                "action": "call_tool",
                "action_input": {
                    "tool": "lookup_order",
                    "args": {"order_id": "ORD-67890"},
                },
            },
            {
                "thought": "Order found. Creating a formal complaint record.",
                "action": "call_tool",
                "action_input": {
                    "tool": "create_complaint_record",
                    "args": {
                        "customer_id": "CUST-555",
                        "category": "damaged_product",
                        "severity": "medium",
                        "description": "Product arrived with visible damage",
                    },
                },
            },
            {
                "thought": "Complaint recorded. Computing compensation for the customer.",
                "action": "call_tool",
                "action_input": {
                    "tool": "compute_compensation",
                    "args": {
                        "complaint_id": "CMP-CUST-555-001",
                        "category": "damaged_product",
                        "severity": "medium",
                    },
                },
            },
            {
                "thought": "Compensation calculated. Informing the customer of the resolution.",
                "action": "respond",
                "action_input": {
                    "answer": "I am sorry about the damage to your product. I have created "
                    "complaint record CMP-CUST-555-001. You are eligible for a compensation "
                    "of 200 USD. This has been auto-approved and will be credited to your "
                    "account. Resolution target: 3 business days."
                },
            },
        ]
        return steps[min(call_idx, len(steps) - 1)]

    def test_complaint_full_workflow_with_tools(self):
        """Engine processes retrieve -> lookup -> create_complaint -> compensate -> respond."""
        llm = ScenarioLLM(self._complaint_happy_llm)
        engine = _build_engine(
            corpus_texts=[
                "Complaints about damaged or missing items must be reported within 14 days.",
                "Compensation up to 00 may be auto-approved by the system.",
                "Compensation exceeding 00 requires manager approval.",
                "Standard complaint resolution target is 3 business days.",
                "Use lookup_order to find the order, create_complaint_record to log it, compute_compensation to calculate.",
            ],
            corpus_source="retail_complaints_policy.yaml",
            tools={
                "lookup_order": _mock_tool(
                    "lookup_order",
                    {
                        "order_id": "ORD-67890",
                        "order_found": True,
                        "order_date": "2026-03-20",
                        "order_age_days": 10,
                        "order_total": 199.00,
                        "items": [
                            {
                                "item_id": "ITEM-010",
                                "name": "MALM Desk",
                                "price": 199.00,
                            }
                        ],
                    },
                ),
                "create_complaint_record": _mock_tool(
                    "create_complaint_record",
                    {
                        "complaint_id": "CMP-CUST-555-001",
                        "status": "open",
                        "category": "damaged_product",
                        "severity": "medium",
                    },
                ),
                "compute_compensation": _mock_tool(
                    "compute_compensation",
                    {
                        "complaint_id": "CMP-CUST-555-001",
                        "compensation_amount": 200,
                        "approval_required": False,
                        "message": "Compensation estimate: 200. Auto-approved.",
                    },
                ),
            },
            domain="complaints",
            goal="Help customers resolve complaints about products, deliveries, and services",
            llm_fn=llm,
        )
        result = engine.handle("My MALM desk arrived damaged, order ORD-67890")

        # Answer is present
        assert result["answer"], "Complaint agent must return an answer"

        # All tools were used
        assert "lookup_order" in result["tools_used"]
        assert "create_complaint_record" in result["tools_used"]
        assert "compute_compensation" in result["tools_used"]

        # Knowledge was retrieved
        assert result["knowledge_retrieved"] is True

        # At least 4 steps
        assert result["step_count"] >= 4

        # No escalation on happy path (medium severity, auto-approved)
        assert result.get("escalation") is not True

    def test_complaint_tools_accumulate_in_slots(self):
        """Tool results should accumulate in the engine slots."""
        llm = ScenarioLLM(self._complaint_happy_llm)
        engine = _build_engine(
            corpus_texts=["Report damaged items within 14 days for resolution."],
            tools={
                "lookup_order": _mock_tool(
                    "lookup_order", {"order_found": True, "order_age_days": 10}
                ),
                "create_complaint_record": _mock_tool(
                    "create_complaint_record",
                    {"complaint_id": "CMP-001", "status": "open"},
                ),
                "compute_compensation": _mock_tool(
                    "compute_compensation",
                    {"compensation_amount": 200, "approval_required": False},
                ),
            },
            domain="complaints",
            goal="Resolve customer complaints",
            llm_fn=llm,
        )
        result = engine.handle("Damaged desk from order ORD-67890")

        slots = result.get("slots", {})
        assert slots.get("order_found") is True
        assert slots.get("complaint_id") == "CMP-001"
        assert slots.get("compensation_amount") == 200
        assert slots.get("approval_required") is False

    def test_complaint_react_trace_contains_expected_actions(self):
        """The ReAct trace should contain retrieve, call_tool, and respond actions."""
        llm = ScenarioLLM(self._complaint_happy_llm)
        engine = _build_engine(
            corpus_texts=["Complaint handling: damaged items within 14 days."],
            tools={
                "lookup_order": _mock_tool("lookup_order", {"order_found": True}),
                "create_complaint_record": _mock_tool(
                    "create_complaint_record", {"complaint_id": "CMP-001"}
                ),
                "compute_compensation": _mock_tool(
                    "compute_compensation", {"compensation_amount": 200}
                ),
            },
            domain="complaints",
            goal="Resolve customer complaints",
            llm_fn=llm,
        )
        result = engine.handle("Damaged product from ORD-67890")

        actions = [s["action"] for s in result["react_trace"]]
        assert "retrieve_knowledge" in actions
        assert "call_tool" in actions
        assert "respond" in actions

    def test_complaint_with_replacement_part_tool(self):
        """Complaint agent can order replacement parts for missing hardware."""

        def _missing_parts_llm(system: str, user: str, call_idx: int):
            steps = [
                {
                    "thought": "Checking the complaint policy for missing parts.",
                    "action": "retrieve_knowledge",
                    "action_input": {"query": "missing parts replacement hardware"},
                },
                {
                    "thought": "Policy says hardware parts can be ordered free. Ordering now.",
                    "action": "call_tool",
                    "action_input": {
                        "tool": "order_replacement_part",
                        "args": {
                            "order_id": "ORD-111",
                            "item_id": "ITEM-001",
                            "part_number": "123456",
                        },
                    },
                },
                {
                    "thought": "Replacement part ordered. Informing the customer.",
                    "action": "respond",
                    "action_input": {
                        "answer": "I have ordered replacement part 123456 free of charge. "
                        "Expected delivery in 5-7 business days."
                    },
                },
            ]
            return steps[min(call_idx, len(steps) - 1)]

        llm = ScenarioLLM(_missing_parts_llm)
        engine = _build_engine(
            corpus_texts=[
                "Missing hardware can be ordered free of charge via the spare parts system.",
                "Hardware with a 6-digit part number can be ordered using order_replacement_part.",
            ],
            tools={
                "order_replacement_part": _mock_tool(
                    "order_replacement_part",
                    {
                        "replacement_order_id": "RPL-ORD-111-123456",
                        "status": "ordered",
                        "cost": 0,
                        "estimated_delivery": "5-7 business days",
                    },
                ),
            },
            domain="complaints",
            goal="Help customers with missing parts",
            llm_fn=llm,
        )
        result = engine.handle(
            "I am missing screws (part 123456) from my order ORD-111"
        )

        assert result["answer"], "Should return an answer about replacement parts"
        assert "order_replacement_part" in result["tools_used"]
        assert result["knowledge_retrieved"] is True
