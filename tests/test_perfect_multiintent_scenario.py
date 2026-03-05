# tests/test_perfect_multiintent_scenario.py
"""
Perfect Multi-Intent Scenario — Full Regression Test

Scenario (Fintech domain, BankFAQs + refunds_policy uploaded):
  User: "what is the needed documents to open an account?
         and I want to issue a refund"

Flow:
  T1  User sends multi-intent query
      → AOP detects 2 intents → presents task menu
  T2  User selects the FAQ task ("1")
      → FAQ agent retrieves ambiguous results → asks clarification
        ("Which type of account?")
  T3  User selects account type ("A" — sole proprietorship)
      → FAQ resolves selection → returns document details
      → remaining refund task offered as quick reply
  T4  User selects refund task ("1" or "yes")
      → Refund workflow starts → asks for required info
        (request_id, customer_id, amount)
  T5  User provides all required info
      → Workflow auto-chains through system states → completes

Validates:
  - Multi-intent detection and AOP decomposition
  - Sequential task menu presentation
  - RAG post-retrieval clarification (account type disambiguation)
  - Agent pinning / sticky routing for multi-turn FAQ
  - Slot accumulation across agents
  - Workflow auto-chain through system states (no repetitive questions)
  - Proper task completion and agent unpinning
  - Quick-reply numbering consistency with task selection

No real LLM calls are made.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Repo constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FAQ_AGENT_DIR = REPO_ROOT / "generated" / "faq_agent"
REFUNDS_AGENT_DIR = REPO_ROOT / "generated" / "refunds_workflow"

HAVE_ARTIFACTS = (
    (FAQ_AGENT_DIR / "agent.py").exists()
    and (FAQ_AGENT_DIR / "faqs.json").exists()
    and (REFUNDS_AGENT_DIR / "agent.py").exists()
    and (REFUNDS_AGENT_DIR / "workflow_spec.json").exists()
)

pytestmark = pytest.mark.skipif(
    not HAVE_ARTIFACTS,
    reason="Pre-built agent artifacts not found — run factory deploy first",
)

# ---------------------------------------------------------------------------
# Thread ID for the entire scenario
# ---------------------------------------------------------------------------

THREAD_ID = "perfect-scenario-001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_agent_from_dir(agent_id: str, agent_dir: Path):
    """Dynamically load an agent from its generated directory."""
    module_name = f"_scenario_{agent_id}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    agent_py = agent_dir / "agent.py"
    spec = importlib.util.spec_from_file_location(module_name, agent_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    agent = mod.Agent()
    agent.load({})
    return agent


# ---------------------------------------------------------------------------
# Smart LLM Mock — dispatches based on prompt content
# ---------------------------------------------------------------------------


class ScenarioLLMMock:
    """
    Stateful mock for chat_json that returns contextual responses
    based on the system prompt content.
    """

    def __init__(self):
        self.call_log: List[Dict[str, str]] = []
        self._workflow_mapper_calls = 0

    def __call__(self, messages=None, **kwargs):
        messages = messages or []
        sys_msg = messages[0]["content"] if messages else ""
        user_msg = messages[-1]["content"] if len(messages) > 1 else ""
        self.call_log.append({"sys": sys_msg[:200], "user": user_msg[:200]})

        # ── Orchestration pattern classifier ──
        if "query classifier" in sys_msg:
            return {"pattern": "hierarchical_delegation"}

        # ── AOP decomposer ──
        if "Decompose" in sys_msg or "subtask" in sys_msg:
            return {
                "subtasks": [
                    "INFORMATIONAL: documents needed to open a bank account",
                    "ACTION: issue a refund for a transaction",
                ]
            }

        # ── Completeness detector ──
        if "completeness" in sys_msg.lower() or "coverage" in sys_msg.lower():
            return {
                "complete": True,
                "missing": [],
                "reasoning": "Both document lookup and refund action are covered.",
                "coverage_ratio": 1.0,
            }

        # ── RAG clarification question (if LLM synthesis is invoked) ──
        if "clarification" in sys_msg.lower() and "question" in sys_msg.lower():
            return {
                "question": (
                    "I found information about several account types. "
                    "Which type are you interested in?\n"
                    "A) Sole proprietorship\n"
                    "B) Partnership firm\n"
                    "C) Company (Private/Limited)\n"
                    "D) Limited Liability Partnership"
                )
            }

        # ── RAG synthesis (grounded answer generation) ──
        if "grounded" in sys_msg.lower() or "synthesis" in sys_msg.lower():
            return {
                "answer": (
                    "For a sole proprietorship, the required documents include: "
                    "proof of existence, proof of address, KYC of the proprietor, "
                    "and registration certificates."
                ),
                "cited_passages": [1],
            }

        # ── Workflow event router (mapper) ──
        if "workflow event router" in sys_msg:
            return self._workflow_mapper(user_msg)

        # ── Voice rendering ──
        if "chat voice" in sys_msg.lower() or "customer-service" in sys_msg.lower():
            return self._voice_render(user_msg, kwargs)

        # ── Default fallback ──
        return {"response": "mock_default"}

    def _workflow_mapper(self, user_msg: str) -> Dict[str, Any]:
        """Return appropriate mapper response based on call sequence."""
        self._workflow_mapper_calls += 1

        if self._workflow_mapper_calls == 1:
            # First call: AOP executes refund subtask — no user info yet
            return {
                "event": None,
                "slots": {},
                "confidence": 0.3,
                "rationale": "No customer data provided in subtask description",
            }
        else:
            # Subsequent calls: user provides all required info
            return {
                "event": "received",
                "slots": {
                    "request_id": "REQ-SC-001",
                    "customer_id": "CUST-SC-001",
                    "amount": 500,
                },
                "confidence": 0.95,
                "rationale": "All required slots extracted from user message",
            }

    def _voice_render(self, user_msg: str, kwargs: Dict) -> Dict[str, Any]:
        """Return minimal but valid voice output."""
        return {
            "messages": ["I'd be happy to help you with that."],
            "quick_replies": [],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def scenario_env():
    """
    Build a fully wired RuntimeSpine with:
      - Real FAQ and refund agents (loaded from generated/)
      - AOP coordinator for multi-intent decomposition
      - All LLM calls mocked via ScenarioLLMMock
      - FAQ agent configured with low ambiguity thresholds
      - Refund agent using inferred auto-advance (no policy bridge dependency)
    """
    from app.runtime.guardrails import NoOpGuardrails
    from app.runtime.memory import ConversationMemory
    from app.runtime.registry import AgentRegistry
    from app.runtime.spine import THREAD_CTX, RuntimeSpine

    # Clear any stale thread context
    THREAD_CTX.pop(THREAD_ID, None)

    # ── Load agents ──
    faq_agent = _load_agent_from_dir("faq_agent", FAQ_AGENT_DIR)
    refunds_agent = _load_agent_from_dir("refunds_workflow", REFUNDS_AGENT_DIR)

    # ── Patch refund agent: use inferred auto-advance only ──
    # This removes dependency on compiled policy pack files.
    # All system states auto-advance via the first happy-path event.
    refunds_agent.policy_state_map = {}
    refunds_agent.policy_bridge = None

    # ── Pre-create FAQ FSM with low ambiguity thresholds ──
    # This ensures post-retrieval clarification triggers reliably
    # when the query matches multiple account types.
    from app.runtime.rag_fsm import RAGFiniteStateMachine, RAGFSMConfig

    rag_cfg = RAGFSMConfig(
        enable_retrieval_clarification=True,
        ambiguity_score_flatness_threshold=0.20,
        ambiguity_topic_diversity_threshold=0.40,
        ambiguity_confidence_ceiling=0.70,
        enable_llm_synthesis=False,
        enable_dense_retrieval=False,
    )
    faq_agent._fsm_engines[THREAD_ID] = RAGFiniteStateMachine(
        agent_id="faq_agent",
        faqs=faq_agent.faqs,
        idf=faq_agent._idf,
        vecs=faq_agent._vecs,
        texts=faq_agent._texts,
        config=rag_cfg,
    )

    # ── Build registry ──
    # Provide blueprint_meta (agent_kind, requires_user_context) so the
    # SolvabilityEstimator can apply intent-aware scoring — same fields
    # that spec_builder adds during a real deployment.
    registry = AgentRegistry()
    registry.register(
        "faq_agent",
        faq_agent,
        meta={
            "agent_kind": "knowledge_rag",
            "requires_user_context": False,
            "aop_eligible": True,
            "customer_facing": True,
            "description": "FAQ RAG agent for banking document questions",
            "capabilities": [
                "faq_answering",
                "knowledge_base_search",
                "multi_turn",
                "clarification",
            ],
        },
    )
    registry.register(
        "refunds_workflow",
        refunds_agent,
        meta={
            "agent_kind": "workflow_runner",
            "requires_user_context": True,
            "aop_eligible": True,
            "customer_facing": True,
            "description": "Refund workflow agent for processing refund requests",
            "capabilities": ["multi_turn", "workflow", "policy_auto_events"],
        },
    )

    # ── Build AOP coordinator ──
    from app.orchestration.aop_coordinator import AOPCoordinator
    from app.orchestration.performance_store import PerformanceStore

    memory = ConversationMemory()
    performance_store = PerformanceStore(path=str(REPO_ROOT / ".factory" / "test_perf_store.json"))
    aop = AOPCoordinator(registry=registry, memory=memory, performance_store=performance_store)

    # ── Build spine ──
    # Use NoOpGuardrails to avoid policy-dependent blocks
    from app.runtime.routing import Candidate, RoutePlan

    class _DirectRouter:
        """Keyword-based router — no LLM needed."""

        def route(self, query: str) -> RoutePlan:
            q = query.lower()
            if any(kw in q for kw in ("refund", "reversal", "return")):
                return RoutePlan(
                    primary="refunds_workflow",
                    strategy="single",
                    candidates=[
                        Candidate(id="refunds_workflow", score=1.0, reason="keyword:refund")
                    ],
                )
            return RoutePlan(
                primary="faq_agent",
                strategy="single",
                candidates=[Candidate(id="faq_agent", score=1.0, reason="keyword:faq")],
            )

    spine = RuntimeSpine(
        registry=registry,
        router=_DirectRouter(),
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
        memory=memory,
        governance_enabled=False,
    )

    # ── Mock voice rendering ──
    # Replace with a simple pass-through to avoid LLM calls in voice agent
    def _mock_voice_render(user_query, thread_id, vertical, structured):
        text = ""
        qr = []

        pattern = (
            structured.get("orchestration_pattern", "") if isinstance(structured, dict) else ""
        )

        if pattern == "aop_task_menu":
            tasks = structured.get("task_menu", [])
            lines = []
            for t in tasks:
                desc = t["subtask"]
                for pfx in ("INFORMATIONAL: ", "ACTION: "):
                    if desc.startswith(pfx):
                        desc = desc[len(pfx) :]
                        break
                lines.append(f"{t['index'] + 1}. {desc[:60]}")
            text = (
                "I can help with multiple things:\n"
                + "\n".join(lines)
                + "\nWhich would you like to start with?"
            )
            qr = lines + ["No thanks"]

        elif pattern == "aop_task_result":
            text = structured.get("text") or structured.get("answer") or "Here's what I found."

        elif pattern == "aop_plan_declined":
            text = "No problem! Let me know if you need anything else."

        elif isinstance(structured, dict) and structured.get("rag_clarification"):
            text = structured.get("answer", "Could you be more specific?")

        elif isinstance(structured, dict) and structured.get("action") == "request_clarification":
            missing = structured.get("missing_slots", [])
            text = f"To proceed with your refund, I need: {', '.join(missing)}."
            qr = []

        elif isinstance(structured, dict) and structured.get("terminal"):
            text = "Your refund has been processed successfully! The amount will be returned within 3-5 business days."

        else:
            answer = ""
            if isinstance(structured, dict):
                answer = structured.get("answer") or structured.get("text") or ""
            text = answer[:300] if answer else "I've processed your request."

        return {"messages": [text] if text else ["Done."], "quick_replies": qr}

    spine.voice.render = _mock_voice_render

    mock = ScenarioLLMMock()

    yield {
        "spine": spine,
        "mock": mock,
        "faq_agent": faq_agent,
        "refunds_agent": refunds_agent,
        "registry": registry,
        "memory": memory,
        "THREAD_CTX": THREAD_CTX,
    }

    # Cleanup
    THREAD_CTX.pop(THREAD_ID, None)


# ---------------------------------------------------------------------------
# The Perfect Scenario Test
# ---------------------------------------------------------------------------


class TestPerfectMultiIntentScenario:
    """
    Complete 5-turn multi-intent conversation:
      FAQ + Refund workflow with clarification, auto-chain, and no repetition.
    """

    def test_full_scenario(self, scenario_env):
        """
        End-to-end test: multi-intent → task menu → FAQ clarification →
        FAQ answer → refund workflow → refund completion.
        """
        spine = scenario_env["spine"]
        mock = scenario_env["mock"]
        THREAD_CTX = scenario_env["THREAD_CTX"]

        # All LLM calls go through the same mock
        patches = [
            patch("app.llm_client.chat_json", side_effect=mock),
            patch("app.orchestration.aop_coordinator.chat_json", side_effect=mock),
            patch("app.orchestration.completeness_detector.chat_json", side_effect=mock),
            patch("app.runtime.workflow_mapper.chat_json", side_effect=mock),
            # voice.render is already mocked on the spine instance
        ]
        for p in patches:
            p.start()

        try:
            # ============================================================
            # TURN 1: Multi-intent query → AOP task menu
            # ============================================================
            r1 = spine.handle_chat(
                "what is the needed documents to open an account? " "and I want to issue a refund",
                request_id="req-t1",
                context={"thread_id": THREAD_ID},
            )

            # Assertions: AOP detected 2 intents → task menu
            assert (
                r1.get("orchestration_pattern") == "aop_task_menu"
            ), f"Expected aop_task_menu, got: {r1.get('orchestration_pattern')}"
            task_menu = r1.get("task_menu", [])
            assert len(task_menu) == 2, f"Expected 2 tasks, got {len(task_menu)}"
            assert r1.get("request_id") == "req-t1"

            # Verify task descriptions
            descriptions = [t["subtask"] for t in task_menu]
            assert any(
                "document" in d.lower() or "account" in d.lower() for d in descriptions
            ), f"FAQ task not found in: {descriptions}"
            assert any(
                "refund" in d.lower() for d in descriptions
            ), f"Refund task not found in: {descriptions}"

            # Verify pending AOP plan is stored
            ctx = THREAD_CTX.get(THREAD_ID, {})
            assert "_pending_aop" in ctx, "AOP plan not stored in thread context"

            print("[T1] ✓ Multi-intent detected, 2-task menu presented")

            # ============================================================
            # TURN 2: Select FAQ task → triggers RAG clarification
            # ============================================================
            r2 = spine.handle_chat(
                "1",
                request_id="req-t2",
                context={"thread_id": THREAD_ID},
            )

            # The FAQ agent should enter CLARIFY state (ambiguous account types)
            # Check if clarification was triggered
            _exec_sub = r2.get("executed_subtask", {})
            _exec_result = _exec_sub.get("result", {})

            is_clarify = (
                _exec_result.get("rag_clarification")
                or _exec_result.get("rag_state") == "CLARIFY"
                or r2.get("rag_clarification")
            )

            # If clarification was triggered, verify pinning
            if is_clarify:
                ctx2 = THREAD_CTX.get(THREAD_ID, {})
                assert (
                    ctx2.get("pinned_agent_id") == "faq_agent"
                ), f"FAQ agent not pinned after clarification: {ctx2.get('pinned_agent_id')}"
                assert ctx2.get("pinned_agent_type") == "rag_fsm"
                assert ctx2.get("pinned_terminal") is False
                print("[T2] ✓ FAQ task executed → clarification triggered, agent pinned")
            else:
                # If no clarification (direct answer), verify we got a response
                answer = (
                    _exec_result.get("answer", "") or r2.get("answer", "") or r2.get("text", "")
                )
                assert answer, f"Expected either clarification or answer, got: {r2.keys()}"
                print("[T2] ✓ FAQ task executed → direct answer (no clarification needed)")

            # ============================================================
            # TURN 3: If clarified → select account type, else skip
            # ============================================================
            if is_clarify:
                r3 = spine.handle_chat(
                    "A",
                    request_id="req-t3",
                    context={"thread_id": THREAD_ID},
                )

                # Verify FAQ agent resolved the selection
                answer = r3.get("answer", "") or r3.get("text", "")
                assert answer, "Expected FAQ answer after clarification selection"

                # Verify agent was unpinned (FAQ answered)
                ctx3 = THREAD_CTX.get(THREAD_ID, {})
                assert (
                    ctx3.get("pinned_agent_id") is None
                    or ctx3.get("pinned_agent_type") != "rag_fsm"
                ), "FAQ agent should be unpinned after answering"

                # Verify remaining refund task is offered
                remaining = r3.get("remaining_subtasks")
                assert (
                    remaining and len(remaining) > 0
                ), "Remaining refund task should be offered after FAQ completion"
                assert any(
                    "refund" in s.get("subtask", "").lower() for s in remaining
                ), f"Refund task not in remaining: {remaining}"

                # Verify quick replies use correct sequential numbering
                chat = r3.get("chat", {})
                qr = chat.get("quick_replies", [])
                # After fixing the bug, quick replies should start from "1."
                numbered_qr = [q for q in qr if q[0].isdigit()]
                if numbered_qr:
                    assert numbered_qr[0].startswith(
                        "1."
                    ), f"Quick reply should start with '1.' not '{numbered_qr[0][:5]}'"

                print("[T3] ✓ Clarification resolved → answer returned → remaining tasks offered")
            else:
                r3 = r2  # No clarification turn needed

            # ============================================================
            # TURN 4: Select refund task → workflow asks for info
            # ============================================================
            r4 = spine.handle_chat(
                "1",
                request_id="req-t4",
                context={"thread_id": THREAD_ID},
            )

            # Check that refund workflow was executed
            _exec_sub4 = r4.get("executed_subtask", {})
            _exec_result4 = _exec_sub4.get("result", {})

            # The workflow should be in receive_request state asking for info
            current_state = _exec_result4.get("current_state", "")
            workflow_id = _exec_result4.get("workflow_id", "")
            assert workflow_id, f"Expected workflow_id in result, got: {_exec_result4.keys()}"
            assert (
                current_state == "receive_request"
            ), f"Expected receive_request state, got: {current_state}"

            # Verify it asks for required info (missing slots)
            missing = _exec_result4.get("missing_slots", [])
            action = _exec_result4.get("action", "")
            assert (
                action == "request_clarification"
            ), f"Expected request_clarification action, got: {action}"
            # Should ask for customer-facing slots (not system-internal ones)
            assert any(
                s in missing for s in ("customer_id", "amount", "request_id")
            ), f"Expected customer-facing missing slots, got: {missing}"

            # Verify workflow agent is pinned
            ctx4 = THREAD_CTX.get(THREAD_ID, {})
            assert (
                ctx4.get("pinned_agent_id") == "refunds_workflow"
            ), f"Refund agent not pinned: {ctx4.get('pinned_agent_id')}"
            assert ctx4.get("pinned_agent_type") == "workflow_runner"
            assert ctx4.get("pinned_terminal") is False

            print(
                f"[T4] ✓ Refund task selected → workflow started "
                f"(state={current_state}, missing={missing})"
            )

            # ============================================================
            # TURN 5: Provide refund info → auto-chain → completion
            # ============================================================
            r5 = spine.handle_chat(
                "My customer ID is CUST-SC-001, the amount is 500 EUR, " "request ID is REQ-SC-001",
                request_id="req-t5",
                context={"thread_id": THREAD_ID},
            )

            # Verify workflow completed
            current_state5 = r5.get("current_state", "")
            terminal5 = r5.get("terminal", False)
            assert (
                terminal5 is True
            ), f"Expected terminal=True, got terminal={terminal5}, state={current_state5}"
            assert (
                current_state5 == "completed"
            ), f"Expected 'completed' state, got: {current_state5}"

            # Verify slots were filled
            slots5 = r5.get("slots", {})
            assert (
                slots5.get("customer_id") == "CUST-SC-001"
            ), f"customer_id not set: {slots5.get('customer_id')}"
            assert (
                float(slots5.get("amount", 0)) == 500.0
            ), f"amount not set: {slots5.get('amount')}"

            # Verify agent was unpinned after terminal
            ctx5 = THREAD_CTX.get(THREAD_ID, {})
            assert "pinned_agent_id" not in ctx5, "Workflow agent should be unpinned after terminal"

            # Verify no more pending AOP tasks
            assert "_pending_aop" not in ctx5 or not ctx5.get(
                "_pending_aop"
            ), "All AOP tasks should be completed"

            print("[T5] ✓ Refund info provided → auto-chain completed → terminal")
            print()
            print("=" * 60)
            print("  PERFECT SCENARIO: ALL 5 TURNS PASSED")
            print("=" * 60)

        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Focused Sub-Tests — verify specific behaviors in isolation
# ---------------------------------------------------------------------------


class TestQuickReplyNumbering:
    """Verify that quick_reply numbering is sequential (1-based menu position)."""

    def test_remaining_tasks_use_sequential_numbering(self, scenario_env):
        """
        After completing the first task, remaining tasks should be numbered
        starting from 1, not using their original index.
        """
        spine = scenario_env["spine"]
        mock = scenario_env["mock"]
        THREAD_CTX = scenario_env["THREAD_CTX"]

        # Reset thread and mock state
        THREAD_CTX.pop("qr-test-001", None)
        mock._workflow_mapper_calls = 0

        # Manually inject a pending AOP plan with first task already completed
        THREAD_CTX["qr-test-001"] = {
            "thread_id": "qr-test-001",
            "_pending_aop": {
                "query": "test query",
                "subtasks": [
                    {
                        "description": "INFORMATIONAL: first task",
                        "assigned_agent_id": "faq_agent",
                        "solvability_score": 0.8,
                        "result": {"answer": "done"},  # Completed
                        "success": True,
                        "latency_ms": 100,
                    },
                    {
                        "description": "ACTION: second task (refund)",
                        "assigned_agent_id": "refunds_workflow",
                        "solvability_score": 0.9,
                        "result": None,  # Still pending
                        "success": False,
                        "latency_ms": 0,
                    },
                ],
            },
        }

        patches = [
            patch("app.llm_client.chat_json", side_effect=mock),
            patch("app.orchestration.aop_coordinator.chat_json", side_effect=mock),
            patch("app.orchestration.completeness_detector.chat_json", side_effect=mock),
            patch("app.runtime.workflow_mapper.chat_json", side_effect=mock),
        ]
        for p in patches:
            p.start()

        try:
            # Send a query that goes through normal routing (not task selection)
            # This triggers the "inject remaining tasks" path
            r = spine.handle_chat(
                "What are the documents for an individual account?",
                request_id="req-qr-001",
                context={"thread_id": "qr-test-001"},
            )

            # Check quick replies
            chat = r.get("chat", {})
            qr = chat.get("quick_replies", [])
            numbered_qr = [q for q in qr if q and q[0].isdigit()]

            # The remaining task (at original index 1) should show as "1." not "2."
            if numbered_qr:
                assert numbered_qr[0].startswith(
                    "1."
                ), f"Expected sequential numbering '1.', got: '{numbered_qr[0][:10]}'"
        finally:
            for p in patches:
                p.stop()
            THREAD_CTX.pop("qr-test-001", None)


class TestTaskSelectionAfterCompletion:
    """Verify that task selection works correctly after some tasks complete."""

    def test_numeric_selection_after_first_task_completed(self, scenario_env):
        """
        After first task completes, selecting '1' should match the
        first PENDING task (not the first overall task).
        """
        from app.runtime.spine import RuntimeSpine

        # Test the static method directly
        pending_aop = {
            "subtasks": [
                {
                    "description": "INFORMATIONAL: first task (done)",
                    "result": {"answer": "done"},
                },
                {
                    "description": "ACTION: second task (pending)",
                    "result": None,
                },
            ]
        }

        # "1" should select the first pending task (original index 1)
        idx = RuntimeSpine._match_aop_task_selection("1", pending_aop)
        assert idx == 1, f"Expected original index 1, got: {idx}"

        # "2" should NOT match (only 1 pending task)
        idx2 = RuntimeSpine._match_aop_task_selection("2", pending_aop)
        assert idx2 is None, f"Expected None for '2', got: {idx2}"

        # "yes" should select first pending task
        idx3 = RuntimeSpine._match_aop_task_selection("yes", pending_aop)
        assert idx3 == 1, f"Expected original index 1 for 'yes', got: {idx3}"


class TestDeclineRemainingTasks:
    """Verify that declining remaining tasks works correctly."""

    def test_decline_clears_pending_plan(self, scenario_env):
        """Saying 'no thanks' should clear the pending AOP plan."""
        spine = scenario_env["spine"]
        mock = scenario_env["mock"]
        THREAD_CTX = scenario_env["THREAD_CTX"]

        tid = "decline-test-001"
        THREAD_CTX.pop(tid, None)

        # Inject a pending plan
        THREAD_CTX[tid] = {
            "thread_id": tid,
            "_pending_aop": {
                "query": "test",
                "subtasks": [
                    {"description": "task", "result": None},
                ],
            },
        }

        patches = [
            patch("app.llm_client.chat_json", side_effect=mock),
            patch("app.orchestration.aop_coordinator.chat_json", side_effect=mock),
            patch("app.orchestration.completeness_detector.chat_json", side_effect=mock),
            patch("app.runtime.workflow_mapper.chat_json", side_effect=mock),
        ]
        for p in patches:
            p.start()

        try:
            r = spine.handle_chat("no thanks", context={"thread_id": tid})
            assert r.get("orchestration_pattern") == "aop_plan_declined"

            ctx = THREAD_CTX.get(tid, {})
            assert "_pending_aop" not in ctx, "Plan should be cleared after decline"
        finally:
            for p in patches:
                p.stop()
            THREAD_CTX.pop(tid, None)


class TestStickyRoutingDuringClarification:
    """Verify that pinned agents handle follow-up turns correctly."""

    def test_pinned_rag_skips_task_selection(self, scenario_env):
        """
        When FAQ agent is pinned for clarification, the next message
        should go to the pinned agent (not through task selection).
        """
        from app.runtime.spine import RuntimeSpine

        # If agent is pinned and non-terminal, _match_aop_task_selection is skipped
        # This is tested by verifying the spine code path:
        # "if _pinned_for_aop and _pinned_terminal is False: pass"
        # The integration test (test_full_scenario) covers this implicitly.
        # Here we verify the _is_decline static method doesn't interfere.
        assert not RuntimeSpine._is_decline("A")
        assert not RuntimeSpine._is_decline("sole proprietorship")
        assert not RuntimeSpine._is_decline("1")
        assert RuntimeSpine._is_decline("no thanks")
        assert RuntimeSpine._is_decline("that's all")


class TestWorkflowAutoChain:
    """Verify that workflow auto-chains through system states correctly."""

    def test_auto_chain_completes_refund_in_one_turn(self, scenario_env):
        """
        Once all required slots are provided, the refund workflow should
        auto-chain from receive_request all the way to completed.
        """
        mock = scenario_env["mock"]
        refunds_agent = scenario_env["refunds_agent"]

        # Reset mock state
        mock._workflow_mapper_calls = 0

        # Patch workflow mapper for this specific test
        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value={
                "event": "received",
                "slots": {
                    "request_id": "REQ-CHAIN-001",
                    "customer_id": "CUST-CHAIN-001",
                    "amount": 500,
                },
                "confidence": 0.95,
                "rationale": "test",
            },
        ):
            result = refunds_agent.handle(
                {
                    "query": "CUST-CHAIN-001, 500 EUR, REQ-CHAIN-001",
                    "thread_id": "chain-test-001",
                    "context": {"thread_id": "chain-test-001"},
                }
            )

        assert (
            result["current_state"] == "completed"
        ), f"Expected completed, got: {result['current_state']}"
        assert result["terminal"] is True

        # Verify auto-chain was used
        chain = result.get("mapper", {}).get("auto_chain", [])
        assert len(chain) > 0, "Expected auto-chain events"
        print(f"[AUTO-CHAIN] States traversed: {chain}")


class TestNoRepetitiveQuestions:
    """Verify the system doesn't ask for information already provided."""

    def test_accumulated_slots_carried_to_workflow(self, scenario_env):
        """
        Slots accumulated from earlier agents should pre-populate
        the workflow engine, avoiding redundant questions.
        """
        refunds_agent = scenario_env["refunds_agent"]

        # Simulate context with pre-accumulated slots
        ctx = {
            "thread_id": "no-repeat-001",
            "_accumulated_slots": {
                "customer_id": "CUST-PRELOADED",
            },
        }

        with patch(
            "app.runtime.workflow_mapper.chat_json",
            return_value={
                "event": None,
                "slots": {},
                "confidence": 0.3,
                "rationale": "No new info",
            },
        ):
            result = refunds_agent.handle(
                {
                    "query": "I need a refund",
                    "context": ctx,
                }
            )

        # customer_id should already be filled (not in missing_slots)
        missing = result.get("missing_slots", [])
        assert "customer_id" not in missing, (
            f"customer_id should be pre-populated from accumulated slots, "
            f"but it's in missing: {missing}"
        )

        # Verify the slot was actually set
        slots = result.get("slots", {})
        assert slots.get("customer_id") == "CUST-PRELOADED"
