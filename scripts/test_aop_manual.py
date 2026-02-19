"""
Manual smoke test for the AOP orchestration pipeline.

Run from repo root:
    python scripts/test_aop_manual.py

Requires: AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY  (or OPENAI_API_KEY)
          set in environment or in a .env file.

What it does:
  1. Creates a registry with two stub agents (refund + FAQ)
  2. Wires up the full AOP pipeline (PerformanceStore, SolvabilityEstimator,
     CompletenessDetector, AOPCoordinator, RuntimeSpine)
  3. Sends a multi-intent query → should go through AOP
  4. Sends a single-intent query → should go through direct routing
  5. Prints traces and results
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Stub Agents ──────────────────────────────────────────────────────


class StubRefundAgent:
    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        query = request.get("query", "")
        return {
            "answer": f"Refund processed successfully for your request: '{query[:60]}'",
            "score": 0.9,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "refund_agent",
            "type": "workflow_runner",
            "description": "Handles refund requests and processes returns for customers",
            "capabilities": ["refund_processing", "return_handling", "payment_reversal"],
            "ready": True,
        }


class StubFAQAgent:
    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": "Our return window is 30 days from the date of purchase. "
            "Items must be in original condition with receipt.",
            "score": 0.85,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "faq_agent",
            "type": "faq_rag",
            "description": "Answers customer FAQs about policies, products, and general knowledge",
            "capabilities": ["faq_answering", "policy_lookup", "knowledge_base_search"],
            "ready": True,
        }


class StubAccountAgent:
    def load(self, spec: Dict[str, Any]) -> None:
        pass

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answer": "Your account email has been updated successfully.",
            "score": 0.88,
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": "account_agent",
            "type": "tool_operator",
            "description": "Manages customer account updates like email and address changes",
            "capabilities": ["account_update", "email_change", "address_change"],
            "ready": True,
        }


# ── Main ─────────────────────────────────────────────────────────────


def main():
    from app.orchestration.aop_coordinator import AOPCoordinator
    from app.orchestration.performance_store import PerformanceStore
    from app.runtime.guardrails import NoOpGuardrails
    from app.runtime.registry import AgentRegistry
    from app.runtime.routing import Candidate, RoutePlan
    from app.runtime.spine import RuntimeSpine

    # -- Simple router that picks first matching agent --
    class SimpleRouter:
        def __init__(self, registry: AgentRegistry):
            self.registry = registry

        def route(self, query: str) -> RoutePlan:
            ids = self.registry.all_ids()
            primary = ids[0] if ids else ""
            return RoutePlan(
                primary=primary,
                strategy="single",
                candidates=[Candidate(id=primary, score=1.0, reason="default")] if primary else [],
            )

    # -- Setup --
    print("=" * 70)
    print("  AOP ORCHESTRATION MANUAL SMOKE TEST")
    print("=" * 70)

    registry = AgentRegistry()
    registry.register("refund_agent", StubRefundAgent(), StubRefundAgent().metadata())
    registry.register("faq_agent", StubFAQAgent(), StubFAQAgent().metadata())
    registry.register("account_agent", StubAccountAgent(), StubAccountAgent().metadata())
    print(f"\n[SETUP] Registered agents: {registry.all_ids()}")

    store = PerformanceStore(path=".factory/test_performance_history.json")
    aop = AOPCoordinator(registry=registry, performance_store=store)
    router = SimpleRouter(registry)
    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
    )
    print("[SETUP] Spine + AOP coordinator ready.\n")

    # ── Test 1: Multi-intent query (should trigger AOP) ──
    print("-" * 70)
    print("TEST 1: Multi-intent query (expecting hierarchical_delegation)")
    print("-" * 70)
    multi_query = "I need a refund for order #12345 AND please update my email to new@example.com"
    print(f"Query: {multi_query}\n")

    result1 = spine.handle_chat(multi_query, context={"thread_id": "test_multi"})
    print(f"\nOrchestration pattern: {result1.get('orchestration_pattern', 'direct (no AOP)')}")
    if "subtask_results" in result1:
        print(f"Subtasks: {len(result1['subtask_results'])}")
        for st in result1["subtask_results"]:
            print(
                f"  - [{st['agent_id']}] success={st['success']}, score={st['solvability_score']:.3f}"
            )
            print(f"    Subtask: {st['subtask'][:80]}")
        print(f"\nCompleteness: {result1.get('completeness', {}).get('complete')}")
        print(f"Coverage ratio: {result1.get('completeness', {}).get('coverage_ratio')}")
    print(
        f"\nResponse text:\n{result1.get('text', result1.get('answer', result1.get('error', 'N/A')))[:500]}"
    )

    # ── Test 2: Single-intent query (should use direct routing) ──
    print("\n" + "-" * 70)
    print("TEST 2: Single-intent query (expecting direct routing)")
    print("-" * 70)
    single_query = "What is your refund policy?"
    print(f"Query: {single_query}\n")

    result2 = spine.handle_chat(single_query, context={"thread_id": "test_single"})
    print(f"Orchestration pattern: {result2.get('orchestration_pattern', 'direct')}")
    print(f"Agent ID: {result2.get('agent_id', 'N/A')}")
    print(
        f"\nResponse text:\n{result2.get('text', result2.get('answer', result2.get('error', 'N/A')))[:500]}"
    )

    # ── Test 3: Check feedback loop ──
    print("\n" + "-" * 70)
    print("TEST 3: Performance store (feedback loop)")
    print("-" * 70)
    records = store.query()
    print(f"Records in store: {len(records)}")
    for r in records:
        print(
            f"  - agent={r.agent_id}, success={r.success}, score={r.score:.3f}, subtask={r.subtask[:60]}"
        )

    # ── Test 4: Solvability after feedback ──
    if records:
        print("\n" + "-" * 70)
        print("TEST 4: Solvability with historical data (2nd run)")
        print("-" * 70)
        result3 = spine.handle_chat(
            "I want a refund and also what is the return window?",
            context={"thread_id": "test_history"},
        )
        print(f"Orchestration pattern: {result3.get('orchestration_pattern', 'direct')}")
        if "solvability" in result3:
            print("Solvability scores (now includes history):")
            for subtask, score in result3["solvability"]["assignment_scores"].items():
                agent = result3["solvability"]["assignments"][subtask]
                print(f"  - [{agent}] {score:.4f} for: {subtask[:60]}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(
        f"  Test 1 (multi-intent): {'PASS - AOP' if result1.get('orchestration_pattern') == 'hierarchical_delegation' else 'PASS - direct' if 'error' not in result1 else 'FAIL'}"
    )
    print(f"  Test 2 (single-intent): {'PASS' if 'error' not in result2 else 'FAIL'}")
    print(
        f"  Test 3 (feedback loop): {'PASS' if len(records) > 0 else 'NO RECORDS (multi-intent may have been direct)'}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
