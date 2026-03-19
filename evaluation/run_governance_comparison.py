# evaluation/run_governance_comparison.py
"""
RQ3 Governance Comparison Runner (REAL LLM MODE)

Runs governance-specific evaluation scenarios under LOW, MEDIUM, and HIGH
governance levels using REAL agents with REAL LLM calls.  Collects RQ3-specific
metrics and exports comparison tables for thesis figure generation.

Key design: Each governance level uses IDENTICAL scenarios and agents —
the ONLY variable is the GovernanceConfig.  Real agents produce natural
language responses via LLM, and governance guardrails (blocked phrases,
hallucination detection, jargon stripping) operate on those real outputs.

Usage:
    python -m evaluation.run_governance_comparison
    python -m evaluation.run_governance_comparison --output results/rq3/
    python -m evaluation.run_governance_comparison --levels low,high
    python -m evaluation.run_governance_comparison --dry-run

Outputs:
    governance_comparison.json  — per-level metrics + trade-off deltas
    governance_results.csv      — one row per (scenario, governance_level)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Project root on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestration.aop_coordinator import AOPCoordinator  # noqa: E402
from app.orchestration.performance_store import PerformanceStore  # noqa: E402
from app.runtime.governance_config import (  # noqa: E402
    GovernanceConfig,
    GovernanceLevel,
)
from app.runtime.governance_guardrails import GovernanceAwareGuardrails  # noqa: E402
from app.runtime.policy_pack import PolicyPack  # noqa: E402
from app.runtime.registry import AgentRegistry  # noqa: E402
from app.runtime.router import LLMRouter  # noqa: E402
from app.runtime.spine import RuntimeSpine  # noqa: E402

from evaluation.harness import EvaluationHarness  # noqa: E402
from evaluation.governance_metrics import (  # noqa: E402
    GovernanceScenarioResult,
    compute_comparison_table,
)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation.rq3")

# ── Scenarios path (governance-specific ONLY) ────────────────────────
RQ3_SCENARIOS_PATH = (
    Path(__file__).resolve().parent / "scenarios" / "governance_scenarios.json"
)

# ── Rate limiting ───────────────────────────────────────────────────
INTER_SCENARIO_DELAY = float(os.getenv("EVAL_DELAY_SECONDS", "1.0"))


# ── Load real agents ─────────────────────────────────────────────────


def _load_real_agents(registry: AgentRegistry) -> None:
    """Load real generated agents from the factory spec into the registry.

    Same loader as RQ1 — ensures identical agents across evaluations.
    Set EVAL_SKIP_DENSE=1 to skip dense embedding pre-computation.
    """
    skip_dense = os.getenv("EVAL_SKIP_DENSE", "").strip()
    if skip_dense == "1":
        import app.runtime.embeddings as _emb_mod

        def _no_embed(*a, **kw):
            raise RuntimeError("Dense retrieval disabled for evaluation")

        _emb_mod.get_embed_fn = _no_embed
        logger.info("Dense retrieval disabled (EVAL_SKIP_DENSE=1), using TF-IDF only")

    factory_spec_path = ROOT / ".factory" / "factory_spec.json"
    if not factory_spec_path.exists():
        raise FileNotFoundError(
            f"Factory spec not found: {factory_spec_path}\n"
            "Run the factory planner first to generate agent specs."
        )

    spec = json.loads(factory_spec_path.read_text(encoding="utf-8"))
    gen_dir = ROOT / "generated"

    loaded = 0
    for agent_spec in spec.get("agents", []):
        if agent_spec.get("type") != "autogen":
            continue

        agent_id = agent_spec["id"]
        agent_dir = gen_dir / agent_id

        if not (agent_dir / "agent.py").exists():
            logger.warning("Skipping %s: no generated agent.py", agent_id)
            continue

        logger.info("Loading real agent: %s", agent_id)
        agent = registry.import_generated_agent(agent_id, agent_dir)
        agent.load({})

        meta = agent_spec.get("blueprint_meta", {})
        meta["ready"] = True
        registry.register(agent_id, agent, meta)
        loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "No real agents loaded. Ensure generated/ contains agent packages."
        )
    logger.info("Loaded %d real agents: %s", loaded, registry.all_ids())


# ── Verify API keys ─────────────────────────────────────────────────


def _verify_api_keys() -> None:
    """Check that LLM API credentials are available."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_azure = bool(
        os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    if not has_openai and not has_azure:
        raise RuntimeError(
            "No LLM API key found.\n"
            "Set OPENAI_API_KEY or (AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT).\n"
            "Real evaluation requires actual LLM API access."
        )


# ── Build spine with governance-aware guardrails ─────────────────────


def build_governed_spine(
    tmp_dir: Path,
    config: GovernanceConfig,
) -> tuple:
    """Create a RuntimeSpine with REAL agents and governance-aware guardrails.

    Returns:
        (spine, guardrails) — guardrails reference needed to drain events.
    """
    registry = AgentRegistry()
    _load_real_agents(registry)

    pack = PolicyPack(
        name="governance_eval",
        version="1",
        max_query_chars=config.max_query_chars,
        intent_rules={
            "refund_request": {"mode": "allow"},
            "refund_policy_query": {"mode": "allow"},
        },
        route_to_intent={
            "refunds_workflow": "refund_request",
            "faq_rag": "refund_policy_query",
        },
        blocked_phrases=["guaranteed refund", "we will definitely"],
        pii_redaction=False,
    )

    guardrails = GovernanceAwareGuardrails(pack=pack, config=config)

    perf_store = PerformanceStore(
        path=str(tmp_dir / f"eval_perf_{config.level.value}.json")
    )
    aop = AOPCoordinator(registry=registry, performance_store=perf_store)

    # Real LLM-based router
    router = LLMRouter(registry)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=guardrails,
        aop_coordinator=aop,
    )

    return spine, guardrails


# ── Run single governance level ──────────────────────────────────────


def run_single_level(
    level: GovernanceLevel,
    scenarios: List[Dict[str, Any]],
    tmp_dir: Path,
) -> List[GovernanceScenarioResult]:
    """Run all scenarios under a single governance level with REAL agents."""
    config = GovernanceConfig.for_level(level)
    spine, guardrails = build_governed_spine(tmp_dir, config)

    results: List[GovernanceScenarioResult] = []

    for i, sc in enumerate(scenarios, 1):
        logger.info(
            "[%s] [%d/%d] Running scenario: %s",
            level.value,
            i,
            len(scenarios),
            sc["id"],
        )

        # Clear governance events before each scenario
        guardrails.get_events()

        # NO mocking — real LLM calls
        harness = EvaluationHarness(spine, RQ3_SCENARIOS_PATH)
        sc_result = harness.run_scenario(sc)

        # Drain governance events
        gov_events = guardrails.drain_events()

        # Count governance-specific metrics from events
        blocks = sum(1 for e in gov_events if e.get("action") == "blocked")
        escalations = sum(1 for e in gov_events if e.get("action") == "escalated")
        mutations = sum(1 for e in gov_events if e.get("action") == "mutated")
        skips = sum(1 for e in gov_events if e.get("action") == "skipped")

        # Autonomy: derived from OBSERVED governance interventions.
        total_actions = max(sc_result.agent_calls, 1)
        interventions = blocks + mutations
        autonomy = 1.0 - min(interventions / total_actions, 1.0)

        gov_result = GovernanceScenarioResult(
            scenario_id=sc["id"],
            governance_level=level.value,
            category=sc["category"],
            success=sc_result.success,
            pattern_correct=sc_result.pattern_correct,
            agent_correct=sc_result.agent_correct,
            latency_ms=sc_result.latency_ms,
            agent_calls=sc_result.agent_calls,
            governance_blocks=blocks,
            governance_escalations=escalations,
            governance_mutations=mutations,
            governance_skips=skips,
            agent_initiated_actions=total_actions - interventions,
            total_actions=total_actions,
            autonomy_score=autonomy,
            intervention_rate=(blocks + escalations) / max(total_actions, 1),
            error=sc_result.error,
            governance_events=gov_events,
        )
        results.append(gov_result)

        status = "PASS" if gov_result.success else "FAIL"
        print(
            f"  [{level.value:6s}] [{status}] {sc['id']:25s}  "
            f"blocks={blocks} mutations={mutations} skips={skips} "
            f"autonomy={autonomy:.2f}  "
            f"lat={gov_result.latency_ms:.0f}ms"
        )

        # Rate limiting between scenarios
        if i < len(scenarios):
            time.sleep(INTER_SCENARIO_DELAY)

    return results


# ── Export ────────────────────────────────────────────────────────────


def export_comparison(
    comparison: Dict[str, Any],
    all_results: Dict[str, List[GovernanceScenarioResult]],
    output_dir: Path,
) -> None:
    """Export comparison results to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON summary
    summary_path = output_dir / "governance_comparison.json"
    summary_path.write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )

    # CSV: one row per (scenario, level)
    csv_path = output_dir / "governance_results.csv"
    fieldnames = [
        "scenario_id",
        "governance_level",
        "category",
        "success",
        "pattern_correct",
        "agent_correct",
        "latency_ms",
        "agent_calls",
        "governance_blocks",
        "governance_escalations",
        "governance_mutations",
        "governance_skips",
        "agent_initiated_actions",
        "total_actions",
        "autonomy_score",
        "intervention_rate",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for level_name, results in all_results.items():
            for r in results:
                writer.writerow(
                    {
                        "scenario_id": r.scenario_id,
                        "governance_level": r.governance_level,
                        "category": r.category,
                        "success": r.success,
                        "pattern_correct": r.pattern_correct,
                        "agent_correct": r.agent_correct,
                        "latency_ms": round(r.latency_ms, 2),
                        "agent_calls": r.agent_calls,
                        "governance_blocks": r.governance_blocks,
                        "governance_escalations": r.governance_escalations,
                        "governance_mutations": r.governance_mutations,
                        "governance_skips": r.governance_skips,
                        "agent_initiated_actions": r.agent_initiated_actions,
                        "total_actions": r.total_actions,
                        "autonomy_score": round(r.autonomy_score, 4),
                        "intervention_rate": round(r.intervention_rate, 4),
                        "error": r.error,
                    }
                )

    print(f"\n  Results exported to: {output_dir.resolve()}")


# ── Main comparison runner ───────────────────────────────────────────


def run_governance_comparison(
    output_dir: Path,
    levels: Optional[List[str]] = None,
    max_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    """Main entry point: run comparison across governance levels with REAL agents."""
    _verify_api_keys()

    tmp_dir = Path(tempfile.mkdtemp(prefix="gov_eval_"))

    # Load ONLY governance-specific scenarios (not ground_truth.json)
    if not RQ3_SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"Governance scenarios not found: {RQ3_SCENARIOS_PATH}")
    scenarios = json.loads(RQ3_SCENARIOS_PATH.read_text(encoding="utf-8"))

    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
        logger.info("Dry-run mode: running %d scenarios only", max_scenarios)

    # Determine which levels to run
    level_list = [GovernanceLevel.LOW, GovernanceLevel.MEDIUM, GovernanceLevel.HIGH]
    if levels:
        level_list = [GovernanceLevel(lv) for lv in levels]

    all_results: Dict[str, List[GovernanceScenarioResult]] = {}
    total_start = time.time()

    for level in level_list:
        print(f"\n{'=' * 60}")
        print(f"GOVERNANCE LEVEL: {level.value.upper()} (REAL LLM MODE)")
        print(f"{'=' * 60}")
        results = run_single_level(level, scenarios, tmp_dir)
        all_results[level.value] = results

    total_elapsed = time.time() - total_start

    # Compute comparison
    comparison = compute_comparison_table(all_results)
    comparison["execution_mode"] = "real_llm"
    comparison["total_wall_time_seconds"] = round(total_elapsed, 2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("RQ3 GOVERNANCE COMPARISON SUMMARY (REAL LLM MODE)")
    print(f"{'=' * 60}")
    for level_name, metrics in comparison["per_level"].items():
        print(f"\n  [{level_name.upper()}]")
        print(f"    Task Completion:    {metrics['task_completion_rate']:.1%}")
        print(f"    Autonomy Score:     {metrics['autonomy_score']:.1%}")
        print(f"    Intervention Rate:  {metrics['intervention_rate']:.1%}")
        print(f"    Avg Latency:        {metrics['avg_latency_ms']:.1f} ms")
        print(f"    False Positive:     {metrics['false_positive_rate']:.1%}")
        print(f"    Gov. Blocks:        {metrics['total_governance_blocks']}")
        print(f"    Gov. Mutations:     {metrics['total_governance_mutations']}")

    if comparison.get("tradeoffs"):
        t = comparison["tradeoffs"]
        print("\n  TRADE-OFF DELTAS (LOW -> HIGH):")
        print(f"    Completion drop:    {t['completion_delta']:+.1%}")
        print(f"    Autonomy drop:      {t['autonomy_delta']:+.1%}")
        print(f"    Intervention rise:  {t['intervention_delta']:+.1%}")
        print(f"    Latency increase:   {t['latency_delta_ms']:+.1f} ms")

    print(f"\n  Total wall time:      {total_elapsed:.1f}s")

    export_comparison(comparison, all_results, output_dir)
    return comparison


# ── pytest integration ───────────────────────────────────────────────


def test_governance_comparison_runs():
    """pytest entry point: run all governance levels and assert no crashes."""
    output_dir = Path(tempfile.mkdtemp(prefix="gov_eval_out_"))
    comparison = run_governance_comparison(output_dir)

    assert "per_level" in comparison
    assert len(comparison["per_level"]) == 3
    for level_name in ["low", "medium", "high"]:
        assert level_name in comparison["per_level"]
        metrics = comparison["per_level"][level_name]
        assert metrics["total_scenarios"] > 0


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ3 Governance Comparison Runner (REAL LLM mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rq3",
        help="Output directory for results (default: evaluation/results/rq3)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated levels to run (e.g. low,high)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 3 scenarios for verification",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RQ3: SAFETY/COMPLIANCE TRADE-OFF EVALUATION")
    print("REAL LLM MODE")
    print("=" * 60)

    levels = args.levels.split(",") if args.levels else None
    max_sc = 3 if args.dry_run else None
    run_governance_comparison(Path(args.output), levels=levels, max_scenarios=max_sc)
