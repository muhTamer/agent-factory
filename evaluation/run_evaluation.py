# evaluation/run_evaluation.py
"""
Week 3 Evaluation Runner — DSRM Stage 5 (REAL LLM MODE)

Runs RQ1 orchestration evaluation using REAL agents with REAL LLM calls.
No mocks, no stubs — genuine ReAct reasoning, RAG retrieval, and tool execution.

Usage:
    python -m evaluation.run_evaluation                         # full run
    python -m evaluation.run_evaluation --output results/       # custom output dir
    python -m evaluation.run_evaluation --scenario deleg_01     # single scenario
    python -m evaluation.run_evaluation --dry-run               # 3 scenarios only
    pytest evaluation/run_evaluation.py -v                      # as pytest

Outputs:
    evaluation_results.csv    — one row per scenario
    evaluation_summary.json   — aggregate metrics
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Force UTF-8 stdout/stderr on Windows (avoids charmap codec errors) ──
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

# ── Project root on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.orchestration.aop_coordinator import AOPCoordinator  # noqa: E402
from app.orchestration.performance_store import PerformanceStore  # noqa: E402
from app.runtime.guardrails import NoOpGuardrails  # noqa: E402
from app.runtime.registry import AgentRegistry  # noqa: E402
from app.runtime.router import LLMRouter  # noqa: E402
from app.runtime.spine import RuntimeSpine  # noqa: E402

from evaluation.harness import EvaluationHarness, ScenarioResult  # noqa: E402

# ── Logging ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent_factory.log"

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_log_formatter)
_file_handler = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(level=logging.WARNING, handlers=[_stream_handler, _file_handler])

logger = logging.getLogger("evaluation.rq1")
logger.setLevel(logging.INFO)

# Suppress noisy libraries
for _noisy in ("httpx", "httpcore", "openai", "azure"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Scenarios path ──────────────────────────────────────────────────
SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "ground_truth.json"

# ── Rate limiting & timeouts ────────────────────────────────────────
INTER_SCENARIO_DELAY = float(os.getenv("EVAL_DELAY_SECONDS", "1.0"))
SCENARIO_TIMEOUT = int(os.getenv("EVAL_SCENARIO_TIMEOUT", "300"))  # 5 min default


# ── Load real agents from generated/ ────────────────────────────────


def _load_real_agents(registry: AgentRegistry) -> None:
    """Load real generated agents from the factory spec into the registry.

    Each agent uses DomainAgentEngine with:
      - Real LLM calls (chat_json → OpenAI/Azure)
      - Real RAG retrieval (TF-IDF corpus from .workspace/)
      - Real tool execution (stub tools for safety, real tool interface)
      - Real ReAct reasoning loop

    Set EVAL_SKIP_DENSE=1 to skip dense embedding pre-computation
    (agents fall back to TF-IDF only — much faster startup).
    """
    # Optionally disable dense retrieval to avoid slow embedding startup.
    # With EVAL_SKIP_DENSE=1 agents use TF-IDF only (still real retrieval,
    # just not hybrid). This cuts startup from ~5 min to ~5 sec.
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

        # Register with rich blueprint_meta for LLM router
        meta = agent_spec.get("blueprint_meta", {})
        meta["ready"] = True
        registry.register(agent_id, agent, meta)
        loaded += 1

    if loaded == 0:
        raise RuntimeError(
            "No real agents loaded. Ensure generated/ contains agent packages."
        )
    logger.info("Loaded %d real agents: %s", loaded, registry.all_ids())


# ── Verify LLM API connectivity ─────────────────────────────────────


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
    provider = "Azure OpenAI" if has_azure else "OpenAI"
    logger.info("LLM API provider: %s", provider)


# ── Build spine with REAL agents ─────────────────────────────────────


def build_eval_spine(tmp_dir: Path, estimator_kind: str = "llm") -> RuntimeSpine:
    """Create a RuntimeSpine with REAL agents and LLM-based routing.

    The real fleet (from factory_spec.json) has three domain_agent instances:
      - refunds_agent       — Refund processing (ReAct + RAG + tools)
      - complaints_agent    — Complaint handling (ReAct + RAG + tools)
      - faq_agent           — Multi-category FAQ (ReAct + RAG)
    FAQ/informational queries are handled by the most relevant domain agent.

    All agents use DomainAgentEngine with real LLM calls.
    """
    registry = AgentRegistry()
    _load_real_agents(registry)

    # Performance store for solvability estimation
    perf_store = PerformanceStore(path=str(tmp_dir / "eval_perf.json"))

    # AOP coordinator (uses real LLM for decomposition)
    aop = AOPCoordinator(registry=registry, performance_store=perf_store)

    # Swap solvability estimator (AOPCoordinator defaults to TF-IDF internally)
    aop.swap_estimator(estimator_kind)
    logger.info("Solvability estimator: %s", aop.active_estimator_kind)

    # LLM-based router (uses real LLM for intent classification + agent scoring)
    router = LLMRouter(registry)

    spine = RuntimeSpine(
        registry=registry,
        router=router,
        guardrails=NoOpGuardrails(),
        aop_coordinator=aop,
    )

    return spine


# ── Run evaluation ──────────────────────────────────────────────────


def run_evaluation(
    output_dir: Path,
    scenario_filter: Optional[str] = None,
    max_scenarios: Optional[int] = None,
    estimator_kind: str = "llm",
) -> Dict[str, Any]:
    """Execute all scenarios with REAL LLM calls and write results."""
    import tempfile

    _verify_api_keys()

    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_"))
    spine = build_eval_spine(tmp_dir, estimator_kind=estimator_kind)

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if scenario_filter:
        filter_ids = [f.strip() for f in scenario_filter.split(",")]
        scenarios = [s for s in scenarios if s["id"] in filter_ids]
        if not scenarios:
            print(f"[ERROR] No scenarios matching: {scenario_filter}")
            return {}
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
        logger.info("Dry-run mode: running %d scenarios only", max_scenarios)

    print(f"\nRunning {len(scenarios)} scenarios with REAL LLM agents...")
    print(f"Inter-scenario delay: {INTER_SCENARIO_DELAY}s")
    print()

    # Run each scenario — NO mocks, real LLM calls
    all_results = []
    total_start = time.time()
    passed_count = 0
    failed_count = 0

    logger.info("=" * 60)
    logger.info("AGENT FACTORY — Starting %d scenarios", len(scenarios))
    logger.info("Log file: %s", LOG_FILE.resolve())
    logger.info("=" * 60)

    for i, sc in enumerate(scenarios, 1):
        sc_start = time.time()
        logger.info(
            "[%d/%d] ▶ %s (category=%s)",
            i,
            len(scenarios),
            sc["id"],
            sc["category"],
        )

        harness = EvaluationHarness(spine, SCENARIOS_PATH)

        # Run with thread-based timeout to prevent hanging on API calls
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(harness.run_scenario, sc)
                result = future.result(timeout=SCENARIO_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "[%d/%d] ✗ %s TIMEOUT after %ds",
                i,
                len(scenarios),
                sc["id"],
                SCENARIO_TIMEOUT,
            )
            result = ScenarioResult(
                scenario_id=sc["id"],
                category=sc.get("category", ""),
                description=sc.get("description", ""),
                success=False,
                outcome_detail=f"timeout:{SCENARIO_TIMEOUT}s",
                error=f"Timeout after {SCENARIO_TIMEOUT}s",
            )

        all_results.append(result)

        if result.success:
            passed_count += 1
        else:
            failed_count += 1

        elapsed = (time.time() - sc_start) * 1000
        elapsed_total = time.time() - total_start
        avg_per_sc = elapsed_total / i
        eta = avg_per_sc * (len(scenarios) - i)

        status = "SOFT" if result.soft_pass else ("PASS" if result.success else "FAIL")
        log_line = (
            f"  [{status}] {sc['id']:20s}  "
            f"pattern={'OK' if result.pattern_correct else 'MISS':4s}  "
            f"agent={'OK' if result.agent_correct else 'MISS':4s}  "
            f"latency={elapsed:.0f}ms  "
            f"[{passed_count}✓ {failed_count}✗ | "
            f"ETA {int(eta//60)}m{int(eta%60):02d}s]"
        )
        logger.info(log_line)
        sys.stdout.flush()

        # Rate limiting between scenarios
        if i < len(scenarios):
            time.sleep(INTER_SCENARIO_DELAY)

    total_elapsed = time.time() - total_start

    # Compute metrics
    harness = EvaluationHarness(spine, SCENARIOS_PATH)
    metrics = harness.compute_metrics(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (REAL LLM MODE)")
    print("=" * 60)
    print(f"  Total scenarios:         {metrics.get('total_scenarios', 0)}")
    print(f"  Passed:                  {metrics.get('passed', 0)}")
    print(f"  Failed:                  {metrics.get('failed', 0)}")
    print(f"  Orchestration Accuracy:  {metrics.get('orchestration_accuracy', 0):.1%}")
    print(f"  Reasoning Accuracy:      {metrics.get('reasoning_accuracy', 0):.1%}")
    print(f"  Agent Accuracy:          {metrics.get('agent_accuracy', 0):.1%}")
    print(f"  Outcome Accuracy:        {metrics.get('outcome_accuracy', 0):.1%}")
    print(f"  Avg Latency:             {metrics.get('avg_latency_ms', 0):.1f} ms")

    solv = metrics.get("solvability_correlation")
    print(
        f"  Solvability Correlation: {solv:.4f}"
        if solv is not None
        else "  Solvability Correlation: N/A"
    )

    comp = metrics.get("completeness_rate")
    print(
        f"  Completeness Rate:       {comp:.1%}"
        if comp is not None
        else "  Completeness Rate:       N/A"
    )

    print(f"\n  Total wall time:         {total_elapsed:.1f}s")

    print("\n  Latency by category:")
    for cat, lat in metrics.get("latency_by_category", {}).items():
        steps = metrics.get("steps_by_category", {}).get(cat, 0)
        print(f"    {cat:30s}  {lat:8.1f} ms  ({steps:.1f} agent calls)")

    # Export results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness.export_csv(all_results, output_dir / "evaluation_results.csv")
    harness.export_json(all_results, output_dir / "evaluation_results.json")

    # Add execution metadata to summary
    metrics["execution_mode"] = "real_llm"
    metrics["total_wall_time_seconds"] = round(total_elapsed, 2)
    metrics["inter_scenario_delay_seconds"] = INTER_SCENARIO_DELAY

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── pytest integration ──────────────────────────────────────────────


def test_all_scenarios_pass():
    """pytest entry point: run all scenarios and assert targets met."""
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="eval_out_"))
    metrics = run_evaluation(output_dir)

    assert (
        metrics["total_scenarios"] >= 25
    ), f"Expected >=25 scenarios, got {metrics['total_scenarios']}"
    assert (
        metrics["orchestration_accuracy"] >= 0.80
    ), f"Orchestration accuracy {metrics['orchestration_accuracy']:.1%} < 80% target"
    assert (
        metrics["reasoning_accuracy"] >= 0.75
    ), f"Reasoning accuracy {metrics['reasoning_accuracy']:.1%} < 75% target"


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RQ1 evaluation (REAL LLM mode)")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results",
        help="Output directory for results (default: evaluation/results)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run a single scenario by ID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 3 scenarios for verification",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="llm",
        choices=["neural", "tfidf", "llm"],
        help="Solvability estimator to use (default: llm)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AGENT FACTORY ORCHESTRATION — EVALUATION HARNESS")
    print(f"DSRM Stage 5 | REAL LLM MODE | Estimator: {args.estimator}")
    print("=" * 60 + "\n")

    max_sc = 3 if args.dry_run else None
    run_evaluation(
        Path(args.output),
        scenario_filter=args.scenario,
        max_scenarios=max_sc,
        estimator_kind=args.estimator,
    )
