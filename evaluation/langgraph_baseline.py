# evaluation/langgraph_baseline.py
"""
LangGraph Supervisor Baseline for RQ1 Comparison.

Replicates the same 3 domain agents (FAQ, Refunds, Complaints) using
LangGraph's Supervisor pattern to provide an independent orchestration
baseline. Same model (gpt-5-mini), same knowledge, same scenarios.

Architecture: Supervisor → routes to one agent → agent answers → done.
This is architecturally closest to our Spine → Router → Agent pattern.

Usage:
    python -m evaluation.langgraph_baseline
    python -m evaluation.langgraph_baseline --dry-run
    python -m evaluation.langgraph_baseline --scenario faq_01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Force UTF-8 stdout/stderr on Windows ─────────────────────────────
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from langchain_openai import AzureChatOpenAI  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from langgraph_supervisor import create_supervisor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluation.langgraph_baseline")

SCENARIOS_PATH = ROOT / "evaluation" / "scenarios" / "ground_truth.json"
INTER_SCENARIO_DELAY = float(os.getenv("EVAL_DELAY_SECONDS", "1.0"))


# ── TF-IDF RAG (same as AutoGen baseline) ────────────────────────────


class SimpleRAG:
    """Lightweight TF-IDF search over a corpus.json file."""

    def __init__(self, corpus_path: Path):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._cosine_similarity = cosine_similarity
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        self._texts = [item["text"] for item in data]
        self._sources = [item.get("source", "") for item in data]
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._vectorizer.fit_transform(self._texts)

    def search(self, query: str, top_k: int = 3) -> str:
        q_vec = self._vectorizer.transform([query])
        scores = self._cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.01:
                results.append(f"[{self._sources[idx]}] {self._texts[idx]}")
        if not results:
            return "No relevant knowledge found."
        return "\n\n".join(results)


# ── Build LangGraph model ─────────────────────────────────────────────


def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        temperature=1.0,
    )


# ── Build LangGraph agents + supervisor ───────────────────────────────


def _build_supervisor(llm: AzureChatOpenAI):
    """Create supervisor with 3 domain agents matching our generated agents."""
    gen_dir = ROOT / "generated"
    agents = []

    agent_configs = [
        ("customer_faq_agent_v1", "faq"),
        ("refunds_agent_v1", "refunds"),
        ("complaints_agent_v1", "complaints"),
    ]

    for agent_id, domain in agent_configs:
        config_path = gen_dir / agent_id / "config.json"
        corpus_path = gen_dir / agent_id / "corpus.json"

        config = json.loads(config_path.read_text(encoding="utf-8"))
        rag = SimpleRAG(corpus_path)

        # Create tool function (closure)
        def make_search_tool(rag_instance: SimpleRAG, name: str):
            def knowledge_search(query: str) -> str:
                """Search the knowledge base for relevant information about the query."""
                return rag_instance.search(query, top_k=3)

            knowledge_search.__name__ = f"{name}_knowledge_search"
            knowledge_search.__qualname__ = f"{name}_knowledge_search"
            return knowledge_search

        search_tool = make_search_tool(rag, domain)

        policies_text = "\n".join(f"- {p}" for p in config.get("policies", []))
        system_message = (
            f"You are {agent_id}, a specialist in the {domain} domain.\n\n"
            f"GOAL: {config['goal']}\n\n"
            f"POLICIES:\n{policies_text}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Use the knowledge_search tool to find relevant information before answering.\n"
            f"- Ground your answers in the knowledge base results.\n"
            f"- If the query is outside your domain, say so clearly.\n"
            f"- Be concise and helpful.\n"
        )

        agent = create_react_agent(
            model=llm,
            tools=[search_tool],
            name=agent_id,
            prompt=system_message,
        )
        agents.append(agent)

    # Supervisor routes to the correct agent (like our Spine)
    supervisor = create_supervisor(
        agents=agents,
        model=llm,
        prompt=(
            "You are a banking customer service supervisor. "
            "Route each customer query to the most appropriate specialist agent:\n"
            "- customer_faq_agent_v1: informational queries about accounts, banking products, procedures\n"
            "- refunds_agent_v1: refund requests, payment disputes, transaction reversals\n"
            "- complaints_agent_v1: customer complaints, service issues, escalations\n\n"
            "For queries involving multiple concerns, route to the primary agent first. "
            "Only hand off to ONE agent per query."
        ),
        output_mode="last_message",
        supervisor_name="supervisor",
    )

    return supervisor.compile()


# ── Scenario runner ───────────────────────────────────────────────────


def _detect_pattern(scenario: Dict[str, Any], agents_involved: List[str]) -> str:
    """Detect orchestration pattern from LangGraph execution."""
    expected = scenario.get("turns", [{}])[0].get("expected", {}).get("pattern")
    if expected == "hierarchical_delegation":
        if len(set(agents_involved)) > 1:
            return "hierarchical_delegation"
        return "direct"
    return "direct"


def run_single_scenario(supervisor, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single scenario through LangGraph supervisor."""
    sc_id = scenario["id"]
    category = scenario["category"]
    turns = scenario.get("turns", [])
    query = turns[0]["query"] if turns else ""
    expected = turns[0].get("expected", {}) if turns else {}

    t0 = time.perf_counter()

    try:
        result = supervisor.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 20},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        messages = result.get("messages", [])

        # Extract agent routing — find which agent(s) were called
        first_agent = ""
        responding_agent = ""
        answer_text = ""
        agents_involved = []

        for m in messages:
            name = getattr(m, "name", "") or ""
            msg_type = type(m).__name__

            # Track agent involvement (skip supervisor and user)
            if name and name != "supervisor" and msg_type != "HumanMessage":
                if name not in agents_involved:
                    agents_involved.append(name)
                if not first_agent:
                    first_agent = name

            # Get the final answer text
            if msg_type == "AIMessage" and hasattr(m, "content"):
                content = m.content
                if isinstance(content, str) and content.strip():
                    responding_agent = name or responding_agent
                    answer_text = content

        # Check pattern
        expected_pattern = expected.get("pattern")
        actual_pattern = _detect_pattern(scenario, agents_involved)
        pattern_ok = (expected_pattern is None) or (actual_pattern == expected_pattern)

        # Check agent — use first routed agent
        expected_agent = expected.get("agent_contains")
        agent_ok = True
        if expected_agent is not None:
            agent_ok = expected_agent.lower() in (first_agent or "").lower()

        # Check answer keywords
        expected_keywords = expected.get("answer_contains", [])
        keywords_ok = (
            all(kw.lower() in answer_text.lower() for kw in expected_keywords)
            if expected_keywords
            else True
        )

        success = pattern_ok and agent_ok and keywords_ok

        return {
            "scenario_id": sc_id,
            "category": category,
            "description": scenario.get("description", ""),
            "success": success,
            "pattern_correct": pattern_ok,
            "agent_correct": agent_ok,
            "answer_keywords_found": keywords_ok,
            "latency_ms": latency_ms,
            "first_agent": first_agent,
            "responding_agent": responding_agent,
            "agents_involved": agents_involved,
            "answer_text": answer_text[:500],
            "error": None,
        }

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.error("Scenario %s failed: %s", sc_id, e)
        return {
            "scenario_id": sc_id,
            "category": category,
            "description": scenario.get("description", ""),
            "success": False,
            "pattern_correct": False,
            "agent_correct": False,
            "answer_keywords_found": False,
            "latency_ms": latency_ms,
            "first_agent": "",
            "responding_agent": "",
            "agents_involved": [],
            "answer_text": "",
            "error": str(e),
        }


# ── Main evaluation loop ─────────────────────────────────────────────


def run_langgraph_evaluation(
    output_dir: Path,
    scenario_filter: Optional[str] = None,
    max_scenarios: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all scenarios through LangGraph supervisor and compare."""

    llm = _build_llm()
    supervisor = _build_supervisor(llm)

    logger.info("Built LangGraph supervisor with 3 domain agents")

    # Load scenarios
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    if scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == scenario_filter]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    print(f"\nRunning {len(scenarios)} scenarios with LangGraph Supervisor...")
    print(f"Model: {os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-5-mini')}")
    print(f"Inter-scenario delay: {INTER_SCENARIO_DELAY}s\n")

    all_results = []
    total_start = time.time()

    for i, sc in enumerate(scenarios, 1):
        logger.info("[%d/%d] Running scenario: %s", i, len(scenarios), sc["id"])

        result = run_single_scenario(supervisor, sc)
        all_results.append(result)

        status = "PASS" if result["success"] else "FAIL"
        print(
            f"  [{status}] {sc['id']:20s}  "
            f"agent={result['first_agent']:30s}  "
            f"pattern={'OK' if result['pattern_correct'] else 'MISS':4s}  "
            f"keywords={'OK' if result['answer_keywords_found'] else 'MISS':4s}  "
            f"latency={result['latency_ms']:.0f}ms"
        )

        if i < len(scenarios):
            time.sleep(INTER_SCENARIO_DELAY)

    total_elapsed = time.time() - total_start

    # Compute metrics
    n = len(all_results)
    metrics = {
        "framework": "langgraph_supervisor",
        "total_scenarios": n,
        "passed": sum(1 for r in all_results if r["success"]),
        "failed": sum(1 for r in all_results if not r["success"]),
        "orchestration_accuracy": (
            round(sum(1 for r in all_results if r["pattern_correct"]) / n, 4)
            if n
            else 0
        ),
        "agent_accuracy": (
            round(sum(1 for r in all_results if r["agent_correct"]) / n, 4) if n else 0
        ),
        "reasoning_accuracy": (
            round(sum(1 for r in all_results if r["success"]) / n, 4) if n else 0
        ),
        "avg_latency_ms": (
            round(sum(r["latency_ms"] for r in all_results) / n, 2) if n else 0
        ),
        "execution_mode": "real_llm",
        "total_wall_time_seconds": round(total_elapsed, 2),
    }

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in all_results))
    metrics["by_category"] = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cn = len(cat_results)
        metrics["by_category"][cat] = {
            "n": cn,
            "orchestration_accuracy": round(
                sum(1 for r in cat_results if r["pattern_correct"]) / cn, 4
            ),
            "agent_accuracy": round(
                sum(1 for r in cat_results if r["agent_correct"]) / cn, 4
            ),
            "reasoning_accuracy": round(
                sum(1 for r in cat_results if r["success"]) / cn, 4
            ),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cat_results) / cn, 2),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("LANGGRAPH SUPERVISOR BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Total scenarios:         {metrics['total_scenarios']}")
    print(f"  Passed:                  {metrics['passed']}")
    print(f"  Failed:                  {metrics['failed']}")
    print(f"  Orchestration Accuracy:  {metrics['orchestration_accuracy']:.1%}")
    print(f"  Agent Accuracy:          {metrics['agent_accuracy']:.1%}")
    print(f"  Reasoning Accuracy:      {metrics['reasoning_accuracy']:.1%}")
    print(f"  Avg Latency:             {metrics['avg_latency_ms']:.1f} ms")
    print(f"  Total wall time:         {total_elapsed:.1f}s")

    print("\n  By category:")
    for cat, cm in metrics["by_category"].items():
        print(
            f"    {cat:30s}  n={cm['n']:2d}  "
            f"orch={cm['orchestration_accuracy']:.1%}  "
            f"agent={cm['agent_accuracy']:.1%}  "
            f"latency={cm['avg_latency_ms']:.0f}ms"
        )

    # Export
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "langgraph_baseline_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_path = output_dir / "langgraph_baseline_summary.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LangGraph Supervisor baseline for RQ1 comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/langgraph_baseline",
    )
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("LANGGRAPH SUPERVISOR BASELINE — RQ1 COMPARISON")
    print("Same agents, same model, same scenarios")
    print("=" * 60 + "\n")

    max_sc = 3 if args.dry_run else None
    run_langgraph_evaluation(
        Path(args.output),
        scenario_filter=args.scenario,
        max_scenarios=max_sc,
    )
