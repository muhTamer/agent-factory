"""
Retry AutoGen baseline scenarios that failed due to timeout / rate-limiting.

Reads the existing results JSON, identifies timeout failures, reruns them
one-by-one with extra delay, then merges results back and recomputes summary.

Usage:
    python -m evaluation.retry_autogen_timeouts
    python -m evaluation.retry_autogen_timeouts --delay 20 --timeout 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results" / "autogen_baseline"
RESULTS_FILE = RESULTS_DIR / "autogen_baseline_results.json"
SUMMARY_FILE = RESULTS_DIR / "autogen_baseline_summary.json"
SCENARIOS_PATH = ROOT / "evaluation" / "scenarios" / "ground_truth.json"

LOG = logging.getLogger("evaluation.retry_autogen_timeouts")


def identify_timeout_failures(results: List[Dict]) -> List[str]:
    """Return scenario_ids that failed due to timeout / rate-limiting."""
    timeout_ids = []
    for r in results:
        if not r["success"]:
            detail = r.get("outcome_detail", "") or ""
            error = r.get("error", "") or ""
            is_timeout = (
                "timeout" in error.lower()
                or "timeout" in detail.lower()
                or (not r.get("responding_agent") and not r.get("first_agent"))
                or r.get("latency_ms", 0) >= 290000
            )
            if is_timeout:
                timeout_ids.append(r["scenario_id"])
    return timeout_ids


def recompute_summary(all_results: List[Dict], wall_time: float) -> Dict:
    """Recompute summary metrics from merged results."""
    n = len(all_results)
    if n == 0:
        return {"total_scenarios": 0}

    metrics = {
        "framework": "autogen_agentchat",
        "total_scenarios": n,
        "passed": sum(1 for r in all_results if r["success"]),
        "soft_passed": sum(1 for r in all_results if r.get("soft_pass")),
        "failed": sum(1 for r in all_results if not r["success"]),
        "orchestration_accuracy": round(
            sum(1 for r in all_results if r["pattern_correct"]) / n, 4
        ),
        "agent_accuracy": round(
            sum(1 for r in all_results if r["agent_correct"]) / n, 4
        ),
        "outcome_accuracy": round(
            sum(1 for r in all_results if r.get("outcome_correct", False)) / n, 4
        ),
        "reasoning_accuracy": round(sum(1 for r in all_results if r["success"]) / n, 4),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in all_results) / n, 2),
        "execution_mode": "real_llm",
        "total_wall_time_seconds": round(wall_time, 2),
    }

    categories = sorted(set(r["category"] for r in all_results))
    metrics["by_category"] = {}
    for cat in categories:
        cr = [r for r in all_results if r["category"] == cat]
        cn = len(cr)
        metrics["by_category"][cat] = {
            "n": cn,
            "orchestration_accuracy": round(
                sum(1 for r in cr if r["pattern_correct"]) / cn, 4
            ),
            "agent_accuracy": round(sum(1 for r in cr if r["agent_correct"]) / cn, 4),
            "outcome_accuracy": round(
                sum(1 for r in cr if r.get("outcome_correct", False)) / cn, 4
            ),
            "reasoning_accuracy": round(sum(1 for r in cr if r["success"]) / cn, 4),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in cr) / cn, 2),
        }
    return metrics


async def run_retry_round(
    all_results: List[Dict],
    all_scenarios: List[Dict],
    agents,
    model_client,
    run_single_scenario,
    delay: float,
    timeout: int,
    round_num: int,
) -> List[Dict]:
    """Run one retry round for all timeout failures. Returns merged results."""
    timeout_ids = identify_timeout_failures(all_results)
    if not timeout_ids:
        return all_results

    retry_scenarios = [s for s in all_scenarios if s["id"] in timeout_ids]
    LOG.info("")
    LOG.info("=" * 60)
    LOG.info(
        "RETRY ROUND %d — %d timeout failures to retry", round_num, len(retry_scenarios)
    )
    LOG.info("Scenarios: %s", [s["id"] for s in retry_scenarios])
    LOG.info("=" * 60)

    retry_results: Dict[str, Dict] = {}

    for i, sc in enumerate(retry_scenarios, 1):
        sid = sc["id"]
        LOG.info(
            "[%d/%d] RETRY ▶ %s (category=%s)",
            i,
            len(retry_scenarios),
            sid,
            sc["category"],
        )

        sc_start = time.time()
        try:
            result = await asyncio.wait_for(
                run_single_scenario(agents, sc, model_client),
                timeout=timeout,
            )
            result["retry_run"] = True
            result["retry_round"] = round_num
        except asyncio.TimeoutError:
            sc_elapsed = (time.time() - sc_start) * 1000.0
            LOG.error("  [FAIL] %s TIMEOUT again after %ds", sid, timeout)
            result = {
                "scenario_id": sid,
                "category": sc["category"],
                "description": sc.get("description", ""),
                "success": False,
                "soft_pass": False,
                "pattern_correct": False,
                "agent_correct": False,
                "answer_keywords_found": False,
                "outcome_correct": False,
                "outcome_detail": f"timeout:{timeout}s (retry round {round_num})",
                "latency_ms": sc_elapsed,
                "first_agent": "",
                "responding_agent": "",
                "agents_involved": [],
                "tools_used": [],
                "answer_text": "",
                "error": f"Timeout after {timeout}s (retry round {round_num})",
                "retry_run": True,
                "retry_round": round_num,
            }

        retry_results[sid] = result
        status = (
            "SOFT"
            if result.get("soft_pass")
            else ("PASS" if result["success"] else "FAIL")
        )
        LOG.info(
            "  [%s] %s  agent=%-25s  latency=%dms",
            status,
            sid,
            result.get("responding_agent", ""),
            round(result["latency_ms"]),
        )

        if i < len(retry_scenarios):
            LOG.info("  Waiting %ds before next...", int(delay))
            await asyncio.sleep(delay)

    # Merge: replace timeout failures with retry results
    merged = []
    for r in all_results:
        if r["scenario_id"] in retry_results:
            merged.append(retry_results[r["scenario_id"]])
        else:
            merged.append(r)

    # Report round results
    retry_pass = sum(1 for r in retry_results.values() if r["success"])
    retry_fail = sum(1 for r in retry_results.values() if not r["success"])
    LOG.info(
        "Round %d done: %d PASS, %d still failing", round_num, retry_pass, retry_fail
    )

    return merged


async def main(delay: float, timeout: int, max_rounds: int = 10):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                ROOT / "evaluation" / "logs" / "autogen_retry.log",
                mode="w",
                encoding="utf-8",
            ),
        ],
    )

    # ── Load existing results ──────────────────────────────────────
    if not RESULTS_FILE.exists():
        LOG.error("No results at %s — run full baseline first", RESULTS_FILE)
        return
    all_results: List[Dict] = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    LOG.info("Loaded %d existing results", len(all_results))

    # ── Check if anything to retry ─────────────────────────────────
    timeout_ids = identify_timeout_failures(all_results)
    if not timeout_ids:
        LOG.info("No timeout failures — nothing to retry!")
        return
    LOG.info("Found %d initial timeout failures", len(timeout_ids))

    # ── Backup originals (once) ────────────────────────────────────
    backup_r = RESULTS_DIR / "autogen_baseline_results_pre_retry.json"
    backup_s = RESULTS_DIR / "autogen_baseline_summary_pre_retry.json"
    if not backup_r.exists():
        shutil.copy2(RESULTS_FILE, backup_r)
        if SUMMARY_FILE.exists():
            shutil.copy2(SUMMARY_FILE, backup_s)
        LOG.info("Backed up originals to *_pre_retry.json")

    # ── Load scenarios ─────────────────────────────────────────────
    all_scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    # ── Import & build AutoGen agents ──────────────────────────────
    from evaluation.autogen_baseline import (
        _build_agents,
        _build_model_client,
        run_single_scenario,
    )

    model_client = _build_model_client()
    agents = _build_agents(model_client)
    LOG.info("Built %d AutoGen agents", len(agents))

    # ── Retry rounds loop — keep going until zero timeouts ─────────
    total_start = time.time()
    merged = all_results

    for round_num in range(1, max_rounds + 1):
        remaining = identify_timeout_failures(merged)
        if not remaining:
            LOG.info("All timeout failures resolved after %d round(s)!", round_num - 1)
            break

        LOG.info(
            "%d timeout failures remaining before round %d", len(remaining), round_num
        )

        # Increase delay progressively to give rate limits more room
        round_delay = delay + (round_num - 1) * 5
        LOG.info("Using delay=%ds for round %d", int(round_delay), round_num)

        merged = await run_retry_round(
            merged,
            all_scenarios,
            agents,
            model_client,
            run_single_scenario,
            round_delay,
            timeout,
            round_num,
        )

        # Save intermediate results after each round
        RESULTS_FILE.write_text(
            json.dumps(merged, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        LOG.info("Saved intermediate results after round %d", round_num)
    else:
        remaining = identify_timeout_failures(merged)
        if remaining:
            LOG.warning(
                "Reached max %d rounds with %d timeouts still remaining: %s",
                max_rounds,
                len(remaining),
                remaining,
            )

    total_elapsed = time.time() - total_start

    # Read original wall time
    orig_wall = 0
    if backup_s.exists():
        orig_summary = json.loads(backup_s.read_text(encoding="utf-8"))
        orig_wall = orig_summary.get("total_wall_time_seconds", 0)

    # ── Recompute & save final ─────────────────────────────────────
    metrics = recompute_summary(merged, orig_wall + total_elapsed)

    RESULTS_FILE.write_text(
        json.dumps(merged, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    SUMMARY_FILE.write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )

    # ── Report ─────────────────────────────────────────────────────
    # Compare merged results against originals to find retried scenarios
    retried = [r for r in merged if r.get("retry_run")]
    retry_pass = sum(1 for r in retried if r["success"])
    retry_soft = sum(1 for r in retried if r.get("soft_pass"))
    retry_fail = sum(1 for r in retried if not r["success"])

    LOG.info("")
    LOG.info("=" * 60)
    LOG.info("RETRY COMPLETE")
    LOG.info("=" * 60)
    LOG.info(
        "  Retried: %d | %d PASS, %d SOFT, %d FAIL",
        len(retried),
        retry_pass,
        retry_soft,
        retry_fail,
    )
    LOG.info("")
    LOG.info("MERGED SUMMARY (all %d scenarios):", metrics["total_scenarios"])
    LOG.info(
        "  Passed:                 %d (was %d)",
        metrics["passed"],
        sum(1 for r in all_results if r["success"]),
    )
    LOG.info("  Soft-passed:            %d", metrics["soft_passed"])
    LOG.info(
        "  Failed:                 %d (was %d)",
        metrics["failed"],
        sum(1 for r in all_results if not r["success"]),
    )
    LOG.info(
        "  Orchestration Accuracy: %.1f%%", metrics["orchestration_accuracy"] * 100
    )
    LOG.info("  Agent Accuracy:         %.1f%%", metrics["agent_accuracy"] * 100)
    LOG.info("  Outcome Accuracy:       %.1f%%", metrics["outcome_accuracy"] * 100)
    LOG.info("  Avg Latency:            %.0fms", metrics["avg_latency_ms"])
    LOG.info("")
    LOG.info("  Saved to: %s", RESULTS_DIR.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retry AutoGen timeout failures with extended timeout"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=20.0,
        help="Seconds between retries (default: 20)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-scenario timeout seconds (default: 600)",
    )
    args = parser.parse_args()

    (ROOT / "evaluation" / "logs").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AUTOGEN BASELINE — RETRY TIMEOUT FAILURES")
    print(f"  Inter-scenario delay: {args.delay}s")
    print(f"  Per-scenario timeout: {args.timeout}s (was 300s)")
    print("=" * 60 + "\n")

    asyncio.run(main(delay=args.delay, timeout=args.timeout))
