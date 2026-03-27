"""Quick smoke test: 2 scenarios per category to verify outcome + policy checks."""

from __future__ import annotations
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for _s in ("stdout", "stderr"):
        s = getattr(sys, _s, None)
        if s and hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pick scenarios per category — includes multi-turn, policy-compliance, and scope checks
SMOKE_IDS = [
    # informational_routing: multi-turn FAQ
    "faq_06",
    "faq_08",
    "b77_faq_05",
    # actionable_routing (refund/complaint policy checks)
    "refund_01",
    "b77_complaint_02",
    # hierarchical_delegation
    "deleg_04",
    "b77_deleg_02",
    # hitl_escalation (manager approval policy check)
    "hitl_01",
    "hitl_06",
    # graceful_degradation
    "edge_01",
    "edge_02",
]


def main():
    from evaluation.run_evaluation import (
        run_evaluation,
        SCENARIOS_PATH,
    )
    import tempfile

    all_scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    subset = [s for s in all_scenarios if s["id"] in SMOKE_IDS]
    print(f"Smoke test: {len(subset)} scenarios ({len(SMOKE_IDS)} requested)")
    for s in subset:
        n_turns = len(s.get("turns", []))
        last_turn = s["turns"][-1]["expected"].get("expected_outcome", {})
        kw = last_turn.get("answer_contains_any", []) or s.get(
            "expected_outcome", {}
        ).get("answer_contains_any", [])
        blacklist = last_turn.get("answer_not_contains", [])
        extra = f" blacklist={blacklist}" if blacklist else ""
        print(
            f"  {s['id']:<20} turns={n_turns} category={s['category']:<28} kw={kw}{extra}"
        )

    # Temporarily patch SCENARIOS_PATH content
    smoke_path = Path(tempfile.mkdtemp()) / "smoke_ground_truth.json"
    smoke_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")

    # Monkey-patch the path
    import evaluation.run_evaluation as mod

    mod.SCENARIOS_PATH = smoke_path

    output_dir = Path("evaluation/results/smoke_test")
    metrics = run_evaluation(output_dir=output_dir)

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"  Passed: {metrics.get('passed', 0)}/{metrics.get('total_scenarios', 0)}")
    print(f"  Failed: {metrics.get('failed', 0)}")
    return metrics


if __name__ == "__main__":
    main()
