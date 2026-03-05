# evaluation/rq2_harness.py
"""
RQ2 Evaluation Harness — Explainability & IEEE Compliance Metrics

Extends the existing evaluation framework (harness.py) with RQ2-specific
metrics: IEEE compliance rates, explainability coverage, PII handling,
and governance mechanism activity.

Usage:
    python -m evaluation.rq2_harness
    python -m evaluation.rq2_harness --output evaluation/results/rq2/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unittest.mock as _mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.governance.explainability import ExplainabilityEngine  # noqa: E402
from app.governance.ieee_compliance import IEEEComplianceChecker  # noqa: E402
from app.governance.message_envelope import wrap_response  # noqa: E402
from app.runtime.spine import RuntimeSpine  # noqa: E402
from app.runtime.trace import Trace  # noqa: E402

SCENARIOS_PATH = Path(__file__).resolve().parent / "scenarios" / "ground_truth.json"


# ── Result dataclass ─────────────────────────────────────────────────


@dataclass
class RQ2ScenarioResult:
    """RQ2-specific evaluation result for a single scenario."""

    scenario_id: str
    category: str
    description: str

    # IEEE compliance (0.0 - 1.0)
    p3394_compliance: float = 0.0
    ieee_2894_compliance: float = 0.0
    ieee_3152_compliance: float = 0.0
    overall_compliance: float = 0.0

    # Explainability
    explanation_levels_available: int = 0  # out of 3
    has_summary: bool = False
    has_detailed: bool = False
    has_full: bool = False
    provenance_present: bool = False
    agent_identity_disclosed: bool = False
    decisions_documented: int = 0

    # Governance activity
    guardrail_checks: int = 0
    guardrail_interventions: int = 0
    trace_event_count: int = 0

    # Timing
    latency_ms: float = 0.0
    error: Optional[str] = None


# ── Harness ──────────────────────────────────────────────────────────


class RQ2Harness:
    """Runs scenarios and evaluates RQ2 explainability/compliance metrics."""

    def __init__(
        self,
        spine: RuntimeSpine,
        scenarios_path: str | Path,
    ):
        self.spine = spine
        self.checker = IEEEComplianceChecker()
        self.explainer = ExplainabilityEngine()
        self.scenarios = self._load_scenarios(scenarios_path)

    def run_all(self) -> List[RQ2ScenarioResult]:
        results = []
        for sc in self.scenarios:
            results.append(self.run_scenario(sc))
        return results

    def run_scenario(self, scenario: Dict[str, Any]) -> RQ2ScenarioResult:
        """Execute a scenario and extract RQ2 metrics from governance data."""
        sc_id = scenario["id"]
        category = scenario["category"]
        description = scenario.get("description", "")
        turns = scenario.get("turns", [])
        thread_id = f"rq2_eval_{sc_id}"

        result = RQ2ScenarioResult(
            scenario_id=sc_id,
            category=category,
            description=description,
        )

        if not turns:
            result.error = "No turns in scenario"
            return result

        # Run first turn (RQ2 evaluates per-request governance, not multi-turn)
        query = turns[0]["query"]
        t0 = time.perf_counter()
        try:
            resp = self.spine.handle_chat(
                query,
                request_id=f"rq2_{sc_id}",
                context={"thread_id": thread_id},
            )
        except Exception as e:
            result.error = str(e)
            return result

        result.latency_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(resp, dict):
            result.error = "Response is not a dict"
            return result

        # Build trace, envelope, and explanations for evaluation
        trace = Trace.start(query=query, request_id=f"rq2_{sc_id}")
        # Simulate key trace events from response structure
        trace.add("request_received")
        if resp.get("orchestration_pattern"):
            trace.add("orchestration_pattern", pattern=resp["orchestration_pattern"])
        if resp.get("router_plan"):
            rp = resp["router_plan"]
            trace.add(
                "route",
                primary=rp.get("primary"),
                strategy=rp.get("strategy"),
                candidates=rp.get("candidates", []),
            )
        if resp.get("agent_id"):
            trace.add("execute", agent_id=resp["agent_id"])
            trace.add("select", selected_agent=resp["agent_id"], score=resp.get("score", 0))
        if resp.get("subtask_results"):
            trace.add(
                "aop_execute",
                results=[
                    {
                        "subtask": st.get("subtask"),
                        "agent": st.get("agent_id"),
                        "success": st.get("success"),
                    }
                    for st in resp["subtask_results"]
                ],
            )
        if resp.get("solvability"):
            trace.add("aop_solvability", assignments=resp["solvability"].get("assignments", {}))
        if resp.get("completeness"):
            trace.add(
                "aop_completeness",
                complete=resp["completeness"].get("complete"),
                missing=resp["completeness"].get("missing", []),
            )
        trace.add("guard_post_ok")

        ctx = {"thread_id": thread_id, "intent": "unknown"}

        # Generate explanations
        explanations = self.explainer.generate_all_levels(trace, resp)
        expl_dicts = {k: v.to_dict() for k, v in explanations.items()}

        # Generate envelope
        envelope = wrap_response(resp, trace, ctx)
        envelope_dict = envelope.to_dict()

        # Run compliance checks
        report = self.checker.check_all(
            message=envelope_dict,
            trace=trace,
            response=resp,
            explanations=expl_dicts,
            envelope=envelope_dict,
        )

        # Populate result
        by_std = report.by_standard
        result.p3394_compliance = by_std.get("P3394", 0.0)
        result.ieee_2894_compliance = by_std.get("2894-2024", 0.0)
        result.ieee_3152_compliance = by_std.get("3152-2024", 0.0)
        result.overall_compliance = report.compliance_rate

        # Explainability
        result.has_summary = "summary" in expl_dicts and bool(
            expl_dicts["summary"].get("narrative")
        )
        result.has_detailed = "detailed" in expl_dicts and bool(
            expl_dicts["detailed"].get("narrative")
        )
        result.has_full = "full" in expl_dicts and bool(expl_dicts["full"].get("narrative"))
        result.explanation_levels_available = sum(
            [result.has_summary, result.has_detailed, result.has_full]
        )

        # Provenance
        for level_data in expl_dicts.values():
            prov = level_data.get("provenance", [])
            if prov:
                result.provenance_present = True
                break

        # Agent identity
        result.agent_identity_disclosed = bool(resp.get("agent_id"))

        # Decisions documented
        detailed = expl_dicts.get("detailed", {})
        result.decisions_documented = len(detailed.get("decisions", []))

        # Governance activity (from trace)
        guard_stages = {"guard_pre_ok", "guard_post_ok", "guard_pre_block", "guard_post_block"}
        guard_events = [e for e in trace.events if e.stage in guard_stages]
        result.guardrail_checks = len(guard_events)
        result.guardrail_interventions = sum(1 for e in guard_events if "block" in e.stage)
        result.trace_event_count = len(trace.events)

        return result

    # ── Metrics ──────────────────────────────────────────────────────

    def compute_metrics(self, results: List[RQ2ScenarioResult]) -> Dict[str, Any]:
        """Compute aggregate RQ2 metrics."""
        if not results:
            return {}

        n = len(results)
        valid = [r for r in results if r.error is None]
        nv = len(valid) or 1  # avoid division by zero

        return {
            "total_scenarios": n,
            "successful": len(valid),
            "errors": n - len(valid),
            # IEEE compliance
            "p3394_compliance": round(sum(r.p3394_compliance for r in valid) / nv, 4),
            "ieee_2894_compliance": round(sum(r.ieee_2894_compliance for r in valid) / nv, 4),
            "ieee_3152_compliance": round(sum(r.ieee_3152_compliance for r in valid) / nv, 4),
            "overall_compliance": round(sum(r.overall_compliance for r in valid) / nv, 4),
            # Explainability
            "mean_explanation_levels": round(
                sum(r.explanation_levels_available for r in valid) / nv, 2
            ),
            "summary_coverage": round(sum(1 for r in valid if r.has_summary) / nv, 4),
            "detailed_coverage": round(sum(1 for r in valid if r.has_detailed) / nv, 4),
            "full_coverage": round(sum(1 for r in valid if r.has_full) / nv, 4),
            "provenance_rate": round(sum(1 for r in valid if r.provenance_present) / nv, 4),
            "agent_identity_rate": round(
                sum(1 for r in valid if r.agent_identity_disclosed) / nv, 4
            ),
            "mean_decisions_documented": round(sum(r.decisions_documented for r in valid) / nv, 2),
            # Governance
            "mean_guardrail_checks": round(sum(r.guardrail_checks for r in valid) / nv, 2),
            "total_guardrail_interventions": sum(r.guardrail_interventions for r in valid),
            "mean_trace_events": round(sum(r.trace_event_count for r in valid) / nv, 2),
            # Latency
            "mean_latency_ms": round(sum(r.latency_ms for r in valid) / nv, 2),
            # Per-category compliance
            "compliance_by_category": self._compliance_by_category(valid),
        }

    def _compliance_by_category(
        self, results: List[RQ2ScenarioResult]
    ) -> Dict[str, Dict[str, float]]:
        cats: Dict[str, List[RQ2ScenarioResult]] = {}
        for r in results:
            cats.setdefault(r.category, []).append(r)

        out = {}
        for cat, rs in sorted(cats.items()):
            nc = len(rs) or 1
            out[cat] = {
                "n": len(rs),
                "overall": round(sum(r.overall_compliance for r in rs) / nc, 4),
                "p3394": round(sum(r.p3394_compliance for r in rs) / nc, 4),
                "2894": round(sum(r.ieee_2894_compliance for r in rs) / nc, 4),
                "3152": round(sum(r.ieee_3152_compliance for r in rs) / nc, 4),
            }
        return out

    # ── Export ────────────────────────────────────────────────────────

    def export_csv(self, results: List[RQ2ScenarioResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "scenario_id",
            "category",
            "description",
            "p3394_compliance",
            "ieee_2894_compliance",
            "ieee_3152_compliance",
            "overall_compliance",
            "explanation_levels_available",
            "has_summary",
            "has_detailed",
            "has_full",
            "provenance_present",
            "agent_identity_disclosed",
            "decisions_documented",
            "guardrail_checks",
            "guardrail_interventions",
            "trace_event_count",
            "latency_ms",
            "error",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "scenario_id": r.scenario_id,
                        "category": r.category,
                        "description": r.description,
                        "p3394_compliance": round(r.p3394_compliance, 4),
                        "ieee_2894_compliance": round(r.ieee_2894_compliance, 4),
                        "ieee_3152_compliance": round(r.ieee_3152_compliance, 4),
                        "overall_compliance": round(r.overall_compliance, 4),
                        "explanation_levels_available": r.explanation_levels_available,
                        "has_summary": r.has_summary,
                        "has_detailed": r.has_detailed,
                        "has_full": r.has_full,
                        "provenance_present": r.provenance_present,
                        "agent_identity_disclosed": r.agent_identity_disclosed,
                        "decisions_documented": r.decisions_documented,
                        "guardrail_checks": r.guardrail_checks,
                        "guardrail_interventions": r.guardrail_interventions,
                        "trace_event_count": r.trace_event_count,
                        "latency_ms": round(r.latency_ms, 2),
                        "error": r.error,
                    }
                )

    def export_json(self, results: List[RQ2ScenarioResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        from dataclasses import asdict

        data = [asdict(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def _load_scenarios(path: str | Path) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scenarios file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))


# ── Runner ───────────────────────────────────────────────────────────


def run_rq2_evaluation(output_dir: Path) -> Dict[str, Any]:
    """Execute RQ2 evaluation using the same mock spine as RQ1."""
    import tempfile

    # Reuse the eval spine builder from run_evaluation.py
    from evaluation.run_evaluation import build_eval_spine
    from evaluation.mock_factory import build_scenario_mock

    tmp_dir = Path(tempfile.mkdtemp(prefix="rq2_eval_"))
    spine = build_eval_spine(tmp_dir)

    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    all_results: List[RQ2ScenarioResult] = []
    for sc in scenarios:
        mock_responses = sc.get("mock_responses", {})
        mock_fn = build_scenario_mock(mock_responses)

        def voice_mock(**kw):
            return {"messages": ["OK"], "quick_replies": []}

        with _mock.patch("app.llm_client.chat_json", mock_fn), _mock.patch(
            "app.orchestration.aop_coordinator.chat_json", mock_fn
        ), _mock.patch("app.orchestration.completeness_detector.chat_json", mock_fn), _mock.patch(
            "app.runtime.voice.chat_json", voice_mock
        ):
            try:
                _mock.patch("app.shared.workflow.chat_json", mock_fn).start()
            except AttributeError:
                pass

            harness = RQ2Harness(spine, SCENARIOS_PATH)
            result = harness.run_scenario(sc)
            all_results.append(result)

            status = "PASS" if result.error is None else "FAIL"
            print(
                f"  [{status}] {sc['id']:20s}  "
                f"compliance={result.overall_compliance:.0%}  "
                f"expl_levels={result.explanation_levels_available}/3  "
                f"latency={result.latency_ms:.0f}ms"
            )

    harness = RQ2Harness(spine, SCENARIOS_PATH)
    metrics = harness.compute_metrics(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("RQ2 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total scenarios:         {metrics.get('total_scenarios', 0)}")
    print(f"  Successful:              {metrics.get('successful', 0)}")
    print(f"  Overall Compliance:      {metrics.get('overall_compliance', 0):.1%}")
    print(f"  IEEE P3394:              {metrics.get('p3394_compliance', 0):.1%}")
    print(f"  IEEE 2894-2024:          {metrics.get('ieee_2894_compliance', 0):.1%}")
    print(f"  IEEE 3152-2024:          {metrics.get('ieee_3152_compliance', 0):.1%}")
    print(f"  Summary Coverage:        {metrics.get('summary_coverage', 0):.1%}")
    print(f"  Detailed Coverage:       {metrics.get('detailed_coverage', 0):.1%}")
    print(f"  Provenance Rate:         {metrics.get('provenance_rate', 0):.1%}")
    print(f"  Agent Identity Rate:     {metrics.get('agent_identity_rate', 0):.1%}")
    print(f"  Mean Latency:            {metrics.get('mean_latency_ms', 0):.1f} ms")

    by_cat = metrics.get("compliance_by_category", {})
    if by_cat:
        print("\n  Compliance by Category:")
        for cat, data in by_cat.items():
            print(f"    {cat:30s}  overall={data['overall']:.0%}  n={data['n']}")

    # Export
    output_dir.mkdir(parents=True, exist_ok=True)
    harness.export_csv(all_results, output_dir / "rq2_results.csv")
    harness.export_json(all_results, output_dir / "rq2_results.json")
    (output_dir / "rq2_summary.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  Results exported to: {output_dir.resolve()}")
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 Evaluation Harness")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rq2",
        help="Output directory",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RQ2 EVALUATION — Explainability & IEEE Compliance")
    print("=" * 60 + "\n")

    run_rq2_evaluation(Path(args.output))
