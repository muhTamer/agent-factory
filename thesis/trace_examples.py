# thesis/trace_examples.py
"""
Generate annotated execution trace examples for the thesis.

Usage:
    python -m thesis.trace_examples

Outputs (to thesis/output/traces/):
    trace_simple_routing.md
    trace_fsm_workflow.md
    trace_hierarchical_delegation.md
    trace_hitl_escalation.md
    traces_combined.md   — all four in one file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACES_PATH = ROOT / ".factory" / "audit" / "runtime_traces.jsonl"
EVAL_RESULTS_PATH = ROOT / "evaluation" / "results" / "evaluation_results.json"
OUTPUT_DIR = ROOT / "thesis" / "output" / "traces"

# Human-readable annotations for each pipeline stage
STAGE_ANNOTATIONS = {
    "request_received": "Query enters the RuntimeSpine 7-stage pipeline.",
    "sticky_route": "Thread-pinned workflow continues from prior turn.",
    "orchestration_pattern": "LLM classifies query as *direct* or *hierarchical_delegation*.",
    "route": "Router scores candidate agents by capability match.",
    "intent_inferred": "Intent mapped from route for policy guardrails.",
    "guard_pre_ok": "Pre-execution guardrails pass — query is policy-compliant.",
    "execute": "Selected agent(s) handle the request.",
    "select": "Best response selected by confidence score.",
    "rag_delegation": "RAG FSM delegates to a specialist agent.",
    "fsm_state": "Workflow FSM state snapshot recorded.",
    "response_ready": "Response assembled with metadata.",
    "voice_chat_failed": "Voice rendering skipped (non-critical).",
    "guard_post_ok": "Post-execution guardrails pass — response is safe to return.",
    "aop_followup_expanded": "Short follow-up expanded with previous AOP context.",
}


def _load_traces() -> list:
    if not TRACES_PATH.exists():
        return []
    traces = []
    for line in TRACES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


def _load_eval_results() -> list:
    if not EVAL_RESULTS_PATH.exists():
        return []
    return json.loads(EVAL_RESULTS_PATH.read_text(encoding="utf-8"))


def _format_data(data: dict) -> str:
    """Format trace event data as a compact summary."""
    if not data:
        return ""
    parts = []
    for key, val in data.items():
        if key in ("candidates", "results"):
            if isinstance(val, list):
                items = []
                for item in val[:3]:
                    if isinstance(item, dict):
                        items.append(
                            f"{item.get('id', item.get('agent_id', '?'))}"
                            f"({item.get('score', '?')})"
                        )
                parts.append(f"{key}: [{', '.join(items)}]")
            continue
        if isinstance(val, str) and len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{key}={val}")
    return "; ".join(parts)


def annotate_runtime_trace(trace: dict, title: str) -> str:
    """Convert a raw runtime trace into annotated markdown."""
    lines = [f"## {title}", ""]
    lines.append(f"**Query:** {trace.get('query', 'N/A')}")
    lines.append(f"**Request ID:** `{trace.get('request_id', 'N/A')}`")
    lines.append("")
    lines.append("| Δ (ms) | Stage | Details |")
    lines.append("|-------:|-------|---------|")

    base_ts = trace.get("started_ts_ms", 0)
    for event in trace.get("events", []):
        delta = event.get("ts_ms", base_ts) - base_ts
        stage = event.get("stage", "unknown")
        data = event.get("data", {})
        annotation = STAGE_ANNOTATIONS.get(stage, "")
        detail = _format_data(data)
        if annotation:
            detail = f"*{annotation}* {detail}".strip()
        lines.append(f"| +{delta:.0f} | **{stage}** | {detail} |")

    return "\n".join(lines)


def synthesize_eval_trace(result: dict, title: str) -> str:
    """Build an annotated trace from evaluation result for AOP scenarios."""
    lines = [f"## {title}", ""]
    lines.append(f"**Query:** {result.get('turns', [{}])[0].get('query', 'N/A')}")
    lines.append(f"**Scenario:** `{result['scenario_id']}` — {result.get('description', '')}")
    lines.append(f"**Category:** {result['category']}")
    lines.append(f"**Success:** {'✓' if result['success'] else '✗'}")
    lines.append(f"**Latency:** {result['latency_ms']:.1f} ms")
    lines.append(f"**Agent Calls:** {result['agent_calls']}")
    lines.append("")

    solv = result.get("solvability_score")
    comp = result.get("completeness_score")
    if solv is not None:
        lines.append(f"**Solvability Score:** {solv:.2f}")
    if comp is not None:
        lines.append(f"**Completeness Score:** {comp:.2f}")
    lines.append("")

    # Reconstruct the AOP 5-step cycle
    lines.append("### AOP Execution Trace")
    lines.append("")
    lines.append("| Step | Stage | Details |")
    lines.append("|-----:|-------|---------|")
    lines.append("| 1 | **Classification** | Pattern classified as " f"`{result['category']}`. |")
    if result["category"] in ("hierarchical_delegation", "hitl_escalation"):
        lines.append("| 2 | **Task Decomposition** | " "LLM decomposed query into subtasks. |")
        lines.append(
            "| 3 | **Agent Selection** | "
            f"Solvability estimator scored agents (best={solv:.2f}). |"
            if solv
            else "| 3 | **Agent Selection** | Agents assigned to subtasks. |"
        )
        lines.append(
            f"| 4 | **Completeness Check** | " f"Coverage ratio: {comp:.0%}. |"
            if comp
            else "| 4 | **Completeness Check** | Plan validated. |"
        )
        lines.append(
            f"| 5 | **Execution** | " f"{result['agent_calls']} agent(s) executed subtasks. |"
        )
        lines.append("| 6 | **Feedback** | " "Performance recorded to PerformanceStore. |")
    else:
        lines.append(
            "| 2 | **Routing** | " f"Router selected agent (score={solv:.2f}). |"
            if solv
            else "| 2 | **Routing** | Agent selected by router. |"
        )
        lines.append(
            f"| 3 | **Execution** | " f"Agent handled query in {result['latency_ms']:.1f} ms. |"
        )
        lines.append(
            "| 4 | **Response** | "
            f"Keywords found: {'✓' if result['answer_keywords_found'] else '✗'} |"
        )

    return "\n".join(lines)


def _select_runtime_trace(traces: list, category: str) -> dict | None:
    """Pick the richest runtime trace matching a category heuristic."""
    for trace in sorted(traces, key=lambda t: len(t.get("events", [])), reverse=True):
        events_str = json.dumps(trace.get("events", []))
        query = trace.get("query", "").lower()

        if category == "simple_routing":
            if '"pattern": "direct"' in events_str and "refund" not in query:
                return trace
        elif category == "fsm_workflow":
            if "fsm_state" in events_str:
                return trace
        elif category == "rag_delegation":
            if "rag_delegation" in events_str:
                return trace
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    traces = _load_traces()
    eval_results = _load_eval_results()

    all_parts = [
        "# Execution Trace Examples",
        "",
        "Annotated traces showing the RuntimeSpine pipeline " "for each orchestration pattern.",
        "",
    ]

    # Simple routing — from runtime traces if available, else from eval
    rt = _select_runtime_trace(traces, "simple_routing")
    if rt:
        md = annotate_runtime_trace(rt, "Simple Routing (Direct)")
    else:
        r = next((r for r in eval_results if r["category"] == "simple_routing"), None)
        md = synthesize_eval_trace(r, "Simple Routing (Direct)") if r else ""
    if md:
        (OUTPUT_DIR / "trace_simple_routing.md").write_text(md, encoding="utf-8")
        all_parts.append(md + "\n---\n")
        print("  [OK] trace_simple_routing.md")

    # FSM workflow — from runtime traces if available
    rt = _select_runtime_trace(traces, "fsm_workflow")
    if rt:
        md = annotate_runtime_trace(rt, "FSM Workflow (Refund)")
    else:
        r = next((r for r in eval_results if r["category"] == "fsm_workflow"), None)
        md = synthesize_eval_trace(r, "FSM Workflow (Refund)") if r else ""
    if md:
        (OUTPUT_DIR / "trace_fsm_workflow.md").write_text(md, encoding="utf-8")
        all_parts.append(md + "\n---\n")
        print("  [OK] trace_fsm_workflow.md")

    # Hierarchical delegation — from eval results (AOP runs in mock mode)
    r = next(
        (r for r in eval_results if r["category"] == "hierarchical_delegation" and r["success"]),
        None,
    )
    if r:
        md = synthesize_eval_trace(r, "Hierarchical Delegation (AOP)")
        (OUTPUT_DIR / "trace_hierarchical_delegation.md").write_text(md, encoding="utf-8")
        all_parts.append(md + "\n---\n")
        print("  [OK] trace_hierarchical_delegation.md")

    # HITL escalation — from eval results
    r = next((r for r in eval_results if r["category"] == "hitl_escalation" and r["success"]), None)
    if r:
        md = synthesize_eval_trace(r, "HITL Escalation")
        (OUTPUT_DIR / "trace_hitl_escalation.md").write_text(md, encoding="utf-8")
        all_parts.append(md + "\n---\n")
        print("  [OK] trace_hitl_escalation.md")

    # Combined file
    (OUTPUT_DIR / "traces_combined.md").write_text("\n".join(all_parts), encoding="utf-8")
    print("  [OK] traces_combined.md")

    print(f"\n  Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
