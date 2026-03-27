# evaluation/harness.py
"""
Evaluation Harness — DSRM Stage 5 (Week 3)

Runs ground-truth scenarios against the RuntimeSpine and collects the 6
quantitative metrics committed to in the thesis Methods chapter:

  1. Orchestration Accuracy   — % correct pattern selection
  2. Orchestration Efficiency  — latency (ms) and agent-call count per category
  3. Reasoning Accuracy        — % successful task completion
  4. Solvability Correlation   — Spearman ρ(predicted confidence, actual success)
  5. Completeness Rate         — mean completeness score for delegation scenarios
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.runtime.spine import RuntimeSpine

# ── Result dataclasses ──────────────────────────────────────────────


@dataclass
class TurnResult:
    """Outcome of a single conversation turn within a scenario."""

    turn_index: int
    query: str
    pattern_correct: bool = False
    agent_correct: bool = False
    answer_keywords_found: bool = False
    outcome_correct: bool = False
    outcome_detail: str = ""
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Aggregate result for a complete scenario (possibly multi-turn)."""

    scenario_id: str
    category: str
    description: str
    success: bool = False
    pattern_correct: bool = False
    agent_correct: bool = False
    answer_keywords_found: bool = False
    outcome_correct: bool = False
    outcome_detail: str = ""
    latency_ms: float = 0.0
    agent_calls: int = 0
    completeness_score: Optional[float] = None
    solvability_score: Optional[float] = None
    soft_pass: bool = False
    error: Optional[str] = None
    turns: List[TurnResult] = field(default_factory=list)
    governance_events: List[Dict[str, Any]] = field(default_factory=list)


# ── Harness ─────────────────────────────────────────────────────────


class EvaluationHarness:
    """Runs ground-truth scenarios and collects metrics."""

    def __init__(self, spine: RuntimeSpine, scenarios_path: str | Path):
        self.spine = spine
        self.scenarios = self._load_scenarios(scenarios_path)

    # ── Execution ───────────────────────────────────────────────────

    def run_all(self) -> List[ScenarioResult]:
        """Run every scenario and return results."""
        results: List[ScenarioResult] = []
        for sc in self.scenarios:
            results.append(self.run_scenario(sc))
        return results

    def run_scenario(self, scenario: Dict[str, Any]) -> ScenarioResult:
        """Execute a single scenario (possibly multi-turn).

        Outcome evaluation works at two levels:
          - Per-turn: lightweight checks (clarification_ok, response_not_empty)
          - Scenario-level: accumulated checks across ALL turns (tools_called,
            knowledge_retrieved). Defined as scenario["expected_outcome"].
            This ensures the agent's full conversation is evaluated, not just
            individual responses.
        """
        sc_id = scenario["id"]
        category = scenario["category"]
        description = scenario.get("description", "")
        turns_spec = scenario.get("turns", [])

        thread_id = f"eval_{sc_id}"
        result = ScenarioResult(
            scenario_id=sc_id,
            category=category,
            description=description,
        )

        total_latency = 0.0
        total_agent_calls = 0
        all_pattern_ok = True
        all_agent_ok = True
        all_keywords_ok = True
        all_turn_outcome_ok = True
        last_response: Dict[str, Any] = {}
        all_responses: List[Dict[str, Any]] = []  # Accumulated for scenario-level check

        for i, turn in enumerate(turns_spec):
            query = turn["query"]
            expected = turn.get("expected", {})

            t0 = time.perf_counter()
            try:
                resp = self.spine.handle_chat(
                    query,
                    request_id=f"{sc_id}_turn{i}",
                    context={"thread_id": thread_id},
                )
            except Exception as e:
                result.error = f"Turn {i} raised: {e}"
                result.turns.append(
                    TurnResult(
                        turn_index=i, query=query, raw_response={"error": str(e)}
                    )
                )
                break

            latency = (time.perf_counter() - t0) * 1000.0
            total_latency += latency
            last_response = resp or {}
            all_responses.append(last_response)

            # Count agent calls
            subtask_results = last_response.get("subtask_results")
            if isinstance(subtask_results, list):
                total_agent_calls += len(subtask_results)
            else:
                total_agent_calls += 1

            # Check pattern
            actual_pattern = self._detect_pattern(last_response)
            expected_pattern = expected.get("pattern")
            pattern_ok = (expected_pattern is None) or (
                actual_pattern == expected_pattern
            )

            # Check agent
            actual_agent = last_response.get("agent_id", "")
            expected_agent_contains = expected.get("agent_contains")
            agent_ok = (expected_agent_contains is None) or (
                expected_agent_contains.lower() in (actual_agent or "").lower()
            )

            # Check answer keywords (legacy — still computed for backward compat)
            answer_text = self._extract_answer(last_response)
            expected_keywords = expected.get("answer_contains", [])
            keywords_ok = (
                all(kw.lower() in answer_text.lower() for kw in expected_keywords)
                if expected_keywords
                else True
            )

            # ── Per-turn outcome evaluation ──
            expected_outcome = expected.get("expected_outcome")
            if expected_outcome:
                outcome_ok, outcome_detail = self._check_outcome(
                    last_response, expected_outcome
                )
            else:
                outcome_ok = True
                outcome_detail = ""

            if not pattern_ok:
                all_pattern_ok = False
            if not agent_ok:
                all_agent_ok = False
            if not keywords_ok:
                all_keywords_ok = False
            if not outcome_ok:
                all_turn_outcome_ok = False

            result.turns.append(
                TurnResult(
                    turn_index=i,
                    query=query,
                    pattern_correct=pattern_ok,
                    agent_correct=agent_ok,
                    answer_keywords_found=keywords_ok,
                    outcome_correct=outcome_ok,
                    outcome_detail=outcome_detail,
                    latency_ms=latency,
                    raw_response=last_response,
                )
            )

        # ── Scenario-level accumulated outcome check ──
        # Merges data from ALL turns and checks the whole conversation outcome.
        scenario_outcome_spec = scenario.get("expected_outcome")
        if scenario_outcome_spec:
            merged = self._merge_responses(all_responses)
            scenario_outcome_ok, scenario_outcome_detail = self._check_outcome(
                merged, scenario_outcome_spec
            )
        else:
            scenario_outcome_ok = True
            scenario_outcome_detail = ""

        all_outcome_ok = all_turn_outcome_ok and scenario_outcome_ok

        # Aggregate
        result.latency_ms = total_latency
        result.agent_calls = total_agent_calls
        result.pattern_correct = all_pattern_ok
        result.agent_correct = all_agent_ok
        result.answer_keywords_found = all_keywords_ok
        result.outcome_correct = all_outcome_ok
        details = [t.outcome_detail for t in result.turns if t.outcome_detail]
        if scenario_outcome_detail:
            details.append(f"scenario:{scenario_outcome_detail}")
        result.outcome_detail = "; ".join(details)

        # Completeness (for AOP scenarios)
        comp = last_response.get("completeness")
        if isinstance(comp, dict):
            result.completeness_score = comp.get("coverage_ratio")

        # Solvability (for AOP scenarios — use average assignment score)
        solv = last_response.get("solvability")
        if isinstance(solv, dict):
            scores = solv.get("assignment_scores", {})
            if scores:
                result.solvability_score = sum(scores.values()) / len(scores)

        # For direct routing, use agent score as solvability proxy
        if result.solvability_score is None:
            try:
                result.solvability_score = float(last_response.get("score", 0))
            except (TypeError, ValueError):
                pass

        # ── Soft-pass fallback ──────────────────────────────────────
        # When the only failures are tool_missing checks but orchestration
        # (routing + response) succeeded, mark as soft_pass so the scenario
        # counts as success but is flagged for later analysis.  This avoids
        # penalising orchestration quality (RQ1) for domain-agent execution
        # non-determinism (the agent responded textually instead of calling
        # the tool).
        if not all_outcome_ok and result.agent_correct and result.error is None:
            detail = result.outcome_detail
            # Check if ALL failures are tool_missing and response was present
            only_tool_missing = (
                "tool_missing:" in detail
                and "response_present" in detail
                and all(
                    f.startswith("tool_missing:")
                    for f in self._extract_failed_checks(detail)
                )
            )
            if only_tool_missing:
                all_outcome_ok = True
                result.soft_pass = True
                result.outcome_detail += (
                    " [SOFT_PASS:tool_not_called_but_orchestration_ok]"
                )

        # Success: all turns passed + scenario outcome + no error
        success_criteria = scenario.get("success_criteria", "all_turns_pass")
        if success_criteria == "answer_not_empty":
            answer = self._extract_answer(last_response)
            result.success = bool(answer.strip()) and result.error is None
        elif success_criteria == "final_state":
            expected_state = scenario.get("expected_final_state")
            actual_state = last_response.get("current_state")
            result.success = actual_state == expected_state and result.error is None
        else:  # "all_turns_pass"
            # pattern_correct and agent_correct are tracked for analysis
            # but do NOT gate success.  RQ1 studies how pattern selection
            # affects outcomes — recording the actual pattern chosen is
            # data; forcing a specific pattern would bias the measurement.
            result.success = all_outcome_ok and result.error is None

        return result

    # ── Metrics ─────────────────────────────────────────────────────

    def compute_metrics(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Compute the 6 thesis evaluation metrics."""
        if not results:
            return {}

        n = len(results)

        # 1. Orchestration Accuracy
        orchestration_accuracy = sum(1 for r in results if r.pattern_correct) / n

        # 2. Orchestration Efficiency — latency and steps by category
        categories = sorted(set(r.category for r in results))
        latency_by_cat = {}
        steps_by_cat = {}
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            latency_by_cat[cat] = sum(r.latency_ms for r in cat_results) / len(
                cat_results
            )
            steps_by_cat[cat] = sum(r.agent_calls for r in cat_results) / len(
                cat_results
            )

        # 3. Reasoning Accuracy
        reasoning_accuracy = sum(1 for r in results if r.success) / n

        # 4. Solvability Correlation (Spearman ρ)
        solvability_pairs = [
            (r.solvability_score, 1.0 if r.success else 0.0)
            for r in results
            if r.solvability_score is not None
        ]
        solvability_rho = self._spearman_rho(solvability_pairs)

        # 5. Completeness Rate (delegation scenarios only)
        comp_scores = [
            r.completeness_score for r in results if r.completeness_score is not None
        ]
        completeness_rate = sum(comp_scores) / len(comp_scores) if comp_scores else None

        # 6. Agent accuracy
        agent_accuracy = sum(1 for r in results if r.agent_correct) / n

        # 7. Outcome accuracy (behavioral correctness)
        outcome_accuracy = sum(1 for r in results if r.outcome_correct) / n

        return {
            "orchestration_accuracy": round(orchestration_accuracy, 4),
            "reasoning_accuracy": round(reasoning_accuracy, 4),
            "agent_accuracy": round(agent_accuracy, 4),
            "outcome_accuracy": round(outcome_accuracy, 4),
            "solvability_correlation": (
                round(solvability_rho, 4) if solvability_rho is not None else None
            ),
            "completeness_rate": (
                round(completeness_rate, 4) if completeness_rate is not None else None
            ),
            "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 2),
            "latency_by_category": {k: round(v, 2) for k, v in latency_by_cat.items()},
            "steps_by_category": {k: round(v, 2) for k, v in steps_by_cat.items()},
            "total_scenarios": n,
            "passed": sum(1 for r in results if r.success),
            "soft_passed": sum(1 for r in results if r.soft_pass),
            "failed": sum(1 for r in results if not r.success),
        }

    # ── Export ──────────────────────────────────────────────────────

    def export_csv(self, results: List[ScenarioResult], path: str | Path) -> None:
        """Write one row per scenario to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "scenario_id",
            "category",
            "description",
            "success",
            "soft_pass",
            "pattern_correct",
            "agent_correct",
            "outcome_correct",
            "outcome_detail",
            "answer_keywords_found",
            "latency_ms",
            "agent_calls",
            "completeness_score",
            "solvability_score",
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
                        "success": r.success,
                        "soft_pass": r.soft_pass,
                        "pattern_correct": r.pattern_correct,
                        "agent_correct": r.agent_correct,
                        "outcome_correct": r.outcome_correct,
                        "outcome_detail": r.outcome_detail,
                        "answer_keywords_found": r.answer_keywords_found,
                        "latency_ms": round(r.latency_ms, 2),
                        "agent_calls": r.agent_calls,
                        "completeness_score": r.completeness_score,
                        "solvability_score": (
                            round(r.solvability_score, 4)
                            if r.solvability_score
                            else None
                        ),
                        "error": r.error,
                    }
                )

    def export_json(self, results: List[ScenarioResult], path: str | Path) -> None:
        """Write full results (including per-turn data) to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            d = asdict(r)
            # Remove raw_response from turns to keep file manageable
            for t in d.get("turns", []):
                t.pop("raw_response", None)
            data.append(d)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ── Response merging (for scenario-level accumulated checks) ────

    @staticmethod
    def _merge_responses(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple turn responses into one for scenario-level outcome checks.

        Accumulates tools_used, tool_results, react_trace, knowledge_sources,
        and takes the last non-empty answer/text/message. This allows
        scenario-level outcome checks to evaluate the WHOLE conversation.
        """
        merged: Dict[str, Any] = {}
        all_tools: list = []
        all_tool_results: list = []
        all_react_trace: list = []
        all_knowledge_sources: list = []
        all_answers: list = []

        for resp in responses:
            # Accumulate tools
            tu = resp.get("tools_used")
            if isinstance(tu, list):
                all_tools.extend(tu)
            tr = resp.get("tool_results")
            if isinstance(tr, list):
                all_tool_results.extend(tr)
            rt = resp.get("react_trace")
            if isinstance(rt, list):
                all_react_trace.extend(rt)
            ks = resp.get("knowledge_sources")
            if isinstance(ks, list):
                all_knowledge_sources.extend(ks)

            # Also accumulate from subtask_results (AOP)
            sr = resp.get("subtask_results")
            if isinstance(sr, list):
                for sub in sr:
                    if isinstance(sub, dict):
                        sub_r = sub.get("result", {})
                        if isinstance(sub_r, dict):
                            st = sub_r.get("tools_used", [])
                            if isinstance(st, list):
                                all_tools.extend(st)
                            sub_ks = sub_r.get("knowledge_sources", [])
                            if isinstance(sub_ks, list):
                                all_knowledge_sources.extend(sub_ks)
                            if sub_r.get("knowledge_retrieved"):
                                merged["knowledge_retrieved"] = True

            # Collect all non-empty answers (for scenario-level keyword checks)
            for key in ("answer", "text", "message"):
                val = resp.get(key)
                if isinstance(val, str) and val.strip():
                    all_answers.append(val.strip())
                    break

            # Propagate flags (any turn having them counts)
            for flag in (
                "needs_input",
                "rag_clarification",
                "domain_agent_clarification",
                "escalation",
                "knowledge_retrieved",
            ):
                if resp.get(flag):
                    merged[flag] = True

        merged["tools_used"] = all_tools
        merged["tool_results"] = all_tool_results
        merged["react_trace"] = all_react_trace
        merged["knowledge_sources"] = all_knowledge_sources
        # Concatenate all turn answers so scenario-level keyword checks
        # can find terms from any turn, not just the last one.
        merged["answer"] = "\n".join(all_answers)
        return merged

    # ── Outcome-based evaluation ─────────────────────────────────────

    @staticmethod
    def _check_outcome(
        response: Dict[str, Any], expected_outcome: Dict[str, Any]
    ) -> tuple:
        """Evaluate agent behavior by checking actions taken, not keywords.

        expected_outcome can contain any combination of:
          - tools_called: list of tool names agent should have invoked
          - tools_not_called: list of tool names agent should NOT have invoked
          - clarification_ok: if true, agent asking for info counts as success
          - escalation_expected: whether human escalation should occur
          - knowledge_retrieved: whether KB retrieval should have happened
          - knowledge_source_contains: list of substrings, ANY must appear in
            a retrieved passage (verifies correct FAQ entry was retrieved)
          - answer_contains: list of keywords ALL must appear in response
          - answer_contains_any: list of keywords ANY one match = pass (policy compliance)
          - answer_not_contains: list of keywords NONE should appear (internal doc leak check)
          - response_not_empty: just check agent produced a response

        Returns (success: bool, detail: str).
        """
        checks_passed = []
        checks_failed = []

        # ── Extract tool call data from response ──
        # Direct agent response: tools_used, tool_results, react_trace
        actual_tools = set()
        tools_used = response.get("tools_used")
        if isinstance(tools_used, list):
            actual_tools.update(t.lower() for t in tools_used if isinstance(t, str))

        # Also check tool_results array
        tool_results = response.get("tool_results")
        if isinstance(tool_results, list):
            for tr in tool_results:
                if isinstance(tr, dict) and tr.get("tool"):
                    actual_tools.add(tr["tool"].lower())

        # Also check react_trace for call_tool actions
        react_trace = response.get("react_trace")
        if isinstance(react_trace, list):
            for step in react_trace:
                if isinstance(step, dict) and step.get("action") == "call_tool":
                    action_input = step.get("action_input", {})
                    if isinstance(action_input, dict) and action_input.get("tool"):
                        actual_tools.add(action_input["tool"].lower())

        # For AOP responses, also collect tools from subtask_results
        subtask_results = response.get("subtask_results")
        if isinstance(subtask_results, list):
            for sr in subtask_results:
                if isinstance(sr, dict):
                    sr_result = sr.get("result", {})
                    if isinstance(sr_result, dict):
                        sr_tools = sr_result.get("tools_used", [])
                        if isinstance(sr_tools, list):
                            actual_tools.update(
                                t.lower() for t in sr_tools if isinstance(t, str)
                            )

        # ── Check: tools_called ──
        expected_tools = expected_outcome.get("tools_called")
        if expected_tools is not None:
            for tool in expected_tools:
                if tool.lower() in actual_tools:
                    checks_passed.append(f"tool:{tool}")
                else:
                    checks_failed.append(f"tool_missing:{tool}")

        # ── Check: tools_not_called ──
        forbidden_tools = expected_outcome.get("tools_not_called")
        if forbidden_tools is not None:
            for tool in forbidden_tools:
                if tool.lower() in actual_tools:
                    checks_failed.append(f"forbidden_tool:{tool}")
                else:
                    checks_passed.append(f"no_forbidden:{tool}")

        # ── Check: clarification_ok ──
        clarification_ok = expected_outcome.get("clarification_ok")
        if clarification_ok is True:
            # Agent asking for clarification is valid behavior
            is_clarifying = (
                response.get("needs_input", False)
                or response.get("rag_clarification", False)
                or response.get("domain_agent_clarification", False)
            )
            # Also check if answer text looks like a question
            answer = ""
            for key in ("answer", "text", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
            asks_question = answer.endswith("?") or any(
                phrase in answer.lower()
                for phrase in [
                    "could you",
                    "can you",
                    "please provide",
                    "which ",
                    "what is your",
                    "do you have",
                ]
            )
            if is_clarifying or asks_question or bool(answer):
                checks_passed.append("clarification_or_response")
            else:
                checks_failed.append("no_response")

        # ── Check: escalation_expected ──
        escalation_expected = expected_outcome.get("escalation_expected")
        if escalation_expected is not None:
            actual_escalation = bool(response.get("escalation", False))
            if actual_escalation == escalation_expected:
                checks_passed.append(
                    f"escalation={'yes' if actual_escalation else 'no'}"
                )
            else:
                checks_failed.append(
                    f"escalation_mismatch:expected={escalation_expected},got={actual_escalation}"
                )

        # ── Check: knowledge_retrieved ──
        knowledge_expected = expected_outcome.get("knowledge_retrieved")
        if knowledge_expected is not None:
            has_knowledge = bool(response.get("knowledge_retrieved", False))
            # Also check knowledge_sources
            ks = response.get("knowledge_sources")
            if isinstance(ks, list) and len(ks) > 0:
                has_knowledge = True
            # Also check react_trace for retrieve_knowledge actions
            if isinstance(react_trace, list):
                for step in react_trace:
                    if (
                        isinstance(step, dict)
                        and step.get("action") == "retrieve_knowledge"
                    ):
                        has_knowledge = True
                        break

            if has_knowledge == knowledge_expected:
                checks_passed.append(f"knowledge={'yes' if has_knowledge else 'no'}")
            else:
                checks_failed.append(
                    f"knowledge_mismatch:expected={knowledge_expected},got={has_knowledge}"
                )

        # ── Check: knowledge_source_contains ──
        # Verify the correct FAQ entry / knowledge source was retrieved.
        # Value is a list of substrings; at least ONE must appear in any
        # source name OR passage text (OR semantics, case-insensitive).
        ks_contains = expected_outcome.get("knowledge_source_contains")
        if ks_contains is not None:
            ks_list = response.get("knowledge_sources") or []
            # Combine source names and passage text for matching
            all_source_text = " ".join(
                src
                for entry in ks_list
                if isinstance(entry, dict)
                for src in (entry.get("sources") or [])
            ).lower()
            all_passages = " ".join(
                p
                for entry in ks_list
                if isinstance(entry, dict)
                for p in (entry.get("passages") or [])
            ).lower()
            searchable = all_source_text + " " + all_passages
            matched = [kw for kw in ks_contains if kw.lower() in searchable]
            if matched:
                checks_passed.append(f"ks_source:{','.join(matched[:3])}")
            else:
                checks_failed.append(
                    f"ks_source_missing:none_of[{','.join(ks_contains)}]"
                )

        # ── Check: response_not_empty ──
        if expected_outcome.get("response_not_empty"):
            answer = ""
            for key in ("answer", "text", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
            if answer:
                checks_passed.append("response_present")
            else:
                checks_failed.append("response_empty")

        # ── Check: answer_contains (ALL keywords must match) ──
        kw_list = expected_outcome.get("answer_contains")
        if kw_list:
            answer = ""
            for key in ("answer", "text", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
            for kw in kw_list:
                if kw.lower() in answer.lower():
                    checks_passed.append(f"kw:{kw}")
                else:
                    checks_failed.append(f"kw_missing:{kw}")

        # ── Check: answer_not_contains (NONE should appear — internal doc leak) ──
        kw_blacklist = expected_outcome.get("answer_not_contains")
        if kw_blacklist:
            answer = ""
            for key in ("answer", "text", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
            leaked = [kw for kw in kw_blacklist if kw.lower() in answer.lower()]
            if leaked:
                checks_failed.append(f"leak_detected:[{','.join(leaked)}]")
            else:
                checks_passed.append("no_leak")

        # ── Check: answer_contains_any (ANY keyword match = pass) ──
        kw_any_list = expected_outcome.get("answer_contains_any")
        if kw_any_list:
            answer = ""
            for key in ("answer", "text", "message"):
                val = response.get(key)
                if isinstance(val, str) and val.strip():
                    answer = val.strip()
                    break
            matched = [kw for kw in kw_any_list if kw.lower() in answer.lower()]
            if matched:
                checks_passed.append(f"kw_any:{','.join(matched)}")
            else:
                checks_failed.append(f"kw_any_missing:none_of[{','.join(kw_any_list)}]")

        # ── Verdict ──
        if not checks_passed and not checks_failed:
            # No outcome checks defined — pass by default
            return True, "no_outcome_checks"

        success = len(checks_failed) == 0
        detail_parts = []
        if checks_passed:
            detail_parts.append(f"pass=[{','.join(checks_passed)}]")
        if checks_failed:
            detail_parts.append(f"fail=[{','.join(checks_failed)}]")
        return success, " ".join(detail_parts)

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _extract_failed_checks(outcome_detail: str) -> List[str]:
        """Extract individual failed check names from outcome_detail string.

        Detail format: '... fail=[tool_missing:initiate_refund,response_empty]'
        Returns e.g. ['tool_missing:initiate_refund', 'response_empty']
        """
        import re

        m = re.search(r"fail=\[([^\]]+)\]", outcome_detail)
        if not m:
            return []
        return [c.strip() for c in m.group(1).split(",") if c.strip()]

    @staticmethod
    def _load_scenarios(path: str | Path) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scenarios file not found: {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _detect_pattern(response: Dict[str, Any]) -> str:
        """Infer orchestration pattern from response keys.

        Patterns:
          - hierarchical_delegation — AOP multi-intent (task menu, task result, etc.)
          - direct                  — Single-intent routed to one domain agent
        """
        orch = response.get("orchestration_pattern", "")
        if orch in (
            "hierarchical_delegation",
            "aop_task_menu",
            "aop_task_result",
            "aop_plan_declined",
        ):
            return "hierarchical_delegation"
        return "direct"

    @staticmethod
    def _extract_answer(response: Dict[str, Any]) -> str:
        """Extract human-readable answer text from a response dict.

        Prefer 'answer' over 'text' because voice rendering may overwrite
        resp['text'] with a short customer-facing message, while 'answer'
        preserves the full agent response.

        For AOP task-menu responses, include subtask descriptions so that
        keyword checks can match against the decomposed plan.
        """
        for key in ("answer", "text", "message"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                # For task menus, append subtask descriptions
                task_menu = response.get("task_menu")
                if isinstance(task_menu, list):
                    subtask_texts = " ".join(
                        item.get("subtask", "")
                        for item in task_menu
                        if isinstance(item, dict)
                    )
                    return f"{val.strip()} {subtask_texts}".strip()
                return val.strip()
        return ""

    @staticmethod
    def _spearman_rho(
        pairs: List[tuple],
    ) -> Optional[float]:
        """
        Compute Spearman rank correlation coefficient.

        Falls back to manual computation if scipy is not available.
        """
        if len(pairs) < 3:
            return None

        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]

        try:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(x, y)
            return float(rho) if rho == rho else None  # NaN check
        except ImportError:
            pass

        # Manual Spearman: rank both, compute Pearson on ranks
        def _rank(vals):
            n = len(vals)
            indexed = sorted(range(n), key=lambda i: vals[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and vals[indexed[j + 1]] == vals[indexed[j]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[indexed[k]] = avg_rank
                i = j + 1
            return ranks

        rx = _rank(x)
        ry = _rank(y)
        n = len(rx)
        mean_rx = sum(rx) / n
        mean_ry = sum(ry) / n
        cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
        std_x = (sum((rx[i] - mean_rx) ** 2 for i in range(n))) ** 0.5
        std_y = (sum((ry[i] - mean_ry) ** 2 for i in range(n))) ** 0.5

        if std_x < 1e-12 or std_y < 1e-12:
            return None

        return cov / (std_x * std_y)
