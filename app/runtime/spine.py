# app/runtime/spine.py
from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from app.runtime.audit_writer import JsonlAuditWriter
from app.runtime.guardrails import Guardrails, NoOpGuardrails
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Router
from app.runtime.trace import Trace
from app.runtime.voice import VoiceAgent

if TYPE_CHECKING:
    from app.orchestration.aop_coordinator import AOPCoordinator
    from app.runtime.memory import ConversationMemory

# Simple in-memory per-thread context store (POC).
# Replace with Redis/Postgres later.
THREAD_CTX: Dict[str, Dict[str, Any]] = {}


class RuntimeSpine:
    """
    Invariant orchestration backbone.

    Correct Pipeline (B3.5 intent-aware):
        route
        -> infer_intent
        -> guard_pre (intent-aware)
        -> execute
        -> select
        -> respond
        -> guard_post
        -> return

    Guardrails operate on semantic intent, not raw text.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        router: Router,
        guardrails: Guardrails | None = None,
        audit_writer: JsonlAuditWriter | None = None,
        aop_coordinator: Optional[AOPCoordinator] = None,
        memory: Optional[ConversationMemory] = None,
        governance_enabled: bool = True,
    ):
        self.registry = registry
        self.router = router
        self.guardrails = guardrails or NoOpGuardrails()
        self.audit_writer = audit_writer or JsonlAuditWriter()
        self.voice = VoiceAgent()
        self.aop_coordinator = aop_coordinator
        self._governance_enabled = governance_enabled
        # Conversation memory — the "M" in PMPA (Wang et al. 2024)
        if memory is not None:
            self.memory = memory
        else:
            try:
                from app.runtime.memory import ConversationMemory as _CM

                self.memory: Optional[ConversationMemory] = _CM()
            except Exception:
                self.memory = None

    # -------------------------
    # Context defaults (non-generated, survives regen)
    # -------------------------
    def _ensure_workflow_resources(self, ctx: Dict[str, Any]) -> None:
        """
        Ensure workflow context contains policies/tools/docs. This avoids relying on generated agent config.json.

        Sources:
        - Environment variables (preferred): AF_POLICIES, AF_TOOLS, AF_DOCS (comma-separated)
        - Fallback defaults for POC: data/refunds_policy.yaml (if exists)
        """
        if not isinstance(ctx, dict):
            return

        # Normalize existing keys
        ctx.setdefault("docs", [])
        ctx.setdefault("policies", [])
        ctx.setdefault("tools", [])

        # Load from env if provided
        env_policies = os.getenv("AF_POLICIES", "").strip()
        env_tools = os.getenv("AF_TOOLS", "").strip()
        env_docs = os.getenv("AF_DOCS", "").strip()

        if env_policies:
            for p in [x.strip() for x in env_policies.split(",") if x.strip()]:
                if p not in ctx["policies"]:
                    ctx["policies"].append(p)

        if env_tools:
            for t in [x.strip() for x in env_tools.split(",") if x.strip()]:
                if t not in ctx["tools"]:
                    ctx["tools"].append(t)

        if env_docs:
            for d in [x.strip() for x in env_docs.split(",") if x.strip()]:
                if d not in ctx["docs"]:
                    ctx["docs"].append(d)

    # -------------------------
    # -------------------------
    # AOP slot propagation
    # -------------------------
    @staticmethod
    def _accumulate_aop_slots(aop_resp: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        """Extract slots from AOP subtask results and carry them forward.

        When AOP handles a multi-intent query, action subtasks executed by
        workflow agents may produce slot data (order_id, amount, etc.).
        This data must be accumulated in the thread context so that
        subsequent turns (e.g. user says "initiate the refund") don't
        re-ask for information already provided.
        """
        subtask_results = aop_resp.get("subtask_results")
        if not isinstance(subtask_results, list):
            return
        accumulated = ctx.setdefault("_accumulated_slots", {})
        for sr in subtask_results:
            result = sr.get("result")
            if not isinstance(result, dict):
                continue
            # Workflow agents return slots in their response
            slots = result.get("slots")
            if isinstance(slots, dict):
                for k, v in slots.items():
                    if v is not None:
                        accumulated[k] = v

    # -------------------------
    # Sequential AOP task helpers
    # -------------------------
    @staticmethod
    def _match_aop_task_selection(query: str, pending_aop: Dict[str, Any]) -> Optional[int]:
        """
        Check if the user is selecting a pending AOP task.
        Returns the subtask index (into the plan's subtask list) or None.
        """
        subtasks = pending_aop.get("subtasks", [])
        # Build list of pending (unexecuted) indices
        pending_indices = [i for i, s in enumerate(subtasks) if s.get("result") is None]
        if not pending_indices:
            return None

        q = query.strip()
        ql = q.lower()

        # Strategy 1: Exact quick-reply match (numbered labels)
        for menu_pos, orig_idx in enumerate(pending_indices):
            desc = subtasks[orig_idx]["description"]
            # Strip INFORMATIONAL:/ACTION: prefix for display matching
            display = desc
            for prefix in ("INFORMATIONAL: ", "ACTION: "):
                if display.startswith(prefix):
                    display = display[len(prefix) :]
                    break
            label = f"{menu_pos + 1}. {display[:60]}"
            if q == label or q == desc or ql == label.lower() or ql == desc.lower():
                return orig_idx

        # Strategy 2: Numeric / ordinal selection ("1", "2", "first", "second")
        _ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        try:
            num = int(q)
            if 1 <= num <= len(pending_indices):
                return pending_indices[num - 1]
        except ValueError:
            for word, idx in _ORDINALS.items():
                if word in ql and 1 <= idx <= len(pending_indices):
                    return pending_indices[idx - 1]

        # Strategy 3: "yes" / "continue" / "next" → select the first pending task
        _CONTINUE_PHRASES = {
            "yes",
            "yes please",
            "sure",
            "ok",
            "okay",
            "go ahead",
            "continue",
            "next",
            "proceed",
            "yes continue",
            "yeah",
            "yep",
            "y",
        }
        if ql in _CONTINUE_PHRASES:
            return pending_indices[0]

        # Strategy 4: Check if query contains a significant substring of a pending subtask
        for menu_pos, orig_idx in enumerate(pending_indices):
            desc = subtasks[orig_idx]["description"].lower()
            # Remove prefix for matching
            for prefix in ("informational: ", "action: "):
                if desc.startswith(prefix):
                    desc = desc[len(prefix) :]
                    break
            # If user's query overlaps significantly with subtask description
            q_words = set(ql.split())
            desc_words = set(desc.split())
            common = q_words & desc_words
            # Require at least 2 meaningful words in common (skip very short words)
            meaningful = {w for w in common if len(w) > 2}
            if len(meaningful) >= 2:
                return orig_idx

        return None

    @staticmethod
    def _is_decline(query: str) -> bool:
        """Check if the user is declining remaining tasks."""
        _DECLINE_PHRASES = {
            "no",
            "no thanks",
            "no thank you",
            "nah",
            "skip",
            "that's all",
            "thats all",
            "i'm good",
            "im good",
            "nothing else",
            "never mind",
            "nevermind",
            "done",
            "that's it",
            "thats it",
            "all good",
            "nope",
            "no more",
            "nothing more",
            "not now",
        }
        return query.strip().lower() in _DECLINE_PHRASES

    @staticmethod
    def _build_task_menu_response(plan: Any, rid: str) -> Dict[str, Any]:
        """Build a response presenting the AOP task menu to the user."""
        pending = plan.pending_subtasks()
        task_list = []
        for i, st in enumerate(pending):
            task_list.append(
                {
                    "index": i,
                    "subtask": st.description,
                    "agent_id": st.assigned_agent_id,
                    "solvability_score": st.solvability_score,
                }
            )

        return {
            "text": "",  # Filled by voice rendering
            "orchestration_pattern": "aop_task_menu",
            "task_menu": task_list,
            "plan_query": plan.query,
            "completeness": {
                "complete": plan.completeness.complete if plan.completeness else True,
                "missing": plan.completeness.missing if plan.completeness else [],
                "coverage_ratio": (plan.completeness.coverage_ratio if plan.completeness else 1.0),
            },
            "solvability": {
                "assignments": plan.solvability.assignments if plan.solvability else {},
                "assignment_scores": (
                    {k: round(v, 4) for k, v in plan.solvability.assignment_scores.items()}
                    if plan.solvability
                    else {}
                ),
            },
            "request_id": rid,
        }

    # Guardrails stages
    # -------------------------
    def _guard_pre(self, query: str, context: dict) -> Tuple[bool, Any]:
        gr = self.guardrails.pre(query, context)
        if not gr.allowed:
            return False, {"error": "Blocked by guardrails (pre).", "reason": gr.reason}

        if gr.mutated_query is not None:
            query = gr.mutated_query
        if gr.mutated_context is not None:
            context = gr.mutated_context

        return True, (query, context)

    def _guard_post(self, response: dict, context: dict) -> Tuple[bool, Any]:
        gr = self.guardrails.post(response, context)
        if not gr.allowed:
            return False, {
                "error": "Blocked by guardrails (post).",
                "reason": gr.reason,
            }

        if gr.mutated_response is not None:
            response = gr.mutated_response

        return True, response

    # -------------------------
    # Orchestration pattern classification
    # -------------------------
    def _classify_orchestration_pattern(self, query: str) -> str:
        """
        Determine whether a query requires AOP hierarchical delegation
        or can be handled by existing direct routing.

        Uses LLM to detect multi-intent queries.
        Returns: "direct" | "hierarchical_delegation"
        """
        from app.llm_client import chat_json

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query classifier for a customer-service multi-agent system.\n"
                    "Determine if the query contains MULTIPLE DISTINCT intents that require "
                    "different agents, or a SINGLE intent.\n\n"
                    "Examples of MULTIPLE intents:\n"
                    '  "I want a refund AND please update my email" -> hierarchical_delegation\n'
                    '  "Cancel my order and tell me your return policy" -> hierarchical_delegation\n\n'
                    "Examples of SINGLE intent:\n"
                    '  "I want a refund for order #123" -> direct\n'
                    '  "What is your refund policy?" -> direct\n'
                    '  "I received a damaged product" -> direct\n\n'
                    "When in doubt, choose direct.\n\n"
                    'Return STRICT JSON: {"pattern": "direct"} or {"pattern": "hierarchical_delegation"}'
                ),
            },
            {"role": "user", "content": query},
        ]

        try:
            raw = chat_json(messages=messages, temperature=1.0)
            pattern = raw.get("pattern", "direct")
            if pattern in ("direct", "hierarchical_delegation"):
                return pattern
            return "direct"
        except Exception:
            return "direct"

    # -------------------------
    # Orchestration stages
    # -------------------------
    def _route(self, query: str):
        plan = self.router.route(query)
        print(f"[ROUTER] plan={plan}")
        return plan

    def _execute_candidates(self, plan, query: str, context: dict):
        results = []

        for cand in plan.candidates:
            agent = self.registry.get(cand.id)
            if not agent:
                continue

            try:
                res = agent.handle(
                    {
                        "query": query,
                        "text": query,
                        "context": context,
                    }
                )

                # Score resolution:
                # 1. Agent returned a positive score → use it (agent found a match).
                # 2. Agent returned score=0.0 (no match) → fall back to 80% of the
                #    router score so the router's intent is preserved and a high-
                #    confidence primary isn't overridden by a workflow with any stub
                #    positive score.
                # 3. Agent returned no score key → use router score directly.
                try:
                    if isinstance(res, dict) and "score" in res:
                        agent_score = float(res["score"])
                        score = agent_score if agent_score > 0 else float(cand.score) * 0.8
                    else:
                        score = float(cand.score)
                except Exception:
                    score = float(cand.score)

                results.append({"agent_id": cand.id, "score": score, "response": res})
            except Exception as e:
                print(f"[ERR] agent {cand.id} failed: {e}")

            if plan.strategy == "single":
                break

        return results

    def _select_best(self, results: list):
        if not results:
            return None
        return max(results, key=lambda x: x["score"])

    def _respond(self, selected: dict, plan, request_id: str):
        resp = selected["response"]
        resp["agent_id"] = selected["agent_id"]
        resp["score"] = selected["score"]
        resp["request_id"] = request_id
        resp["router_plan"] = {
            "primary": plan.primary,
            "strategy": plan.strategy,
            "candidates": [
                {"id": c.id, "score": c.score, "reason": c.reason} for c in plan.candidates
            ],
        }
        return resp

    def _find_tool_operator(self, tool_name: str) -> Optional[str]:
        candidates = [
            f"{tool_name}_operator",
            f"tool_{tool_name}",
            f"tool_{tool_name}_operator",
            f"{tool_name}_tool_operator",
        ]
        for aid in candidates:
            if self.registry.get(aid):
                return aid
        return None

    # -------------------------
    # RQ2: Governance enrichment
    # -------------------------
    def _enrich_governance(
        self,
        trace: Trace,
        response: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> None:
        """Attach IEEE compliance, explainability, and UMF envelope to trace.

        Called in the finally block of handle_chat so every request
        (successful or not) gets governance metadata in the audit trail.
        """
        from app.governance.explainability import ExplainabilityEngine
        from app.governance.ieee_compliance import IEEEComplianceChecker
        from app.governance.message_envelope import wrap_response

        if not isinstance(response, dict) or not response:
            return

        # 1. Generate multi-level explanations
        engine = ExplainabilityEngine()
        explanations = engine.generate_all_levels(trace, response)
        expl_dicts = {k: v.to_dict() for k, v in explanations.items()}

        # 2. Wrap response in UMF envelope
        envelope = wrap_response(response, trace, ctx)
        envelope_dict = envelope.to_dict()

        # 3. Run IEEE compliance checks
        checker = IEEEComplianceChecker()
        report = checker.check_all(
            message=envelope_dict,
            trace=trace,
            response=response,
            explanations=expl_dicts,
            envelope=envelope_dict,
        )

        # 4. Store everything in trace.governance
        trace.governance = {
            "envelope": envelope_dict,
            "explanations": expl_dicts,
            "compliance": report.to_dict(),
        }

    # -------------------------
    # Public entrypoint
    # -------------------------
    def handle_chat(
        self,
        query: str,
        request_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"error": "Query text required."}

        # Persist context per thread so we can "pin" an active workflow.
        incoming_ctx: Dict[str, Any] = context or {}
        thread_id = str(incoming_ctx.get("thread_id") or "default")

        ctx: Dict[str, Any] = THREAD_CTX.get(thread_id, {})
        ctx.update(incoming_ctx)
        ctx["thread_id"] = thread_id

        # ✅ Ensure policies/tools/docs are present (survives regen)
        self._ensure_workflow_resources(ctx)

        rid = request_id or str(uuid.uuid4())
        print(f"[REQ] {rid}: {q}")

        trace = Trace.start(query=q, request_id=rid, context=ctx)
        trace.add("request_received")

        try:
            post = {}  # Initialized for memory recording in finally block
            _effective_q = q  # May be expanded for AOP follow-ups
            ctx["original_query"] = q  # For post-guardrail checks

            # 0️⃣ PENDING AOP PLAN — check if user is selecting a task
            pending_aop = ctx.get("_pending_aop")
            if pending_aop and self.aop_coordinator is not None:
                from app.orchestration.aop_coordinator import AOPPlan

                # If an agent is pinned (non-terminal), the user is following up
                # on an in-progress subtask (e.g. answering a clarification question).
                # Skip task selection — let sticky routing handle the follow-up.
                _pinned_for_aop = ctx.get("pinned_agent_id")
                _pinned_terminal = ctx.get("pinned_terminal")
                if _pinned_for_aop and _pinned_terminal is False:
                    pass  # Fall through to sticky routing (step 1️⃣)

                elif (selected_idx := self._match_aop_task_selection(q, pending_aop)) is not None:
                    # User selected a task → execute only that one
                    aop_plan = AOPPlan.from_serializable(pending_aop)
                    trace.add("aop_task_selected", index=selected_idx)

                    aop_resp = self.aop_coordinator.execute_single_subtask(
                        aop_plan, selected_idx, ctx, trace
                    )
                    aop_resp["request_id"] = rid

                    # Propagate slots from the executed subtask
                    executed_result = aop_resp.get("executed_subtask", {}).get("result")
                    if isinstance(executed_result, dict):
                        slots = executed_result.get("slots")
                        if isinstance(slots, dict):
                            accumulated = ctx.setdefault("_accumulated_slots", {})
                            for k, v in slots.items():
                                if v is not None:
                                    accumulated[k] = v

                    # Pin agent if subtask needs follow-up turns.
                    # Some agents pin themselves via context mutation; only add
                    # fallback pinning if they didn't.
                    _exec_result = aop_resp.get("executed_subtask", {}).get("result") or {}
                    if not ctx.get("pinned_agent_id") and isinstance(_exec_result, dict):
                        _agent = aop_resp.get("executed_subtask", {}).get("agent_id")

                        # RAG clarification → pin for follow-up answer
                        if (
                            _exec_result.get("rag_clarification")
                            or _exec_result.get("rag_state") == "CLARIFY"
                            or _exec_result.get("action") == "clarify"
                        ):
                            if _agent:
                                ctx["pinned_agent_id"] = _agent
                                ctx["pinned_agent_type"] = "rag_fsm"
                                ctx["pinned_terminal"] = False
                                trace.add("rag_pinned", agent_id=_agent)

                        # Workflow non-terminal → pin for slot collection
                        elif _exec_result.get("workflow_id") and not _exec_result.get(
                            "terminal", False
                        ):
                            if _agent:
                                ctx["pinned_agent_id"] = _agent
                                ctx["pinned_agent_type"] = "workflow_runner"
                                ctx["pinned_terminal"] = False
                                trace.add(
                                    "workflow_pinned_from_aop",
                                    agent_id=_agent,
                                    state=_exec_result.get("current_state"),
                                )

                    # Update the stored plan
                    updated = aop_plan.to_serializable()
                    remaining = aop_plan.pending_subtasks()

                    if remaining:
                        ctx["_pending_aop"] = updated
                    else:
                        ctx.pop("_pending_aop", None)
                        trace.add("aop_plan_complete")

                    # Voice render
                    try:
                        voice_thread = str(ctx.get("thread_id") or "default")
                        vertical = ctx.get("domain") or ctx.get("vertical")
                        chat = self.voice.render(
                            user_query=q,
                            thread_id=voice_thread,
                            vertical=vertical,
                            structured=aop_resp,
                        )
                        if isinstance(aop_resp, dict):
                            aop_resp["chat"] = chat
                            if isinstance(chat, dict) and chat.get("messages"):
                                aop_resp["text"] = chat["messages"][0]
                    except Exception as e:
                        trace.add("voice_chat_failed", error=str(e))

                    ok, post = self._guard_post(aop_resp, ctx)
                    if not ok:
                        trace.add("guard_post_block", reason=post.get("reason", ""))
                        post["request_id"] = rid
                        return post
                    trace.add("guard_post_ok")
                    return post

                elif self._is_decline(q):
                    # User declined remaining tasks
                    ctx.pop("_pending_aop", None)
                    trace.add("aop_plan_declined")

                    decline_resp: Dict[str, Any] = {
                        "text": "",
                        "orchestration_pattern": "aop_plan_declined",
                        "request_id": rid,
                    }
                    try:
                        chat = self.voice.render(
                            user_query=q,
                            thread_id=str(ctx.get("thread_id") or "default"),
                            vertical=ctx.get("domain") or ctx.get("vertical"),
                            structured=decline_resp,
                        )
                        decline_resp["chat"] = chat
                        if isinstance(chat, dict) and chat.get("messages"):
                            decline_resp["text"] = chat["messages"][0]
                    except Exception:
                        decline_resp["text"] = "No problem! Let me know if you need anything else."
                    return decline_resp

                else:
                    # Query doesn't match a pending task or decline — keep the
                    # plan alive (user may be following up on a just-answered
                    # subtask) and fall through to normal routing.
                    trace.add("aop_plan_preserved_followup")

            # 1️⃣ ROUTE FIRST (with sticky workflow routing)
            pinned = ctx.get("pinned_agent_id")
            pinned_type = ctx.get("pinned_agent_type")
            pinned_terminal = ctx.get("pinned_terminal")

            if (
                pinned
                and pinned_type in ("workflow_runner", "rag_fsm", "faq_rag", "domain_agent")
                and pinned_terminal is False
            ):
                plan = type("Plan", (), {})()
                plan.primary = pinned
                plan.strategy = "single"
                plan.candidates = [
                    type(
                        "Cand",
                        (),
                        {
                            "id": pinned,
                            "score": 1.0,
                            "reason": "Sticky workflow: continue active workflow for this thread.",
                        },
                    )()
                ]
                print(f"[ROUTER] sticky primary={pinned}")
                trace.add("sticky_route", primary=pinned)

            # 0.5️⃣ AOP PATTERN CLASSIFICATION (before normal routing)
            elif self.aop_coordinator is not None:
                # ── AOP follow-up expansion (PMPA memory-aware) ──
                # Short queries after AOP delegation get expanded with
                # context so the classifier and router understand the follow-up.
                if self.memory:
                    try:
                        _last_turn = self.memory.get_last_turn(thread_id)
                        _aop_patterns = {
                            "hierarchical_delegation",
                            "aop_task_menu",
                            "aop_task_result",
                        }
                        if (
                            _last_turn
                            and isinstance(_last_turn.response, dict)
                            and _last_turn.response.get("orchestration_pattern") in _aop_patterns
                            and len(q.split()) <= 4
                        ):
                            _prev = (
                                _last_turn.response.get("answer")
                                or _last_turn.response.get("text")
                                or ""
                            )[:500]
                            _effective_q = (
                                f"Follow-up to previous request: '{_last_turn.query}'. "
                                f"Previous response summary: {_prev}\n"
                                f"User now says: {q}"
                            )
                            trace.add("aop_followup_expanded", original=q)
                    except Exception:
                        pass

                pattern = self._classify_orchestration_pattern(_effective_q)
                trace.add("orchestration_pattern", pattern=pattern)

                if pattern == "hierarchical_delegation":
                    # Pre-guardrails apply to AOP path too
                    ok_pre, pre_result = self._guard_pre(_effective_q, ctx)
                    if not ok_pre:
                        trace.add("guard_pre_block", reason=pre_result.get("reason", ""))
                        pre_result["request_id"] = rid
                        return pre_result
                    trace.add("guard_pre_ok", intent="hierarchical_delegation")

                    print(f"[AOP] hierarchical delegation for: {q[:80]}")

                    # Plan first (decompose + solvability + completeness, NO execution)

                    aop_plan = self.aop_coordinator.plan_only(_effective_q, ctx, trace)
                    if aop_plan is None:
                        return {
                            "error": "Failed to plan subtasks.",
                            "request_id": rid,
                            "orchestration_pattern": "hierarchical_delegation",
                        }

                    if len(aop_plan.subtasks) <= 1:
                        # Single subtask → execute immediately (unchanged behavior)
                        aop_resp = self.aop_coordinator.orchestrate(_effective_q, ctx, trace)
                        aop_resp["request_id"] = rid
                        self._accumulate_aop_slots(aop_resp, ctx)
                    else:
                        # Multiple subtasks → store plan, present task menu
                        ctx["_pending_aop"] = aop_plan.to_serializable()
                        trace.add(
                            "aop_plan_stored",
                            subtask_count=len(aop_plan.subtasks),
                        )
                        aop_resp = self._build_task_menu_response(aop_plan, rid)

                    # Voice rendering — produce customer-friendly chat text
                    try:
                        voice_thread = str(ctx.get("thread_id") or "default")
                        vertical = ctx.get("domain") or ctx.get("vertical")
                        chat = self.voice.render(
                            user_query=q,
                            thread_id=voice_thread,
                            vertical=vertical,
                            structured=aop_resp,
                        )
                        if isinstance(aop_resp, dict):
                            aop_resp["chat"] = chat
                            if isinstance(chat, dict) and chat.get("messages"):
                                aop_resp["text"] = chat["messages"][0]
                    except Exception as e:
                        trace.add("voice_chat_failed", error=str(e))

                    # Run through guardrails (post) and return
                    ok, post = self._guard_post(aop_resp, ctx)
                    if not ok:
                        trace.add("guard_post_block", reason=post.get("reason", ""))
                        post["request_id"] = rid
                        post.setdefault("text", f"Blocked by policy: {post.get('reason','')}")
                        return post
                    trace.add("guard_post_ok")
                    return post

                # pattern == "direct" -> fall through to normal routing
                plan = self._route(_effective_q)
            else:
                plan = self._route(_effective_q)

            trace.add(
                "route",
                primary=plan.primary,
                strategy=plan.strategy,
                candidates=[
                    {"id": c.id, "score": c.score, "reason": c.reason} for c in plan.candidates
                ],
            )

            if not plan.candidates:
                trace.add("route_empty")
                return {"error": "No routing candidates.", "request_id": rid}

            # 2️⃣ INFER INTENT FROM ROUTING (policy-aware)
            if not ctx.get("intent"):
                route_id = plan.primary
                mapped_intent = None

                pack = getattr(self.guardrails, "pack", None)
                if pack is not None:
                    mapped_intent = pack.route_to_intent.get(route_id)

                ctx["intent"] = mapped_intent or route_id
                trace.add("intent_inferred", route=route_id, intent=ctx["intent"])

            # 3️⃣ GUARDRAILS (PRE) — intent-aware
            ok, pre = self._guard_pre(q, ctx)
            if not ok:
                trace.add(
                    "guard_pre_block",
                    intent=ctx.get("intent"),
                    reason=pre.get("reason", ""),
                )
                pre["request_id"] = rid
                pre.setdefault("text", f"🚫 Blocked by policy: {pre.get('reason','')}")
                pre.setdefault("response", {"text": pre["text"]})
                return pre

            q, ctx = pre
            trace.add("guard_pre_ok", intent=ctx.get("intent"))

            # 4️⃣ EXECUTE
            results = self._execute_candidates(plan, _effective_q, ctx)
            trace.add(
                "execute",
                results=[{"agent_id": r["agent_id"], "score": r["score"]} for r in results],
            )

            if not results:
                trace.add("execute_empty")
                return {"error": "No agent produced a response.", "request_id": rid}

            # 5️⃣ SELECT
            selected = self._select_best(results)
            if not selected:
                trace.add("select_empty")
                return {"error": "No suitable response.", "request_id": rid}

            trace.add("select", agent_id=selected["agent_id"], score=selected["score"])

            # 5.25 RAG delegation re-routing (supports both signal formats)
            _res = selected.get("response") or {}

            # Format A: delegation_target key (RAG FSM style)
            if _res.get("delegation_target"):
                _delegation_target = _res["delegation_target"]
                trace.add(
                    "rag_delegation",
                    from_agent=selected["agent_id"],
                    to_agent=_delegation_target,
                    reason=_res.get("delegation_reason", ""),
                )
                _delegate_agent = self.registry.get(_delegation_target)
                if _delegate_agent:
                    try:
                        _delegate_result = _delegate_agent.handle(
                            {"query": q, "text": q, "context": ctx}
                        )
                        _delegate_score = (
                            float(_delegate_result.get("score", 0.5))
                            if isinstance(_delegate_result, dict)
                            else 0.5
                        )
                        selected = {
                            "agent_id": _delegation_target,
                            "score": _delegate_score,
                            "response": _delegate_result,
                        }
                        trace.add("delegation_executed", agent_id=_delegation_target)
                    except Exception as e:
                        trace.add("delegation_failed", error=str(e))
                # Unpin RAG agent after delegation
                ctx.pop("pinned_agent_id", None)
                ctx.pop("pinned_agent_type", None)
                ctx.pop("pinned_terminal", None)

            # Format B: action=="delegate" with delegate dict (RAG_main style)
            elif (
                isinstance(_res, dict)
                and _res.get("action") == "delegate"
                and isinstance(_res.get("delegate"), dict)
            ):
                delegate_info = _res["delegate"]
                suggested_type = delegate_info.get("suggested_type")
                suggested_id = delegate_info.get("suggested_id")

                delegate_agent_id = None
                if suggested_id and self.registry.get(suggested_id):
                    delegate_agent_id = suggested_id
                elif suggested_type and suggested_type != "unknown":
                    for aid, meta in self.registry.all_meta().items():
                        if meta.get("type") == suggested_type and aid != selected["agent_id"]:
                            delegate_agent_id = aid
                            break

                if delegate_agent_id:
                    delegate_agent = self.registry.get(delegate_agent_id)
                    if delegate_agent:
                        try:
                            delegate_result = delegate_agent.handle(
                                {"query": q, "text": q, "context": ctx}
                            )
                            try:
                                delegate_score = float(delegate_result.get("score", 0.5))
                            except (TypeError, ValueError):
                                delegate_score = 0.5
                            trace.add(
                                "rag_delegation",
                                from_agent=selected["agent_id"],
                                to_agent=delegate_agent_id,
                                reason=delegate_info.get("reason", ""),
                            )
                            selected = {
                                "agent_id": delegate_agent_id,
                                "score": delegate_score,
                                "response": delegate_result,
                            }
                            selected["response"]["delegated_from"] = _res.get("agent_id", "")
                        except Exception as e:
                            trace.add("rag_delegation_failed", error=str(e))
                            print(f"[ERR] RAG delegation to {delegate_agent_id} failed: {e}")

            # 5.5 FSM state snapshot (workflow_runner agents only)
            _res = selected.get("response") or {}
            if "current_state" in _res:
                _fsm_event: Dict[str, Any] = {
                    "agent_id": selected["agent_id"],
                    "workflow_id": _res.get("workflow_id"),
                    "current_state": _res.get("current_state"),
                    "terminal": bool(_res.get("terminal", False)),
                    "action": _res.get("action"),
                }
                _slots = _res.get("slots")
                if isinstance(_slots, dict):
                    _fsm_event["slots"] = {k: v for k, v in _slots.items() if v is not None}
                    # Accumulate slots across agents for cross-turn handoff.
                    # When agent A (e.g. router-intent) extracts customer_id and
                    # agent B (e.g. workflow-refund-orch) needs it later, the
                    # accumulated slots bridge the gap via THREAD_CTX.
                    accumulated = ctx.setdefault("_accumulated_slots", {})
                    for k, v in _slots.items():
                        if v is not None:
                            accumulated[k] = v
                _missing = _res.get("missing_slots")
                if _missing:
                    _fsm_event["missing_slots"] = _missing
                trace.add("fsm_state", **_fsm_event)

            # 6️⃣ RESPOND
            resp = self._respond(selected, plan, rid)

            # Pin workflow runner for this thread until terminal=True
            if (
                isinstance(resp, dict)
                and resp.get("workflow_id")
                and resp.get("agent_id") == plan.primary
            ):
                ctx["pinned_agent_id"] = plan.primary
                ctx["pinned_agent_type"] = "workflow_runner"
                ctx["pinned_terminal"] = bool(resp.get("terminal", False))
                if ctx["pinned_terminal"]:
                    ctx.pop("pinned_agent_id", None)
                    ctx.pop("pinned_agent_type", None)
                    ctx.pop("pinned_terminal", None)

            # Pin RAG agent during clarification (multi-turn FAQ)
            # Supports both signal formats: rag_clarification (FSM) and rag_state+thread_active (RAG_main)
            if isinstance(resp, dict) and (
                resp.get("rag_clarification")
                or (resp.get("rag_state") == "CLARIFY" and resp.get("thread_active") is True)
            ):
                ctx["pinned_agent_id"] = resp.get("agent_id", plan.primary)
                ctx["pinned_agent_type"] = "rag_fsm"
                ctx["pinned_terminal"] = False
                trace.add("rag_pinned", agent_id=ctx["pinned_agent_id"])
            elif (
                isinstance(resp, dict)
                and ctx.get("pinned_agent_type") in ("rag_fsm", "faq_rag")
                and (resp.get("rag_answered") or resp.get("thread_active") is not True)
            ):
                ctx.pop("pinned_agent_id", None)
                ctx.pop("pinned_agent_type", None)
                ctx.pop("pinned_terminal", None)
                trace.add("rag_unpinned")

            # Pin domain agent during multi-turn (ask_user → resume)
            if isinstance(resp, dict) and resp.get("domain_agent_clarification"):
                ctx["pinned_agent_id"] = resp.get("agent_id", plan.primary)
                ctx["pinned_agent_type"] = "domain_agent"
                ctx["pinned_terminal"] = False
                trace.add("domain_agent_pinned", agent_id=ctx["pinned_agent_id"])
            elif (
                isinstance(resp, dict)
                and ctx.get("pinned_agent_type") == "domain_agent"
                and not resp.get("needs_input")
            ):
                ctx.pop("pinned_agent_id", None)
                ctx.pop("pinned_agent_type", None)
                ctx.pop("pinned_terminal", None)
                trace.add("domain_agent_unpinned")

            # Detect transition from pinned → unpinned (covers both
            # spine-detected unpin AND agent self-unpin via context mutation).
            # If remaining AOP tasks exist, inject them into the response.
            _was_rag_pinned = (
                pinned
                and pinned_type in ("rag_fsm", "faq_rag", "domain_agent")
                and pinned_terminal is False
            )
            _now_unpinned = not ctx.get("pinned_agent_id")
            if (
                _was_rag_pinned
                and _now_unpinned
                and isinstance(resp, dict)
                and ctx.get("_pending_aop")
            ):
                _remaining_aop = ctx["_pending_aop"]
                _remaining_subtasks = [
                    {
                        "index": i,
                        "subtask": s["description"],
                        "agent_id": s.get("assigned_agent_id"),
                    }
                    for i, s in enumerate(_remaining_aop.get("subtasks", []))
                    if s.get("result") is None
                ]
                if _remaining_subtasks:
                    resp["remaining_subtasks"] = _remaining_subtasks
                    # Build quick_replies for remaining tasks directly
                    # (no full voice rendering — preserve the agent's answer text)
                    # Use sequential menu-position numbering (1, 2, …) so the
                    # labels match _match_aop_task_selection's expectations.
                    _qr = []
                    for _menu_pos, _rs in enumerate(_remaining_subtasks):
                        _desc = _rs["subtask"]
                        for _pfx in ("INFORMATIONAL: ", "ACTION: "):
                            if _desc.startswith(_pfx):
                                _desc = _desc[len(_pfx) :]
                                break
                        _qr.append(f"{_menu_pos + 1}. {_desc[:60]}")
                    _qr.append("No thanks")
                    if not resp.get("chat"):
                        resp["chat"] = {"messages": [], "quick_replies": _qr}
                    else:
                        # Preserve the agent's answer-specific quick replies,
                        # then append remaining-task options.
                        _existing_qr = resp.get("chat", {}).get("quick_replies", [])
                        resp.setdefault("chat", {})["quick_replies"] = _existing_qr + _qr
                    trace.add(
                        "aop_remaining_offered",
                        count=len(_remaining_subtasks),
                    )

            # Also inject remaining tasks after ANY normal-routing response
            # (not just pinned→unpinned RAG) when _pending_aop is still alive.
            if (
                isinstance(resp, dict)
                and not resp.get("remaining_subtasks")
                and ctx.get("_pending_aop")
            ):
                _remaining_aop2 = ctx["_pending_aop"]
                _remaining_subtasks2 = [
                    {
                        "index": i,
                        "subtask": s["description"],
                        "agent_id": s.get("assigned_agent_id"),
                    }
                    for i, s in enumerate(_remaining_aop2.get("subtasks", []))
                    if s.get("result") is None
                ]
                if _remaining_subtasks2:
                    resp["remaining_subtasks"] = _remaining_subtasks2
                    _qr2 = []
                    for _menu_pos2, _rs2 in enumerate(_remaining_subtasks2):
                        _desc2 = _rs2["subtask"]
                        for _pfx2 in ("INFORMATIONAL: ", "ACTION: "):
                            if _desc2.startswith(_pfx2):
                                _desc2 = _desc2[len(_pfx2) :]
                                break
                        _qr2.append(f"{_menu_pos2 + 1}. {_desc2[:60]}")
                    _qr2.append("No thanks")
                    if not resp.get("chat"):
                        resp["chat"] = {"messages": [], "quick_replies": _qr2}
                    else:
                        _existing_qr2 = resp.get("chat", {}).get("quick_replies", [])
                        resp.setdefault("chat", {})["quick_replies"] = _existing_qr2 + _qr2
                    trace.add(
                        "aop_remaining_offered",
                        count=len(_remaining_subtasks2),
                    )

            trace.add("response_ready", agent_id=resp.get("agent_id"), score=resp.get("score"))

            # 6.5️⃣ VOICE (chat rendering) — for workflow-style structured outputs
            try:
                candidate = resp
                if isinstance(resp, dict) and isinstance(resp.get("result"), dict):
                    candidate = resp["result"]

                is_workflow = isinstance(candidate, dict) and (
                    "workflow_id" in candidate
                    or "current_state" in candidate
                    or candidate.get("status") in ("awaiting_info", "missing_info", "in_progress")
                    or "missing_slots" in candidate
                    or "action" in candidate
                    or "terminal" in candidate
                )

                is_rag_special = isinstance(candidate, dict) and (
                    candidate.get("rag_clarification")
                    or candidate.get("delegation_target")
                    or candidate.get("rag_state") in ("CLARIFY", "DELEGATE")
                    or candidate.get("action") in ("clarify", "delegate")
                )

                if is_workflow or is_rag_special:
                    thread_id = str(
                        (ctx or {}).get("thread_id") or resp.get("thread_id") or "default"
                    )
                    vertical = (ctx or {}).get("domain") or (ctx or {}).get("vertical")

                    chat = self.voice.render(
                        user_query=q,
                        thread_id=thread_id,
                        vertical=vertical,
                        structured=candidate if candidate is not resp else resp,
                    )

                    if isinstance(resp, dict):
                        resp["chat"] = chat
                        if isinstance(chat, dict) and chat.get("messages"):
                            resp["text"] = chat["messages"][0]

            except Exception as e:
                trace.add("voice_chat_failed", error=str(e))
                if isinstance(resp, dict):
                    resp["voice_error"] = str(e)

            # 7️⃣ GUARDRAILS (POST)
            ok, post = self._guard_post(resp, ctx)
            if not ok:
                trace.add("guard_post_block", reason=post.get("reason", ""))
                post["request_id"] = rid
                post.setdefault("text", f"🚫 Blocked by policy: {post.get('reason','')}")
                post.setdefault("response", {"text": post["text"]})
                return post

            trace.add("guard_post_ok")
            return post

        finally:
            try:
                THREAD_CTX[ctx.get("thread_id", "default")] = ctx
            except Exception:
                pass

            # Record turn in conversation memory
            if self.memory:
                try:
                    # Try post first — set in both AOP and direct paths.
                    # resp is only set in the direct path (NameError in AOP path).
                    _resp = (
                        post
                        if (isinstance(post, dict) and post)
                        else (resp if isinstance(resp, dict) else {})
                    )
                except Exception:
                    _resp = {}
                try:
                    self.memory.record_turn(
                        thread_id=thread_id,
                        query=q,
                        response=_resp,
                        agent_id=_resp.get("agent_id") if isinstance(_resp, dict) else None,
                        fsm_state=(
                            _resp.get("current_state") or _resp.get("rag_state")
                            if isinstance(_resp, dict)
                            else None
                        ),
                        slots=_resp.get("slots") if isinstance(_resp, dict) else None,
                    )
                except Exception:
                    pass

            # RQ2 Governance enrichment — generate UMF envelope, explanations,
            # and IEEE compliance report, then attach to the trace before audit.
            if self._governance_enabled:
                try:
                    _final_resp = (
                        post
                        if (isinstance(post, dict) and post)
                        else (resp if isinstance(resp, dict) else {})
                    )
                except Exception:
                    _final_resp = {}
                try:
                    self._enrich_governance(trace, _final_resp, ctx)
                    # Attach governance summary to the response for the UI.
                    # Strip envelope.payload to avoid circular reference
                    # (payload IS the response dict itself).
                    if trace.governance and isinstance(_final_resp, dict):
                        import copy

                        gov = copy.copy(trace.governance)
                        if isinstance(gov.get("envelope"), dict):
                            env = dict(gov["envelope"])
                            env.pop("payload", None)
                            gov["envelope"] = env
                        _final_resp["governance"] = gov
                except Exception as e:
                    print(f"[GOVERNANCE] enrichment failed: {e}")

            try:
                self.audit_writer.write(trace)
            except Exception as e:
                print(f"[AUDIT] failed to write trace: {e}")
