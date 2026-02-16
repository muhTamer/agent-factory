# app/runtime/spine.py
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Tuple

from app.runtime.audit_writer import JsonlAuditWriter
from app.runtime.guardrails import Guardrails, NoOpGuardrails
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Router
from app.runtime.trace import Trace
from app.runtime.voice import VoiceAgent

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
    ):
        self.registry = registry
        self.router = router
        self.guardrails = guardrails or NoOpGuardrails()
        self.audit_writer = audit_writer or JsonlAuditWriter()
        self.voice = VoiceAgent()

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

        # Safe fallback: if user didn’t configure anything, include refunds policy if present
        # (keeps it generic enough for your POC; you can remove once env/config is set)
        if not ctx["policies"]:
            default_refunds = "data/refunds_policy.yaml"
            if os.path.exists(default_refunds):
                ctx["policies"].append(default_refunds)

    # -------------------------
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
            return False, {"error": "Blocked by guardrails (post).", "reason": gr.reason}

        if gr.mutated_response is not None:
            response = gr.mutated_response

        return True, response

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

                # Prefer agent-provided score ONLY if it's present and valid; otherwise use router score
                try:
                    score = (
                        float(res["score"])
                        if isinstance(res, dict) and "score" in res
                        else float(cand.score)
                    )
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
            # 1️⃣ ROUTE FIRST (with sticky workflow routing)
            pinned = ctx.get("pinned_agent_id")
            pinned_type = ctx.get("pinned_agent_type")
            pinned_terminal = ctx.get("pinned_terminal")

            if pinned and pinned_type == "workflow_runner" and pinned_terminal is False:
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
            else:
                plan = self._route(q)

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
                trace.add("guard_pre_block", intent=ctx.get("intent"), reason=pre.get("reason", ""))
                pre["request_id"] = rid
                pre.setdefault("text", f"🚫 Blocked by policy: {pre.get('reason','')}")
                pre.setdefault("response", {"text": pre["text"]})
                return pre

            q, ctx = pre
            trace.add("guard_pre_ok", intent=ctx.get("intent"))

            # 4️⃣ EXECUTE
            results = self._execute_candidates(plan, q, ctx)
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

                if is_workflow:
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

            try:
                self.audit_writer.write(trace)
            except Exception as e:
                print(f"[AUDIT] failed to write trace: {e}")
