# app/runtime/spine.py
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from app.runtime.audit_writer import JsonlAuditWriter
from app.runtime.guardrails import Guardrails, NoOpGuardrails
from app.runtime.registry import AgentRegistry
from app.runtime.routing import Router
from app.runtime.trace import Trace


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
                score = float(res.get("score", cand.score))
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

        ctx: Dict[str, Any] = context or {}
        rid = request_id or str(uuid.uuid4())

        print(f"[REQ] {rid}: {q}")

        trace = Trace.start(query=q, request_id=rid, context=ctx)
        trace.add("request_received")

        try:
            # 1️⃣ ROUTE FIRST
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

                trace.add(
                    "intent_inferred",
                    route=route_id,
                    intent=ctx["intent"],
                )

            # 3️⃣ GUARDRAILS (PRE) — intent-aware
            ok, pre = self._guard_pre(q, ctx)
            if not ok:
                trace.add(
                    "guard_pre_block",
                    intent=ctx.get("intent"),
                    reason=pre.get("reason", ""),
                )

                pre["request_id"] = rid
                pre.setdefault(
                    "text",
                    f"🚫 Blocked by policy: {pre.get('reason','')}",
                )
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
                return {
                    "error": "No agent produced a response.",
                    "request_id": rid,
                }

            # 5️⃣ SELECT
            selected = self._select_best(results)
            if not selected:
                trace.add("select_empty")
                return {
                    "error": "No suitable response.",
                    "request_id": rid,
                }

            trace.add(
                "select",
                agent_id=selected["agent_id"],
                score=selected["score"],
            )

            # 6️⃣ RESPOND
            resp = self._respond(selected, plan, rid)
            trace.add(
                "response_ready",
                agent_id=resp.get("agent_id"),
                score=resp.get("score"),
            )

            # 7️⃣ GUARDRAILS (POST)
            ok, post = self._guard_post(resp, ctx)
            if not ok:
                trace.add(
                    "guard_post_block",
                    reason=post.get("reason", ""),
                )

                post["request_id"] = rid
                post.setdefault(
                    "text",
                    f"🚫 Blocked by policy: {post.get('reason','')}",
                )
                post.setdefault("response", {"text": post["text"]})
                return post

            trace.add("guard_post_ok")
            return post

        finally:
            try:
                self.audit_writer.write(trace)
            except Exception as e:
                print(f"[AUDIT] failed to write trace: {e}")
