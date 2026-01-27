# app/runtime/spine.py
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from app.runtime.registry import AgentRegistry
from app.runtime.routing import Router


class RuntimeSpine:
    def __init__(self, registry: AgentRegistry, router: Router):
        self.registry = registry
        self.router = router

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

        # 1) ROUTE
        plan = self._route(q)
        if not plan.candidates:
            return {"error": "No routing candidates.", "request_id": rid}

        # 2) EXECUTE
        results = self._execute_candidates(plan, q, ctx)
        if not results:
            return {"error": "No agent produced a response.", "request_id": rid}

        # 3) SELECT
        selected = self._select_best(results)
        if not selected:
            return {"error": "No suitable response.", "request_id": rid}

        # 4) RESPOND
        return self._respond(selected, plan, rid)

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
                results.append(
                    {
                        "agent_id": cand.id,
                        "score": score,
                        "response": res,
                    }
                )
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
