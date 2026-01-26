# app/runtime/spine.py
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from app.runtime.registry import AgentRegistry
from app.runtime.router import LLMRouter


class RuntimeSpine:
    """
    Invariant orchestration backbone.
    Step A: move the existing routing + execution logic out of service.py.
    Later: Plan -> Guardrails -> QA -> Audit.
    """

    def __init__(self, registry: AgentRegistry, router: Optional[LLMRouter] = None):
        self.registry = registry
        self.router = router  # can be None (fallback mode)

    def handle_chat(
        self, query: str, request_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = context or {}
        q = (query or "").strip()
        if not q:
            return {"error": "Query text required."}

        rid = request_id or str(uuid.uuid4())
        print(f"[REQ] {rid}: {q}")

        # ---- Fallback: router not initialized ----
        if self.router is None:
            ids = self.registry.all_ids()
            if not ids:
                return {"error": "No agent available.", "request_id": rid}

            aid = ids[0]
            agent = self.registry.get(aid)
            res = agent.handle({"query": q, "text": q, "context": context})
            res["agent_id"] = aid
            res["request_id"] = rid
            return res

        # ---- LLM-based routing ----
        plan = self.router.route(q)
        print(f"[ROUTER] plan={plan}")

        results = []
        if plan.strategy == "single":
            cand = plan.candidates[0] if plan.candidates else None
            if not cand:
                return {"error": "Router provided no candidates.", "request_id": rid}

            agent = self.registry.get(cand.id)
            if not agent:
                return {"error": f"Agent {cand.id} not loaded.", "request_id": rid}

            res = agent.handle({"query": q, "text": q, "context": context})
            res_score = float(res.get("score", cand.score))
            results.append({"agent_id": cand.id, "score": res_score, "response": res})

        else:  # fanout
            for cand in plan.candidates:
                agent = self.registry.get(cand.id)
                if not agent:
                    continue
                try:
                    r = agent.handle({"query": q, "text": q, "context": context})
                    res_score = float(r.get("score", cand.score))
                    results.append({"agent_id": cand.id, "score": res_score, "response": r})
                except Exception as e:
                    print(f"[ERR] fanout call failed for {cand.id}: {e}")

        if not results:
            return {"error": "No agent produced a response.", "request_id": rid}

        best = max(results, key=lambda x: x["score"])
        resp = best["response"]
        resp["agent_id"] = best["agent_id"]
        resp["score"] = best["score"]
        resp["request_id"] = rid
        resp["router_plan"] = {
            "primary": plan.primary,
            "strategy": plan.strategy,
            "candidates": [
                {"id": c.id, "score": c.score, "reason": c.reason} for c in plan.candidates
            ],
        }
        return resp
