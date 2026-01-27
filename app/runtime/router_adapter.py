# app/runtime/router_adapter.py
from __future__ import annotations

from app.runtime.router import LLMRouter
from app.runtime.routing import Router, RoutePlan, Candidate


class LLMRouterAdapter(Router):
    def __init__(self, llm_router: LLMRouter):
        self.llm_router = llm_router

    def route(self, query: str) -> RoutePlan:
        plan = self.llm_router.route(query)
        return RoutePlan(
            primary=plan.primary,
            strategy=plan.strategy,
            candidates=[
                Candidate(id=c.id, score=c.score, reason=c.reason) for c in plan.candidates
            ],
        )
