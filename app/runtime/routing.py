# app/runtime/routing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from app.runtime.registry import AgentRegistry


@dataclass(frozen=True)
class Candidate:
    id: str
    score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class RoutePlan:
    primary: str
    strategy: str  # "single" | "fanout"
    candidates: List[Candidate]


class Router(Protocol):
    def route(self, query: str) -> RoutePlan: ...


class DefaultRouter:
    """
    Deterministic fallback router: always picks the first registered agent.
    This keeps the spine logic clean and stable.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def route(self, query: str) -> RoutePlan:
        ids = self.registry.all_ids()
        if not ids:
            # No candidates; spine will handle as an error
            return RoutePlan(primary="", strategy="single", candidates=[])

        aid = ids[0]
        return RoutePlan(
            primary=aid,
            strategy="single",
            candidates=[Candidate(id=aid, score=0.0, reason="default_router_first_agent")],
        )
