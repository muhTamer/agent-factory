# app/runtime/router.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

from app.runtime.registry import AgentRegistry
from app.llm_client import chat_json  # <-- use your existing helper
import json


@dataclass
class RouteCandidate:
    id: str
    score: float
    reason: str


@dataclass
class RoutePlan:
    primary: str
    candidates: List[RouteCandidate]
    strategy: str  # "single" or "fanout"


class LLMRouter:
    """
    LLM-based router:
      - Looks at query + agent metadata
      - Uses chat_json() to select best agent(s)
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def _build_agent_catalog(self) -> List[Dict[str, Any]]:
        catalog = []
        for aid, meta in self.registry.all_meta().items():
            catalog.append(
                {
                    "id": aid,
                    "type": meta.get("type", ""),
                    "ready": meta.get("ready", False),
                    "extra": {k: v for k, v in meta.items() if k not in {"id", "type", "ready"}},
                }
            )
        return catalog

    def _llm_route(self, query: str, catalog: List[Dict[str, Any]]) -> RoutePlan:
        system = (
            "You are a router for a customer-service multi-agent system.\n"
            "You receive a user query and a catalog of agents.\n"
            "Each agent has: id, type, and hints about capabilities.\n"
            "Decide which agent(s) should handle the query.\n"
            "Return STRICT JSON with keys: primary, candidates, strategy.\n"
            "candidates is a list of {id, score, reason}.\n"
            "strategy is 'single' or 'fanout'."
        )

        user = {
            "query": query,
            "agents": catalog,
        }

        print("\n[ROUTER] Query:", query)
        print("[ROUTER] Catalog:", catalog)
        raw = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            model="gpt-5-mini",  # uses your deployment env var if set
            temperature=1.0,
        )

        primary = raw.get("primary")
        candidates: List[RouteCandidate] = []
        for c in raw.get("candidates", []):
            try:
                cid = str(c["id"])
                score = float(c.get("score", 0.0))
                reason = str(c.get("reason", ""))
                candidates.append(RouteCandidate(id=cid, score=score, reason=reason))
            except Exception:
                continue

        if not candidates and primary:
            candidates = [RouteCandidate(id=primary, score=1.0, reason="fallback")]

        if not primary and candidates:
            primary = candidates[0].id

        strategy = raw.get("strategy") or "single"

        return RoutePlan(
            primary=primary or "",
            candidates=candidates,
            strategy=strategy,
        )

    def route(self, query: str) -> RoutePlan:
        catalog = self._build_agent_catalog()

        # no agents or just one → trivial
        if len(catalog) <= 1:
            primary = catalog[0]["id"] if catalog else ""
            return RoutePlan(
                primary=primary,
                candidates=(
                    [RouteCandidate(id=primary, score=1.0, reason="only agent")] if primary else []
                ),
                strategy="single",
            )

        try:
            return self._llm_route(query, catalog)
        except Exception as e:
            print(f"[ROUTER] LLM routing failed, fallback: {e}")
            primary = catalog[0]["id"] if catalog else ""
            return RoutePlan(
                primary=primary,
                candidates=(
                    [RouteCandidate(id=primary, score=1.0, reason="fallback-first")]
                    if primary
                    else []
                ),
                strategy="single",
            )
