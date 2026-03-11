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
            extra = {k: v for k, v in meta.items() if k not in {"id", "type", "ready"}}
            # Surface knowledge source types for intent-aware routing
            agent = self.registry.get(aid)
            if agent:
                try:
                    agent_meta = agent.metadata() or {}
                    ks = agent_meta.get("knowledge_sources", [])
                    if ks:
                        import os

                        extra["knowledge_source_files"] = [os.path.basename(s) for s in ks]
                except Exception:
                    pass
            catalog.append(
                {
                    "id": aid,
                    "type": meta.get("type", ""),
                    "ready": meta.get("ready", False),
                    "extra": extra,
                }
            )
        return catalog

    def _llm_route(self, query: str, catalog: List[Dict[str, Any]]) -> RoutePlan:
        system = (
            "You are an intent-aware router for a customer-service multi-agent system.\n"
            "You receive a user query and a catalog of agents.\n\n"
            "STEP 1 — Classify the user's intent:\n"
            "  • INFORMATIONAL: the user wants to learn, understand, or ask about\n"
            "    something — they are NOT requesting an action to be performed.\n"
            "  • ACTIONABLE: the user explicitly requests an action to be carried out,\n"
            "    often providing identifiers like a transaction ID, order number, or amount.\n"
            "  • MIXED: the query contains BOTH informational AND actionable intents.\n\n"
            "STEP 2 — Score each agent (0.0-1.0) based on TWO factors:\n"
            "  a) INTENT FIT: how well the agent's role matches the classified intent.\n"
            "     For INFORMATIONAL queries, agents with 'customer_facing: true' or\n"
            "     customer-facing knowledge sources are a strong fit. Domain-specialist\n"
            "     agents are a poor fit — asking ABOUT a topic is not the same as\n"
            "     requesting an action in that domain.\n"
            "     For ACTIONABLE queries, agents with action-oriented capabilities\n"
            "     (initiate, process, assess, execute) are a strong fit.\n"
            "  b) DOMAIN FIT: how well the agent's domain and capabilities match\n"
            "     the topic of the query.\n"
            "  Score reflects the PRODUCT of both factors — an agent must match on\n"
            "  both intent AND domain to score high.\n"
            "  For MIXED queries, use strategy 'fanout' and score both types.\n\n"
            "STEP 3 — Return STRICT JSON:\n"
            '  {"primary": "<best agent id>",\n'
            '   "candidates": [{"id": "…", "score": 0.0-1.0, "reason": "…"}, …],\n'
            '   "strategy": "single" | "fanout"}\n'
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
