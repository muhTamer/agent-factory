# app/runtime/router.py
from __future__ import annotations
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

# ── Force UTF-8 stdout/stderr on Windows (avoids charmap codec errors) ──
if sys.platform == "win32":
    for _stream in ("stdout", "stderr"):
        _s = getattr(sys, _stream, None)
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

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
            # Surface document content metadata for intent-aware routing.
            # These fields are set by spec_builder from .doc_metadata.json
            # content analysis (visibility, categories, topics).
            extra.setdefault(
                "has_customer_facing_docs", meta.get("has_customer_facing_docs", False)
            )
            extra.setdefault(
                "has_internal_policy", meta.get("has_internal_policy", False)
            )
            extra.setdefault("document_categories", meta.get("document_categories", []))
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
            "  a) INTENT FIT: how well the agent's document types match the intent.\n"
            "     For INFORMATIONAL queries, agents with 'has_customer_facing_docs: true'\n"
            "     are a strong fit — they have documents whose content can be shared\n"
            "     with customers. Check 'document_categories' for topic coverage.\n"
            "     Agents with only internal policy ('has_internal_policy: true',\n"
            "     'has_customer_facing_docs: false') are a poor fit for informational\n"
            "     queries — asking ABOUT a topic is not the same as requesting an action.\n"
            "     For ACTIONABLE queries, agents with action-oriented capabilities\n"
            "     and internal policy docs are a strong fit.\n"
            "  b) DOMAIN FIT: how well the agent's domain and capabilities match\n"
            "     the topic of the query.\n"
            "  Score reflects the PRODUCT of both factors — an agent must match on\n"
            "  both intent AND domain to score high.\n"
            "  For MIXED queries (informational + action), score all agents and pick\n"
            "  the best single agent that can handle the primary intent.\n\n"
            "STEP 3 — Return STRICT JSON:\n"
            '  {"primary": "<best agent id>",\n'
            '   "candidates": [{"id": "…", "score": 0.0-1.0, "reason": "…"}, …],\n'
            '   "strategy": "single"}\n'
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
            timeout=60,
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
                    [RouteCandidate(id=primary, score=1.0, reason="only agent")]
                    if primary
                    else []
                ),
                strategy="single",
            )

        # Retry once on timeout before falling back
        last_err = None
        for attempt in range(2):
            try:
                return self._llm_route(query, catalog)
            except Exception as e:
                last_err = e
                if attempt == 0 and (
                    "timeout" in str(e).lower() or "timed out" in str(e).lower()
                ):
                    print("[ROUTER] LLM routing timed out, retrying...")
                    continue
                break

        print(f"[ROUTER] LLM routing failed, fallback: {last_err}")
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
