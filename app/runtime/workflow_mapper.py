# app/runtime/workflow_mapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.llm_client import chat_json

DEFAULT_MODEL = "gpt-5-mini"


@dataclass
class MapResult:
    event: Optional[str]
    slots: Dict[str, Any]
    confidence: float
    rationale: str


def _compact_slot_schema(slot_defs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only what the LLM needs."""
    out = {}
    for k, v in (slot_defs or {}).items():
        if isinstance(v, dict):
            out[k] = {
                "type": v.get("type", "string"),
                "required": bool(v.get("required", False)),
                "description": v.get("description", ""),
            }
        else:
            out[k] = {"type": "string", "required": False, "description": ""}
    return out


def map_query_to_event_and_slots(
    *,
    query: str,
    current_state: str,
    allowed_events: List[str],
    slot_defs: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    current_slots: Optional[Dict[str, Any]] = None,
) -> MapResult:
    """
    Uses LLM to map a user query into:
      - next workflow event (must be one of allowed_events, or null)
      - slot updates (only keys from slot_defs)
    """
    allowed_events = [e for e in allowed_events if isinstance(e, str) and e.strip()]
    slot_schema = _compact_slot_schema(slot_defs)

    system = (
        "You are a workflow event router for a customer-service FSM.\n"
        "Given a user message, the current workflow state, allowed events, and slot schema,\n"
        "you must choose the best next event (or null if not enough info), and extract/update slots.\n\n"
        "STRICT RULES:\n"
        "- event MUST be one of allowed_events OR null.\n"
        "- Only output slot keys that exist in the provided slot schema.\n"
        "- Do NOT hallucinate IDs. If the user did not provide a value, omit it.\n"
        "- Confidence is 0..1.\n"
        "- MISSING vs INVALID: if the user has simply not yet provided required fields, return null "
        "(the engine will ask again). Only use an error/escalation/invalid event if the user "
        "explicitly provided data that is wrong, malformed, or contradictory — NOT because a field "
        "is absent.\n\n"
        "Return JSON with keys: event, slots, confidence, rationale."
    )

    user = {
        "query": query,
        "current_state": current_state,
        "allowed_events": allowed_events,
        "slot_schema": slot_schema,
        "current_slots": current_slots or {},
    }

    raw = chat_json(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": str(user)},
        ],
        model=model,
        temperature=1.0,  # keep default supported value for gpt-5-mini
    )

    event = raw.get("event", None)
    if event is not None and isinstance(event, str):
        event = event.strip() or None

    # hard guard: event must be allowed
    if event is not None and event not in allowed_events:
        event = None

    slots = raw.get("slots") if isinstance(raw.get("slots"), dict) else {}

    # hard guard: only allowed slot keys
    slots = {k: v for k, v in slots.items() if k in slot_schema}

    conf = raw.get("confidence", 0.0)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return MapResult(event=event, slots=slots, confidence=conf, rationale=rationale)
