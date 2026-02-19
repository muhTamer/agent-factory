# app/runtime/voice.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field, ValidationError
from app.llm_client import chat_json


class VoiceOut(BaseModel):
    messages: List[str] = Field(min_length=1, max_length=5)
    quick_replies: List[str] = Field(default_factory=list, max_length=8)


class VoiceAgent:
    """
    Generates user-facing chat messages from structured workflow/router output.
    This agent is the only component that produces customer-visible text.
    """

    def __init__(self) -> None:
        pass

    def render(
        self,
        user_query: str,
        thread_id: str,
        vertical: Optional[str],
        structured: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        structured: output from workflow runner / router / orchestrator (NO user-facing text required)
        Returns: {"messages":[...], "quick_replies":[...]}
        """

        system = (
            "You are the customer-service chat voice for a multi-agent system.\n"
            "You MUST produce JSON only matching this schema:\n"
            '{ "messages": [string...], "quick_replies": [string...] }\n'
            "Rules:\n"
            "- Keep it WhatsApp-like: short, friendly, professional.\n"
            "- Ask AT MOST one question in total.\n"
            "- Do NOT mention internal words like workflow/state/slots/tools.\n"
            "- Do NOT hallucinate policy facts; only use facts present in the provided structured data.\n"
            "- If structured.status is 'awaiting_info' or 'missing_info' OR structured.action indicates clarification, you must ask for the missing info.\n"
            "- If structured contains missing_slots (list of strings), ask the user for those values in ONE natural question.\n"
            "- If structured.action is 'clarify' and structured.question is present, rephrase the clarification question in a friendly customer-facing tone.\n"
            "- If structured.action is 'delegate', acknowledge the customer and explain you are connecting them to the right specialist.\n"
            "- Provide 2-5 quick replies when it helps.\n"
        )

        payload = {
            "thread_id": thread_id,
            "vertical": vertical,
            "user_query": user_query,
            "structured": structured,
        }

        # ---- IMPORTANT ----
        # Replace the next block with your real LLM JSON call.
        # You likely already have a "json mode" helper used in router/workflow mapping.
        raw = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            model="gpt-5-mini",
        )

        # Some clients return a JSON string; normalize to dict
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {"messages": [raw], "quick_replies": []}

        try:
            out = VoiceOut.model_validate(raw)
        except ValidationError:
            # Last-resort: keep it safe + minimal (still not hardcoding flow text, just a generic fallback)
            out = VoiceOut(
                messages=["I can help — could you share a bit more detail?"], quick_replies=[]
            )

        return out.model_dump()
