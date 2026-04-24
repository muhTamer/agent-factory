# app/runtime/voice.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json

from pydantic import BaseModel, Field, ValidationError
from app.llm_client import chat_json


class VoiceOut(BaseModel):
    messages: List[str] = Field(min_length=1, max_length=5)
    quick_replies: List[str] = Field(default_factory=list, max_length=8)


# Keys that contain internal infrastructure data and must NEVER reach
# the voice-rendering LLM.  If the LLM sees policy file paths, allowed
# FSM events, or state transition history it will synthesize internal
# routing decisions as customer-facing options.
_INTERNAL_KEYS = frozenset(
    {
        "mapper",  # FSM event mapping internals (allowed_events, rationale, …)
        "history",  # state transition audit trail
        "policy_config",  # compiled policy pack paths / slot maps
    }
)

# Within the ``context`` sub-dict, only ``docs`` (knowledge-base refs)
# is potentially useful for the voice LLM.  Everything else is internal.
_INTERNAL_CONTEXT_KEYS = frozenset(
    {
        "policies",  # internal policy file paths
        "tools",  # tool registry references
        "_accumulated_slots",
    }
)


def _sanitize_for_voice(structured: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of *structured* with internal fields removed.

    This is the primary defence against the voice LLM synthesising
    internal decision trees (dispute/chargeback options, eligibility
    paths, etc.) from policy data that should never be customer-visible.
    """
    clean = {k: v for k, v in structured.items() if k not in _INTERNAL_KEYS}

    # Scrub internal entries from the context sub-dict
    ctx = clean.get("context")
    if isinstance(ctx, dict):
        clean["context"] = {
            k: v for k, v in ctx.items() if k not in _INTERNAL_CONTEXT_KEYS
        }

    return clean


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
            "- NEVER mention source filenames (e.g. 'BankFAQs.csv', 'refunds_policy.yaml'). "
            "Use natural labels instead: 'our FAQ', 'our refund policy', 'our banking guidelines'.\n"
            "- Do NOT hallucinate policy facts; only use facts present in the provided structured data.\n"
            "- NEVER promise future follow-up, callbacks, or say 'I will get back to you' / 'I have forwarded this'. "
            "Everything has already been processed synchronously — present the actual results directly.\n"
            "\n"
            "CRITICAL — Internal process vs customer-facing:\n"
            "- Policy rules (eligibility criteria, approval thresholds, KYC requirements, AML checks, "
            "execution rules) are INTERNAL AGENT INSTRUCTIONS — they tell the agent what to do, "
            "NOT what to show the customer.\n"
            "- NEVER present internal process steps as customer choices. Do NOT ask: "
            "'Would you like me to run the eligibility check?' or 'Should I request manual approval?'\n"
            "- Internal steps (eligibility evaluation, approval routing, identity verification, "
            "risk checks) happen AUTOMATICALLY. Present the RESULT to the customer, not the process.\n"
            "  Good: 'I am processing your refund request for EUR 120.' / 'Your refund has been approved.'\n"
            "  Bad: 'Would you like me to check eligibility and then decide whether to auto-approve?'\n"
            "- Only ask the customer for information THEY need to provide (order ID, amount, reason). "
            "Never ask them about internal decisions the system should make on its own.\n"
            "\n"
            "- If structured.status is 'awaiting_info' or 'missing_info' OR structured.action indicates clarification, you must ask for the missing info.\n"
            "- If structured contains missing_slots (list of strings), ask the user for those values in ONE natural question.\n"
            "- If structured.rag_clarification is true, produce a natural question asking for more details about their topic.\n"
            "- If structured.domain_agent_clarification is true or structured.needs_input is true, "
            "the domain agent is asking the user for information. Rephrase the question (from structured.answer or structured.text) "
            "in a friendly customer-facing tone. Provide 2-4 quick replies with likely answers the user might give.\n"
            "- If structured.escalation is true, do NOT say 'escalating' or 'connecting with a specialist'. "
            "Instead, apologize that you couldn't fully answer and invite the user to rephrase or ask something else. "
            "Provide quick replies like 'Let me try again', 'Ask something else', 'That's okay'.\n"
            "- If structured.delegation_target is present or structured.action is 'delegate', the delegation has already happened. "
            "Present the result from the delegated agent directly; do NOT say you are 'forwarding' or 'connecting'.\n"
            "- If structured.action is 'clarify' and structured.question is present, rephrase the clarification question in a friendly customer-facing tone.\n"
            "- If structured.grounded_citations is present, the answer was synthesized from retrieved passages; format it naturally and mention sources if relevant (e.g., 'According to our refund policy...').\n"
            "- For multi-part responses (e.g. AOP subtask_results), present EACH result directly to the customer in the same message.\n"
            "- Provide 2-5 quick replies when it helps.\n"
            "\n"
            "SEQUENTIAL TASK MENU:\n"
            "- If structured.orchestration_pattern is 'aop_task_menu', the system identified "
            "multiple tasks the user needs help with. Present them as a friendly numbered list. "
            "Say something like 'I can help you with these:' followed by a brief, clear description "
            "of each task (strip INFORMATIONAL:/ACTION: prefixes — those are internal labels). "
            "Ask which they'd like to start with. "
            "Quick replies MUST be numbered task labels matching the list "
            "(e.g. '1. Current Account documents', '2. Refund for order #4821').\n"
            "- If structured.orchestration_pattern is 'aop_task_result', present the completed "
            "task result naturally. First provide 2-3 quick replies relevant to the answer itself "
            "(e.g. follow-up questions the user might have about the topic). "
            "Then, if structured.remaining_subtasks is non-empty, mention the "
            "remaining task(s) and ask if the user wants to continue. "
            "Append the next remaining task (numbered) plus 'No thanks' AFTER the answer-specific quick replies.\n"
            "- If structured.orchestration_pattern is 'aop_plan_declined', acknowledge briefly "
            "and offer general help. Something like 'No problem! Let me know if there is "
            "anything else I can help with.'\n"
            "- REMAINING TASKS (general): If structured.remaining_subtasks is a non-empty list "
            "(regardless of orchestration_pattern), AFTER presenting the main answer/result, "
            "briefly mention the remaining task(s) and ask if the user would like to continue. "
            "Example: 'You also asked about [task]. Would you like me to help with that?' "
            "Quick replies should include 2-3 answer-relevant follow-ups FIRST, "
            "then the next remaining task (e.g. '1. Refund for order #4821') "
            "plus 'No thanks'. Strip INFORMATIONAL:/ACTION: prefixes from task labels.\n"
        )

        # Strip internal infrastructure data (policy paths, FSM events,
        # state history) so the voice LLM cannot synthesize internal
        # decision trees as customer-facing options.
        sanitized = (
            _sanitize_for_voice(structured)
            if isinstance(structured, dict)
            else structured
        )

        payload = {
            "thread_id": thread_id,
            "vertical": vertical,
            "user_query": user_query,
            "structured": sanitized,
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
            max_tokens=400,
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
                messages=["I can help — could you share a bit more detail?"],
                quick_replies=[],
            )

        return out.model_dump()
