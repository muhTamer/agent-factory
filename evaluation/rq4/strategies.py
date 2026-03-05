# evaluation/rq4/strategies.py
"""
RQ4 Orchestration Strategies — 4 Voice-Rendering Configurations

Each strategy modifies the voice-rendering prompt to produce a different
customer-facing style from the *same* underlying system output.  This
isolates the presentation layer so the judge evaluates HOW information
is communicated, not WHAT information the system produced.

Strategies align with HCI literature on transparency (IEEE 3152),
empathy (Han et al.), and proactive assistance in customer service AI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Strategy:
    """A voice-rendering strategy that modifies how responses are presented."""

    name: str
    slug: str  # machine-readable identifier
    description: str
    voice_system_addendum: str  # injected into voice.py system prompt


# ── 4 Strategies ──────────────────────────────────────────────────────

BASELINE = Strategy(
    name="Baseline",
    slug="baseline",
    description=(
        "Standard response style — default voice rendering with no "
        "additional transparency, empathy, or proactive elements."
    ),
    voice_system_addendum="",  # no modification to default voice prompt
)

TRANSPARENT = Strategy(
    name="Transparent",
    slug="transparent",
    description=(
        "Includes reasoning transparency — briefly explains what steps "
        "were taken, which policies were consulted, and why the system "
        "reached this conclusion."
    ),
    voice_system_addendum=(
        "\n\nADDITIONAL TRANSPARENCY INSTRUCTIONS:\n"
        "- After providing the answer, briefly explain WHAT STEPS the system "
        "took to arrive at this response (e.g. 'I checked your transaction "
        "history and verified eligibility against our refund policy').\n"
        "- If a policy was consulted, mention it naturally (e.g. 'According "
        "to our refund policy...' or 'Based on our account guidelines...').\n"
        "- If multiple agents or sources were involved, mention this "
        "naturally (e.g. 'I looked into both your account details and our "
        "FAQ knowledge base to answer your questions').\n"
        "- Keep the explanation concise — 1-2 sentences, not a full audit "
        "trail. The goal is to help the customer UNDERSTAND the reasoning, "
        "not to overwhelm them with process details.\n"
    ),
)

EMPATHETIC = Strategy(
    name="Empathetic",
    slug="empathetic",
    description=(
        "Prioritizes emotional acknowledgment — recognizes the customer's "
        "situation and feelings before providing the resolution."
    ),
    voice_system_addendum=(
        "\n\nADDITIONAL EMPATHY INSTRUCTIONS:\n"
        "- ALWAYS start by acknowledging the customer's situation or "
        "feelings before providing any answer or solution. For example: "
        "'I understand this must be frustrating' or 'I can see this is "
        "important to you'.\n"
        "- Match the emotional tone: if the customer seems worried, be "
        "reassuring; if frustrated, be understanding; if confused, be "
        "patient and encouraging.\n"
        "- Use warm, human language — avoid robotic or overly formal phrasing.\n"
        "- After the empathetic opening, provide the actual resolution "
        "clearly. Empathy without action is empty — always follow through.\n"
        "- End with a supportive closing (e.g. 'Please do not hesitate to "
        "reach out if you need anything else').\n"
    ),
)

PROACTIVE = Strategy(
    name="Proactive",
    slug="proactive",
    description=(
        "Anticipates follow-up needs — provides extra relevant information "
        "the customer might need next, without being asked."
    ),
    voice_system_addendum=(
        "\n\nADDITIONAL PROACTIVE INSTRUCTIONS:\n"
        "- After answering the customer's question, anticipate what they "
        "might need NEXT and proactively offer that information.\n"
        "- For refund requests: mention expected processing time, how they "
        "will be notified, and what to do if they do not see the refund.\n"
        "- For account inquiries: mention related services or features "
        "they might find useful.\n"
        "- For complaints: outline next steps, timelines, and escalation "
        "options if the customer is not satisfied.\n"
        "- Keep proactive additions relevant and concise — 1-2 extra "
        "sentences, not a wall of text. The goal is to reduce the need "
        "for follow-up questions.\n"
    ),
)


# ── All strategies (ordered) ──────────────────────────────────────────

ALL_STRATEGIES: List[Strategy] = [
    BASELINE,
    TRANSPARENT,
    EMPATHETIC,
    PROACTIVE,
]


def get_strategy(slug: str) -> Strategy:
    """Look up a strategy by slug. Raises KeyError if not found."""
    for s in ALL_STRATEGIES:
        if s.slug == slug:
            return s
    raise KeyError(f"Unknown strategy slug: {slug}")
