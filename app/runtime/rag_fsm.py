# app/runtime/rag_fsm.py
"""
Shared RAG FSM runtime module.

Provides multi-turn state management, solvability estimation,
LLM query expansion, and LLM re-ranking for generated RAG agents.

Academic grounding:
  - AOP solvability estimation (Li et al. 2024) with feedback loop
  - PMPA per-FSM-state integration (Wang et al. 2024)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm_client import chat_json

DEFAULT_MODEL = "gpt-5-mini"

# ── FSM States ──────────────────────────────────────────────────────
ANALYZE = "ANALYZE"
CLARIFY = "CLARIFY"
RETRIEVE = "RETRIEVE"
RESPOND = "RESPOND"
DELEGATE = "DELEGATE"


@dataclass
class RAGThreadState:
    """Per-thread FSM state for a multi-turn RAG conversation."""

    fsm_state: str = ANALYZE
    original_query: str = ""
    refined_query: Optional[str] = None
    clarification_count: int = 0
    tfidf_probe: List[Dict[str, Any]] = field(default_factory=list)
    solvability: float = 0.0
    solvability_reason: str = ""
    last_question: Optional[str] = None


# ── Solvability Estimation (AOP contribution) ──────────────────────


def _llm_analyze_query(
    query: str,
    top_hits: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    LLM-based query classification for solvability.

    Returns dict with keys:
      answerable (float 0-1), confidence (float 0-1), vague (bool),
      clarify_question (str|null), needs_workflow (bool), needs_tool (bool),
      delegate_reason (str|null)
    """
    system = (
        "You are a query classifier for a customer service FAQ system.\n"
        "Given a user query and the top FAQ matches (with scores), determine:\n"
        "1. answerable: can this be answered from these FAQs? (float 0-1)\n"
        "2. confidence: how confident are you in your classification? (float 0-1)\n"
        "3. vague: is the query too vague or ambiguous to answer directly? (bool)\n"
        "4. clarify_question: if vague, what ONE specific clarifying question would help? (str or null)\n"
        "5. needs_workflow: does this need a multi-step process like refund, complaint, account change? (bool)\n"
        "6. needs_tool: does this need a tool/action execution like identity verification? (bool)\n"
        "7. delegate_reason: if needs_workflow or needs_tool, explain why briefly (str or null)\n"
        "Return STRICT JSON with exactly these keys."
    )

    payload = {
        "query": query,
        "top_matches": [
            {"question": h.get("question", ""), "score": round(h.get("score", 0.0), 3)}
            for h in top_hits[:5]
        ],
    }

    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(payload)},
            ],
            model=model,
            temperature=1.0,
        )
        return result if isinstance(result, dict) else {}
    except Exception:
        return {"confidence": 0.5, "vague": False, "needs_workflow": False, "needs_tool": False}


def estimate_solvability(
    query: str,
    tfidf_hits: List[Dict[str, Any]],
    solv_cfg: Dict[str, Any],
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Multi-signal solvability estimation (AOP-inspired).

    Fuses TF-IDF retrieval signals with LLM query classification
    to decide: answer directly, clarify first, or delegate.

    Returns:
        {
            "score": float (0-1),
            "action": "answer" | "clarify" | "delegate",
            "reason": str,
            "delegate_to": str | None,
            "clarify_question": str | None,
            "llm_analysis": dict | None,
            "tfidf_signals": dict,
        }
    """
    # 1. TF-IDF signals (deterministic, no LLM)
    scores = [h.get("score", 0.0) for h in tfidf_hits] if tfidf_hits else [0.0]
    top_score = scores[0] if scores else 0.0
    score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else top_score
    top_k_above = sum(1 for s in scores[:5] if s > 0.15)

    tfidf_signal = (
        0.4 * top_score
        + 0.2 * min(1.0, score_gap * 5)
        + 0.1 * (top_k_above / max(1, min(5, len(scores))))
    )

    tfidf_signals = {
        "top_score": round(top_score, 4),
        "score_gap": round(score_gap, 4),
        "top_k_above_015": top_k_above,
        "tfidf_component": round(tfidf_signal, 4),
    }

    # 2. LLM query analysis
    llm_result = None
    if solv_cfg.get("enable_llm_analysis", True):
        llm_result = _llm_analyze_query(query, tfidf_hits[:3], model=model)

    # 3. Score fusion
    if llm_result:
        try:
            llm_conf = float(llm_result.get("confidence", 0.5))
        except (TypeError, ValueError):
            llm_conf = 0.5
        solvability = tfidf_signal + 0.3 * llm_conf
    else:
        solvability = tfidf_signal + 0.15  # moderate default

    solvability = max(0.0, min(1.0, solvability))

    # 4. Decision thresholds
    answer_thresh = solv_cfg.get("answer_threshold", 0.55)
    delegate_thresh = solv_cfg.get("delegate_threshold", 0.20)

    # Delegation takes priority when LLM detects workflow/tool need
    if llm_result and (llm_result.get("needs_workflow") or llm_result.get("needs_tool")):
        delegate_type = "workflow_runner" if llm_result.get("needs_workflow") else "tool_operator"
        return {
            "score": round(solvability, 4),
            "action": DELEGATE,
            "reason": llm_result.get(
                "delegate_reason", "Query requires workflow or tool execution"
            ),
            "delegate_to": delegate_type,
            "clarify_question": None,
            "llm_analysis": llm_result,
            "tfidf_signals": tfidf_signals,
        }

    if solvability >= answer_thresh:
        return {
            "score": round(solvability, 4),
            "action": "answer",
            "reason": "High solvability: confident FAQ match",
            "delegate_to": None,
            "clarify_question": None,
            "llm_analysis": llm_result,
            "tfidf_signals": tfidf_signals,
        }

    if solvability < delegate_thresh:
        return {
            "score": round(solvability, 4),
            "action": DELEGATE,
            "reason": "Very low solvability: query outside FAQ knowledge scope",
            "delegate_to": "unknown",
            "clarify_question": None,
            "llm_analysis": llm_result,
            "tfidf_signals": tfidf_signals,
        }

    # Uncertain band → clarify
    clarify_q = None
    if llm_result and llm_result.get("vague"):
        clarify_q = llm_result.get("clarify_question")
    if not clarify_q:
        clarify_q = (
            "Could you provide more details about your question so I can find the best answer?"
        )

    return {
        "score": round(solvability, 4),
        "action": "clarify",
        "reason": "Uncertain solvability: query may be vague or ambiguous",
        "delegate_to": None,
        "clarify_question": clarify_q,
        "llm_analysis": llm_result,
        "tfidf_signals": tfidf_signals,
    }


# ── LLM Query Expansion ────────────────────────────────────────────


def expand_query(
    query: str,
    hints: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Use LLM to expand/rephrase the query using context from top FAQ hits.
    Falls back to original query on error.
    """
    system = (
        "You are a search query expander for a customer service FAQ system.\n"
        "Given a user query and some FAQ context hints, produce a more specific "
        "and detailed search query that will match the right FAQ entry.\n"
        "Do NOT answer the question — only produce a better search query.\n"
        'Return STRICT JSON: {"expanded_query": "..."}'
    )

    hint_questions = [h.get("question", "") for h in hints[:3] if h.get("question")]
    payload = {"query": query, "faq_context_hints": hint_questions}

    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(payload)},
            ],
            model=model,
            temperature=1.0,
        )
        expanded = result.get("expanded_query", "") if isinstance(result, dict) else ""
        return expanded.strip() if expanded.strip() else query
    except Exception:
        return query


# ── LLM Re-ranking ─────────────────────────────────────────────────


def rerank_hits(
    query: str,
    hits: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """
    Use LLM to re-rank TF-IDF candidates when top scores are ambiguous.
    Falls back to original order on error.
    """
    if len(hits) < 2:
        return hits

    system = (
        "You are a search result re-ranker for a customer service FAQ system.\n"
        "Given a user query and candidate FAQ entries, re-rank them by relevance.\n"
        'Return STRICT JSON: {"ranked": [{"index": int, "relevance_score": float}]}\n'
        "index is the 0-based position in the input list. relevance_score is 0-1."
    )

    candidates = [
        {
            "index": i,
            "question": h.get("question", ""),
            "answer_preview": str(h.get("answer", ""))[:120],
        }
        for i, h in enumerate(hits[:5])
    ]
    payload = {"query": query, "candidates": candidates}

    try:
        result = chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": str(payload)},
            ],
            model=model,
            temperature=1.0,
        )
        ranked = result.get("ranked", []) if isinstance(result, dict) else []
        if not ranked:
            return hits

        reordered: List[Dict[str, Any]] = []
        seen: set = set()
        for r in sorted(ranked, key=lambda x: x.get("relevance_score", 0), reverse=True):
            idx = r.get("index", -1)
            if isinstance(idx, int) and 0 <= idx < len(hits) and idx not in seen:
                reordered.append(hits[idx])
                seen.add(idx)

        # Append any hits not covered by LLM ranking
        for i, h in enumerate(hits):
            if i not in seen:
                reordered.append(h)

        return reordered
    except Exception:
        return hits
