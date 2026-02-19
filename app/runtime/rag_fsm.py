# app/runtime/rag_fsm.py
"""
RAG Internal FSM — Multi-turn RAG with solvability estimation.

Implements the PMPA pattern (Wang et al. 2024) for RAG agents:
  - Profile: FAQ corpus + config thresholds
  - Memory: ConversationMemory (injected, optional)
  - Planning: solvability estimation -> state selection
  - Action: retrieve / clarify / delegate

States:
  ANALYZE  -> estimate solvability, decide next state
  CLARIFY  -> ask for more info (spine pins agent)
  RETRIEVE -> TF-IDF search + optional memory-based query expansion
  RESPOND  -> format answer with citations (terminal)
  DELEGATE -> signal spine to re-route (terminal)
"""
from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# ── TF-IDF helpers (same pattern as app/shared/rag.py) ──────────────

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s or "")]


def _vec_query(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tf: Dict[str, int] = {}
    for t in _tok(query):
        tf[t] = tf.get(t, 0) + 1
    vec: Dict[str, float] = {}
    norm = 0.0
    for t, f in tf.items():
        w = (1 + math.log(f)) * idf.get(t, 0.0)
        vec[t] = w
        norm += w * w
    norm = math.sqrt(max(1e-9, norm))
    for t in list(vec.keys()):
        vec[t] /= norm
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    s = 0.0
    for t, w in a.items():
        w2 = b.get(t)
        if w2 is not None:
            s += w * w2
    return float(max(0.0, min(1.0, s)))


# ── Enums and Dataclasses ───────────────────────────────────────────


class RAGState(str, Enum):
    """Internal FSM states for multi-turn RAG agent."""

    ANALYZE = "analyze"
    CLARIFY = "clarify"
    RETRIEVE = "retrieve"
    RESPOND = "respond"
    DELEGATE = "delegate"


@dataclass
class SolvabilitySignals:
    """Multi-signal solvability estimation for a RAG query."""

    tfidf_score: float  # Best TF-IDF cosine score from index
    coverage_ratio: float  # Fraction of query tokens in corpus vocab
    top_k_avg: float  # Average score of top-k hits
    confidence: float  # Combined solvability confidence (0.0-1.0)
    should_delegate: bool  # True if confidence below delegation threshold
    reasoning: str


@dataclass
class RAGFSMResult:
    """Result of a single RAG FSM step."""

    state: RAGState
    answer: Optional[str] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    solvability: Optional[SolvabilitySignals] = None
    clarification_question: Optional[str] = None
    delegation_target: Optional[str] = None
    delegation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGFSMConfig:
    """Tunable thresholds for the RAG FSM."""

    solvability_threshold: float = 0.25  # Below this -> delegate
    clarification_threshold: float = 0.15  # Below this + short query -> clarify
    min_query_tokens: int = 2  # Queries shorter than this -> clarify
    max_clarifications: int = 2  # Max clarification rounds before fallback
    top_k: int = 5  # Number of TF-IDF results
    relevance_gate: float = 0.12  # Minimum score to consider a hit relevant
    delegation_target: Optional[str] = None  # Default agent to delegate to


# ── RAG Finite State Machine ────────────────────────────────────────


class RAGFiniteStateMachine:
    """
    Internal FSM for multi-turn RAG agents.

    Implements PMPA (Wang et al. 2024):
      Profile  = FAQ corpus + config thresholds
      Memory   = ConversationMemory (injected)
      Planning = solvability estimation -> state selection
      Action   = retrieve / clarify / delegate
    """

    def __init__(
        self,
        agent_id: str,
        faqs: List[Dict[str, Any]],
        idf: Dict[str, float],
        vecs: List[Dict[str, float]],
        texts: List[str],
        config: Optional[RAGFSMConfig] = None,
        memory: Optional[Any] = None,  # ConversationMemory
    ) -> None:
        self.agent_id = agent_id
        self.faqs = faqs
        self._idf = idf
        self._vecs = vecs
        self._texts = texts
        self.config = config or RAGFSMConfig()
        self.memory = memory
        self.state = RAGState.ANALYZE
        self._clarification_count = 0

    def step(
        self,
        query: str,
        thread_id: str = "default",
    ) -> RAGFSMResult:
        """Execute one FSM step. Returns result with state and signals."""
        if self.state == RAGState.CLARIFY:
            # User is responding to a clarification — re-analyze
            self._clarification_count += 1
            self.state = RAGState.ANALYZE

        if self.state == RAGState.ANALYZE:
            return self._state_analyze(query, thread_id)

        if self.state == RAGState.RETRIEVE:
            return self._state_retrieve(query, thread_id)

        # Terminal states — should not normally be called again
        return RAGFSMResult(state=self.state, answer="Session ended.", score=0.0)

    def estimate_solvability(self, query: str) -> SolvabilitySignals:
        """
        Multi-signal solvability estimation (no LLM).

        Signals:
          1. tfidf_score: best cosine similarity from index
          2. coverage_ratio: query token overlap with corpus vocabulary
          3. top_k_avg: average of top-k hit scores

        Combined: 0.5 * tfidf_score + 0.3 * coverage_ratio + 0.2 * top_k_avg
        """
        tokens = _tok(query)
        if not tokens or not self._vecs:
            return SolvabilitySignals(
                tfidf_score=0.0,
                coverage_ratio=0.0,
                top_k_avg=0.0,
                confidence=0.0,
                should_delegate=True,
                reasoning="Empty query or empty index.",
            )

        # Score all documents
        qv = _vec_query(query, self._idf)
        scores = [(i, _cosine(qv, dv)) for i, dv in enumerate(self._vecs)]
        scores.sort(key=lambda x: x[1], reverse=True)

        tfidf_score = scores[0][1] if scores else 0.0

        # Vocabulary coverage
        covered = sum(1 for t in tokens if t in self._idf)
        coverage_ratio = covered / len(tokens) if tokens else 0.0

        # Top-k average
        top_k = scores[: self.config.top_k]
        top_k_avg = sum(s for _, s in top_k) / len(top_k) if top_k else 0.0

        # Combined confidence
        confidence = 0.5 * tfidf_score + 0.3 * coverage_ratio + 0.2 * top_k_avg
        confidence = max(0.0, min(1.0, confidence))

        should_delegate = confidence < self.config.solvability_threshold

        reasoning = (
            f"tfidf={tfidf_score:.3f}, "
            f"coverage={coverage_ratio:.3f}, "
            f"top_k_avg={top_k_avg:.3f}, "
            f"confidence={confidence:.3f}"
        )

        return SolvabilitySignals(
            tfidf_score=tfidf_score,
            coverage_ratio=coverage_ratio,
            top_k_avg=top_k_avg,
            confidence=confidence,
            should_delegate=should_delegate,
            reasoning=reasoning,
        )

    # ── State handlers ──────────────────────────────────────────────

    def _state_analyze(self, query: str, thread_id: str) -> RAGFSMResult:
        """ANALYZE: estimate solvability, route to next state."""
        solv = self.estimate_solvability(query)

        # Delegate if solvability is too low
        if solv.should_delegate:
            # But first check if we should clarify instead (vague short query)
            tokens = _tok(query)
            if (
                len(tokens) < self.config.min_query_tokens
                and self._clarification_count < self.config.max_clarifications
            ):
                return self._state_clarify(query, thread_id, solv)
            return self._state_delegate(query, solv, thread_id)

        # Clarify if score is marginal and query is short
        tokens = _tok(query)
        if (
            solv.confidence < self.config.solvability_threshold + 0.10
            and len(tokens) < self.config.min_query_tokens
            and self._clarification_count < self.config.max_clarifications
        ):
            return self._state_clarify(query, thread_id, solv)

        # Proceed to retrieve
        self.state = RAGState.RETRIEVE
        return self._state_retrieve(query, thread_id, solv)

    def _state_clarify(
        self,
        query: str,
        thread_id: str,
        solv: Optional[SolvabilitySignals] = None,
    ) -> RAGFSMResult:
        """CLARIFY: ask for more info when query is too vague."""
        if self._clarification_count >= self.config.max_clarifications:
            # Max rounds reached — delegate
            return self._state_delegate(
                query,
                solv or SolvabilitySignals(0.0, 0.0, 0.0, 0.0, True, "Max clarifications reached."),
                thread_id,
            )

        self.state = RAGState.CLARIFY
        question = (
            "Could you provide more details about your question? "
            "For example, what specific topic or product are you asking about?"
        )
        return RAGFSMResult(
            state=RAGState.CLARIFY,
            clarification_question=question,
            solvability=solv,
            metadata={"clarification_round": self._clarification_count + 1},
        )

    def _state_retrieve(
        self,
        query: str,
        thread_id: str,
        solv: Optional[SolvabilitySignals] = None,
    ) -> RAGFSMResult:
        """RETRIEVE: TF-IDF search with optional memory-based query expansion."""
        effective_query = self._expand_query(query, thread_id)

        # Search
        qv = _vec_query(effective_query, self._idf)
        scored: List[Tuple[float, int]] = []
        for i, dv in enumerate(self._vecs):
            s = _cosine(qv, dv)
            scored.append((s, i))
        scored.sort(key=lambda x: x[0], reverse=True)

        top_hits = scored[: self.config.top_k]

        if not top_hits or top_hits[0][0] < self.config.relevance_gate:
            # No relevant hits — delegate
            if solv is None:
                solv = self.estimate_solvability(query)
            return self._state_delegate(query, solv, thread_id)

        return self._state_respond(query, top_hits, thread_id, solv)

    def _state_respond(
        self,
        query: str,
        hits: List[Tuple[float, int]],
        thread_id: str,
        solv: Optional[SolvabilitySignals] = None,
    ) -> RAGFSMResult:
        """RESPOND: format answer with citations."""
        self.state = RAGState.RESPOND

        best_score, best_i = hits[0]
        faq = self.faqs[best_i] if best_i < len(self.faqs) else {}
        answer = str(faq.get("a", faq.get("answer", "")))
        if not answer and best_i < len(self._texts):
            answer = textwrap.shorten(
                self._texts[best_i].replace("\n", " ").strip(),
                width=450,
                placeholder=" ...",
            )

        citations = []
        for s, i in hits[:3]:
            if i < len(self.faqs):
                f = self.faqs[i]
                citations.append(
                    {
                        "question": str(f.get("q", "")),
                        "source": str(f.get("source", "")),
                        "score": round(float(s), 3),
                    }
                )

        return RAGFSMResult(
            state=RAGState.RESPOND,
            answer=answer,
            citations=citations,
            score=round(float(best_score), 3),
            solvability=solv,
        )

    def _state_delegate(
        self,
        query: str,
        solv: SolvabilitySignals,
        thread_id: str,
    ) -> RAGFSMResult:
        """DELEGATE: signal spine to re-route to another agent."""
        self.state = RAGState.DELEGATE
        return RAGFSMResult(
            state=RAGState.DELEGATE,
            solvability=solv,
            delegation_target=self.config.delegation_target,
            delegation_reason=(
                f"Solvability too low (confidence={solv.confidence:.3f}). "
                f"Delegating to a more suitable agent."
            ),
        )

    def _expand_query(self, query: str, thread_id: str) -> str:
        """Expand query using conversation memory (append prior context keywords)."""
        if not self.memory:
            return query

        try:
            ctx = self.memory.get_conversation_context(thread_id, limit=3)
            if not ctx:
                return query

            # Extract keywords from prior turns
            prior_keywords: List[str] = []
            for turn in ctx:
                prior_q = turn.get("query", "")
                prior_a = turn.get("answer", "")
                prior_keywords.extend(_tok(prior_q))
                prior_keywords.extend(_tok(prior_a)[:5])  # Limit answer tokens

            if not prior_keywords:
                return query

            # Deduplicate and append top tokens not already in query
            query_tokens = set(_tok(query))
            new_tokens = [t for t in dict.fromkeys(prior_keywords) if t not in query_tokens][:5]

            if new_tokens:
                return query + " " + " ".join(new_tokens)
        except Exception:
            pass

        return query

    def reset(self) -> None:
        """Reset FSM to ANALYZE state."""
        self.state = RAGState.ANALYZE
        self._clarification_count = 0
