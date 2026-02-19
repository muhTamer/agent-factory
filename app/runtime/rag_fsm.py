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

import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    grounded_citations: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_metadata: Dict[str, Any] = field(default_factory=dict)
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

    # Dense retrieval (off by default — needs embedding API key)
    enable_dense_retrieval: bool = False
    dense_weight: float = 0.6  # Weight for dense scores in hybrid fusion
    sparse_weight: float = 0.4  # Weight for sparse (TF-IDF) scores
    embedding_model: str = "text-embedding-3-small"

    # LLM synthesis (off by default — needs LLM API key)
    enable_llm_synthesis: bool = False
    synthesis_model: str = "gpt-5-mini"
    max_context_passages: int = 3  # Top-k passages sent to LLM
    max_answer_tokens: int = 300

    # Post-retrieval ambiguity detection (proactive clarification)
    enable_retrieval_clarification: bool = False  # Off by default for backward compat
    ambiguity_score_flatness_threshold: float = 0.25  # top-k/top-1 ratio above this = flat
    ambiguity_topic_diversity_threshold: float = 0.55  # avg pairwise Jaccard above this = diverse
    ambiguity_min_hits: int = 3  # Need at least this many hits to check ambiguity
    ambiguity_confidence_ceiling: float = 0.65  # Skip check if confidence is very high


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
        llm_fn: Optional[Callable] = None,  # chat_json-compatible callable
        embed_fn: Optional[Callable] = None,  # embed_fn(texts) -> List[List[float]]
        dense_vecs: Optional[List[List[float]]] = None,  # pre-computed embeddings
    ) -> None:
        self.agent_id = agent_id
        self.faqs = faqs
        self._idf = idf
        self._vecs = vecs
        self._texts = texts
        self.config = config or RAGFSMConfig()
        self.memory = memory
        self._llm_fn = llm_fn
        self._embed_fn = embed_fn
        self._dense_vecs: Optional[List[List[float]]] = dense_vecs
        self.state = RAGState.ANALYZE
        self._clarification_count = 0
        self._pending_passages: Optional[List[Dict[str, Any]]] = None
        self._pending_hits: Optional[List[Tuple[float, int]]] = None
        self._pending_query: Optional[str] = None  # original query before clarification

    def step(
        self,
        query: str,
        thread_id: str = "default",
    ) -> RAGFSMResult:
        """Execute one FSM step. Returns result with state and signals."""
        # Auto-reset from terminal states for new queries
        if self.state in (RAGState.RESPOND, RAGState.DELEGATE):
            self.state = RAGState.ANALYZE

        if self.state == RAGState.CLARIFY:
            # User is responding to a clarification — try to resolve selection
            self._clarification_count += 1

            # If we have cached retrieval context, try selection resolution
            if self._pending_hits:
                resolved_idx = self._resolve_selection(query)
                if resolved_idx is not None and resolved_idx < len(self._pending_hits):
                    # User selected a specific option → go straight to RESPOND
                    selected_hit = self._pending_hits[resolved_idx]
                    hits = [selected_hit]
                    original_query = self._pending_query or query
                    self._pending_passages = None
                    self._pending_hits = None
                    self._pending_query = None
                    self.state = RAGState.RETRIEVE
                    return self._state_respond(original_query, hits, thread_id)

                # Not a selection — combine with original query for better retrieval
                if self._pending_query:
                    query = self._pending_query + " " + query

            self._pending_passages = None
            self._pending_hits = None
            self._pending_query = None
            self.state = RAGState.ANALYZE

        if self.state == RAGState.ANALYZE:
            return self._state_analyze(query, thread_id)

        if self.state == RAGState.RETRIEVE:
            return self._state_retrieve(query, thread_id)

        # Fallback (should not reach here)
        self.state = RAGState.ANALYZE
        return self._state_analyze(query, thread_id)

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

    # ── Ambiguity detection helpers ──────────────────────────────────

    def _score_flatness(self, hits: List[Tuple[float, int]]) -> float:
        """Ratio of worst to best score in top-k. High = flat = ambiguous."""
        if not hits or len(hits) < 2:
            return 0.0
        best = hits[0][0]
        worst = hits[-1][0]
        if best <= 0:
            return 0.0
        return worst / best

    def _topic_diversity(self, query: str, hits: List[Tuple[float, int]]) -> float:
        """Average pairwise Jaccard distance of residual tokens (tokens not in query)."""
        query_tokens = set(_tok(query))
        residuals: List[set] = []
        for _, idx in hits:
            faq = self.faqs[idx] if idx < len(self.faqs) else {}
            q_text = str(faq.get("q", ""))
            passage_tokens = set(_tok(q_text)) - query_tokens
            residuals.append(passage_tokens)

        if len(residuals) < 2:
            return 0.0

        distances: List[float] = []
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                union = residuals[i] | residuals[j]
                inter = residuals[i] & residuals[j]
                if union:
                    distances.append(1.0 - len(inter) / len(union))
                else:
                    distances.append(0.0)

        return sum(distances) / len(distances) if distances else 0.0

    def _build_passages(self, hits: List[Tuple[float, int]]) -> List[Dict[str, Any]]:
        """Build passage dicts from top hits (shared by respond + clarify)."""
        passages: List[Dict[str, Any]] = []
        for s, i in hits[: self.config.max_context_passages]:
            faq = self.faqs[i] if i < len(self.faqs) else {}
            text = ""
            if faq:
                q = str(faq.get("q", ""))
                a = str(faq.get("a", faq.get("answer", "")))
                text = f"Q: {q}\nA: {a}" if q and a else a or q
            if not text and i < len(self._texts):
                text = self._texts[i]
            passages.append(
                {
                    "text": text,
                    "source": str(faq.get("source", "")) if faq else "",
                    "score": round(float(s), 3),
                    "index": i,
                }
            )
        return passages

    # ── Clarification selection resolution ─────────────────────────────

    def _resolve_selection(self, response: str) -> Optional[int]:
        """
        Try to resolve a user response as a selection from clarification options.

        Handles:
          - Letter selections: "A", "B", "C", "D" (case-insensitive)
          - Number selections: "1", "2", "3", "4"
          - Prefixed: "option B", "choice 2", "B)"
          - Keyword match: if response tokens overlap significantly with one passage

        Returns 0-based index into _pending_hits, or None if not a selection.
        """
        text = response.strip().lower()
        n = len(self._pending_hits) if self._pending_hits else 0
        if n == 0:
            return None

        # Strip common prefixes
        for prefix in ("option", "choice", "select", "pick", "number"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        # Strip trailing punctuation: "B)", "B.", "B:"
        text = text.rstrip(").:,;")

        # Single letter: A=0, B=1, C=2, D=3
        if len(text) == 1 and text.isalpha():
            idx = ord(text) - ord("a")
            if 0 <= idx < n:
                return idx

        # Single digit: 1=0, 2=1, 3=2, 4=3
        if text.isdigit():
            idx = int(text) - 1
            if 0 <= idx < n:
                return idx

        # Keyword match: find the passage whose FAQ question best matches
        if self._pending_hits and len(_tok(response)) <= 5:
            response_tokens = set(_tok(response))
            if not response_tokens:
                return None
            best_overlap = 0.0
            best_idx = None
            for idx, (_, faq_idx) in enumerate(self._pending_hits[:n]):
                faq = self.faqs[faq_idx] if faq_idx < len(self.faqs) else {}
                faq_tokens = set(_tok(str(faq.get("q", ""))))
                if not faq_tokens:
                    continue
                overlap = len(response_tokens & faq_tokens) / len(response_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            # Require at least 50% of the response tokens to match
            if best_overlap >= 0.5 and best_idx is not None:
                return best_idx

        return None

    # ── Clarification question generation ────────────────────────────

    def _heuristic_clarification(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Build a clarification question from passage topics without LLM."""
        query_tokens = set(_tok(query))
        options: List[str] = []
        for p in passages[:4]:
            text = p.get("text", "")
            # Extract the FAQ question part
            q_part = text.split("A:")[0].replace("Q:", "").strip() if "A:" in text else text[:120]
            distinguishing = [t for t in _tok(q_part) if t not in query_tokens]
            if distinguishing:
                options.append(" ".join(distinguishing[:5]))

        if options:
            opts_str = ", ".join(f'"{o}"' for o in options[:4])
            return (
                "I found information about several related topics. "
                f"Could you clarify which one you mean? For example: {opts_str}"
            )
        return (
            "Could you provide more details about your question? "
            "For example, what specific topic or product are you asking about?"
        )

    def _generate_clarification_question(self, query: str, passages: List[Dict[str, Any]]) -> str:
        """Generate a context-aware clarification question (LLM or heuristic fallback)."""
        if not self._llm_fn:
            return self._heuristic_clarification(query, passages)

        system = (
            "You are a helpful customer service assistant. The user asked a broad "
            "question that matches several different topics in our knowledge base. "
            "Generate ONE concise clarification question that helps narrow down "
            "which topic they mean. List the 2-4 most distinct options.\n"
            'Return JSON: {"question": "your clarification question here"}'
        )
        context = "\n\n".join(f"[{i + 1}] {p['text'][:200]}" for i, p in enumerate(passages))
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Passages:\n{context}\n\nOriginal query: {query}"},
        ]
        try:
            raw = self._llm_fn(messages=messages, model=self.config.synthesis_model)
            if isinstance(raw, str):
                raw = json.loads(raw)
            question = raw.get("question", "")
            if question:
                return question
        except Exception:
            pass
        return self._heuristic_clarification(query, passages)

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

    def _state_clarify_from_retrieval(
        self,
        query: str,
        thread_id: str,
        solv: Optional[SolvabilitySignals],
        passages: List[Dict[str, Any]],
        hits: List[Tuple[float, int]],
        flatness: float = 0.0,
        diversity: float = 0.0,
    ) -> RAGFSMResult:
        """CLARIFY after retrieval: ambiguous results detected."""
        if self._clarification_count >= self.config.max_clarifications:
            return self._state_respond(query, hits, thread_id, solv)

        # Cache passages and original query for selection resolution
        self._pending_passages = passages
        self._pending_hits = hits
        self._pending_query = query

        question = self._generate_clarification_question(query, passages)

        self.state = RAGState.CLARIFY
        return RAGFSMResult(
            state=RAGState.CLARIFY,
            clarification_question=question,
            solvability=solv,
            metadata={
                "clarification_round": self._clarification_count + 1,
                "clarification_type": "retrieval_informed",
                "ambiguity_flatness": round(flatness, 3),
                "ambiguity_diversity": round(diversity, 3),
                "candidate_passages": len(passages),
            },
        )

    def _state_retrieve(
        self,
        query: str,
        thread_id: str,
        solv: Optional[SolvabilitySignals] = None,
    ) -> RAGFSMResult:
        """RETRIEVE: TF-IDF search with optional dense retrieval and hybrid fusion."""
        effective_query = self._expand_query(query, thread_id)

        # Sparse (TF-IDF) scores
        qv = _vec_query(effective_query, self._idf)
        sparse_scores: List[Tuple[float, int]] = []
        for i, dv in enumerate(self._vecs):
            s = _cosine(qv, dv)
            sparse_scores.append((s, i))

        # Hybrid fusion if dense retrieval is enabled
        if self.config.enable_dense_retrieval and self._embed_fn is not None and self._texts:
            try:
                scored = self._hybrid_retrieve(effective_query, sparse_scores)
            except Exception:
                # Embedding API failed — fall back to sparse-only
                scored = sparse_scores
        else:
            scored = sparse_scores

        scored.sort(key=lambda x: x[0], reverse=True)
        top_hits = scored[: self.config.top_k]

        if not top_hits or top_hits[0][0] < self.config.relevance_gate:
            # No relevant hits — delegate
            if solv is None:
                solv = self.estimate_solvability(query)
            return self._state_delegate(query, solv, thread_id)

        # Post-retrieval ambiguity detection
        if (
            self.config.enable_retrieval_clarification
            and self._clarification_count < self.config.max_clarifications
            and len(top_hits) >= self.config.ambiguity_min_hits
            and (solv is None or solv.confidence < self.config.ambiguity_confidence_ceiling)
        ):
            flatness = self._score_flatness(top_hits)
            diversity = self._topic_diversity(query, top_hits)
            if (
                flatness > self.config.ambiguity_score_flatness_threshold
                and diversity > self.config.ambiguity_topic_diversity_threshold
            ):
                passages = self._build_passages(top_hits)
                return self._state_clarify_from_retrieval(
                    query,
                    thread_id,
                    solv,
                    passages,
                    top_hits,
                    flatness=flatness,
                    diversity=diversity,
                )

        return self._state_respond(query, top_hits, thread_id, solv)

    def _state_respond(
        self,
        query: str,
        hits: List[Tuple[float, int]],
        thread_id: str,
        solv: Optional[SolvabilitySignals] = None,
    ) -> RAGFSMResult:
        """RESPOND: format answer with citations, optionally via LLM synthesis."""
        self.state = RAGState.RESPOND

        best_score, best_i = hits[0]

        # Build passages for top hits (reusable helper)
        passages = self._build_passages(hits)

        # Basic citations (backward compatible)
        citations = []
        for p in passages:
            i = p["index"]
            if i < len(self.faqs):
                f = self.faqs[i]
                citations.append(
                    {
                        "question": str(f.get("q", "")),
                        "source": str(f.get("source", "")),
                        "score": p["score"],
                    }
                )

        # Determine retrieval mode for metadata
        retrieval_mode = "sparse"
        if (
            self.config.enable_dense_retrieval
            and self._embed_fn is not None
            and self._dense_vecs is not None
        ):
            retrieval_mode = "hybrid"

        # LLM synthesis or extractive fallback
        grounded_citations: List[Dict[str, Any]] = []
        synthesis_meta: Dict[str, Any] = {"retrieval_mode": retrieval_mode}
        _use_extractive = True

        if self.config.enable_llm_synthesis and self._llm_fn is not None and passages:
            try:
                synth = self._synthesize_answer(query, passages)
                answer = synth.get("answer", "")
                cited_indices = synth.get("cited_passages", [])
                synthesis_meta["model"] = self.config.synthesis_model
                synthesis_meta["synthesized"] = True
                _use_extractive = False

                # Map cited passage indices to grounded citations
                for ci in cited_indices:
                    idx = ci - 1  # LLM uses 1-based indices
                    if 0 <= idx < len(passages):
                        p = passages[idx]
                        grounded_citations.append(
                            {
                                "passage": textwrap.shorten(
                                    p["text"].replace("\n", " ").strip(),
                                    width=200,
                                    placeholder=" ...",
                                ),
                                "source": p["source"],
                                "score": p["score"],
                            }
                        )
            except Exception:
                # LLM API failed — fall through to extractive
                _use_extractive = True

        if _use_extractive:
            # Extractive fallback: return raw FAQ answer
            faq = self.faqs[best_i] if best_i < len(self.faqs) else {}
            answer = str(faq.get("a", faq.get("answer", "")))
            if not answer and best_i < len(self._texts):
                answer = textwrap.shorten(
                    self._texts[best_i].replace("\n", " ").strip(),
                    width=450,
                    placeholder=" ...",
                )
            synthesis_meta["synthesized"] = False

        return RAGFSMResult(
            state=RAGState.RESPOND,
            answer=answer,
            citations=citations,
            score=round(float(best_score), 3),
            solvability=solv,
            grounded_citations=grounded_citations,
            synthesis_metadata=synthesis_meta,
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

    # ── Dense retrieval helpers ─────────────────────────────────────

    def _ensure_dense_index(self) -> None:
        """Lazy-build dense embeddings for all corpus texts."""
        if self._dense_vecs is not None:
            return
        if not self._embed_fn or not self._texts:
            self._dense_vecs = []
            return
        self._dense_vecs = self._embed_fn(self._texts)

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        """Dot product of two vectors (assumed unit-normalized)."""
        return sum(x * y for x, y in zip(a, b))

    def _hybrid_retrieve(
        self,
        query: str,
        sparse_scores: List[Tuple[float, int]],
    ) -> List[Tuple[float, int]]:
        """Fuse sparse (TF-IDF) and dense (embedding) scores."""
        self._ensure_dense_index()
        if not self._dense_vecs or not self._embed_fn:
            return sparse_scores

        # Embed query
        q_vecs = self._embed_fn([query])
        if not q_vecs or not q_vecs[0]:
            return sparse_scores
        q_vec = q_vecs[0]

        # Build sparse score lookup
        sparse_lookup: Dict[int, float] = {i: s for s, i in sparse_scores}

        # Compute fused scores
        fused: List[Tuple[float, int]] = []
        for i, dv in enumerate(self._dense_vecs):
            dense_score = max(0.0, min(1.0, self._dot(q_vec, dv)))
            sparse_score = sparse_lookup.get(i, 0.0)
            combined = (
                self.config.sparse_weight * sparse_score + self.config.dense_weight * dense_score
            )
            fused.append((combined, i))

        return fused

    # ── LLM synthesis ─────────────────────────────────────────────

    def _synthesize_answer(self, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call LLM to synthesize a grounded answer from retrieved passages."""
        system = (
            "You are a helpful customer service assistant. "
            "Answer the user's question using ONLY the provided passages. "
            "Cite passages by [1], [2], etc. "
            "If the passages don't contain enough info, say so honestly. "
            'Return JSON: {"answer": "...", "cited_passages": [1, 2]}'
        )
        context = "\n\n".join(
            f"[{i + 1}] (source: {p['source']}) {p['text']}" for i, p in enumerate(passages)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            raw = self._llm_fn(messages=messages, model=self.config.synthesis_model)
            if isinstance(raw, str):
                raw = json.loads(raw)
            if not isinstance(raw, dict):
                raw = {"answer": str(raw), "cited_passages": []}
            if "answer" not in raw:
                raw["answer"] = str(raw)
            if "cited_passages" not in raw:
                raw["cited_passages"] = []
            return raw
        except Exception:
            # Fallback: return first passage text
            return {
                "answer": passages[0]["text"] if passages else "",
                "cited_passages": [1] if passages else [],
            }

    # ── Query expansion ───────────────────────────────────────────

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
        self._pending_passages = None
        self._pending_hits = None
        self._pending_query = None
