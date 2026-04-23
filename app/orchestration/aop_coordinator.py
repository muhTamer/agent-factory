# app/orchestration/aop_coordinator.py
"""
AOP Coordinator — Meta-Agent for Agent-Oriented Planning (Li et al. 2024)

Implements the 5-step control loop described in the thesis Theory chapter:
  1. Task decomposition       — LLM breaks query into atomic subtasks
  2. Agent selection           — SolvabilityEstimator scores (subtask, agent) pairs
  3. Completeness check        — CompletenessDetector audits plan coverage
  4. Execution                 — Delegate each subtask to its assigned agent
  5. Feedback loop             — Record results in PerformanceStore

Integration with RuntimeSpine:
  - Called when orchestration pattern == "hierarchical_delegation"
  - Returns Dict[str, Any] compatible with spine's _respond() format
  - Results flow through voice rendering and post-guardrails normally
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm_client import chat_json
from app.orchestration.completeness_detector import (
    CompletenessDetector,
    CompletenessResult,
)
from app.orchestration.performance_store import ExecutionRecord, PerformanceStore
from app.orchestration.solvability_estimator import (
    SolvabilityEstimator,
    SolvabilityResult,
)
from app.runtime.registry import AgentRegistry
from app.runtime.trace import Trace

# Type alias: either estimator exposes the same .estimate() interface.
_Estimator = Any  # SolvabilityEstimator | NeuralSolvabilityEstimator

# ---------------------------------------------------------------------------
# AOP label parser — single regex handles all LLM format variations:
#   "INFORMATIONAL: ...", "INFORMATIONAL — ...", "(INFORMATIONAL) ...",
#   "[INFORMATIONAL] ...", "ACTION - ...", etc.
# ---------------------------------------------------------------------------
_AOP_LABEL_RE = re.compile(
    r"^\s*[\(\[]?\s*(INFORMATIONAL|ACTION)\s*[\)\]]?\s*[:\-—–]?\s*",
    re.IGNORECASE,
)


def parse_aop_label(description: str) -> tuple[str | None, str]:
    """Parse an AOP decomposer label from a subtask description.

    Returns (label, clean_text) where label is "INFORMATIONAL" or "ACTION"
    (upper-cased) if found, else None.  clean_text has the prefix removed.
    """
    m = _AOP_LABEL_RE.match(description)
    if m:
        return m.group(1).upper(), description[m.end() :]
    return None, description


@dataclass
class Subtask:
    """A decomposed subtask from the original query."""

    description: str
    assigned_agent_id: Optional[str] = None
    solvability_score: float = 0.0
    result: Optional[Dict[str, Any]] = None
    success: bool = False
    latency_ms: int = 0


@dataclass
class AOPResult:
    """Complete result of AOP orchestration cycle."""

    query: str
    subtasks: List[Subtask] = field(default_factory=list)
    completeness: Optional[CompletenessResult] = None
    solvability: Optional[SolvabilityResult] = None
    composite_response: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: int = 0
    orchestration_pattern: str = "hierarchical_delegation"


@dataclass
class AOPPlan:
    """A prepared but not-yet-executed AOP plan, stored in thread context."""

    query: str
    subtasks: List[Subtask] = field(default_factory=list)
    completeness: Optional[CompletenessResult] = None
    solvability: Optional[SolvabilityResult] = None
    created_ts_ms: int = 0

    def pending_subtasks(self) -> List[Subtask]:
        """Return subtasks that have not been executed yet."""
        return [st for st in self.subtasks if st.result is None]

    def to_serializable(self) -> Dict[str, Any]:
        """Convert to a JSON-safe dict for storage in THREAD_CTX."""
        return {
            "query": self.query,
            "subtasks": [
                {
                    "description": st.description,
                    "assigned_agent_id": st.assigned_agent_id,
                    "solvability_score": st.solvability_score,
                    "result": st.result,
                    "success": st.success,
                    "latency_ms": st.latency_ms,
                }
                for st in self.subtasks
            ],
            "completeness": {
                "complete": self.completeness.complete if self.completeness else True,
                "missing": self.completeness.missing if self.completeness else [],
                "coverage_ratio": (
                    self.completeness.coverage_ratio if self.completeness else 1.0
                ),
                "reasoning": self.completeness.reasoning if self.completeness else "",
            },
            "solvability": {
                "assignments": self.solvability.assignments if self.solvability else {},
                "assignment_scores": (
                    {
                        k: round(v, 4)
                        for k, v in self.solvability.assignment_scores.items()
                    }
                    if self.solvability
                    else {}
                ),
            },
            "created_ts_ms": self.created_ts_ms,
        }

    @classmethod
    def from_serializable(cls, data: Dict[str, Any]) -> "AOPPlan":
        """Reconstruct from THREAD_CTX stored dict."""
        subtasks = [
            Subtask(
                description=s["description"],
                assigned_agent_id=s.get("assigned_agent_id"),
                solvability_score=s.get("solvability_score", 0.0),
                result=s.get("result"),
                success=s.get("success", False),
                latency_ms=s.get("latency_ms", 0),
            )
            for s in data.get("subtasks", [])
        ]
        comp_data = data.get("completeness", {})
        comp = CompletenessResult(
            complete=comp_data.get("complete", True),
            missing=comp_data.get("missing", []),
            redundant=[],
            coverage_ratio=comp_data.get("coverage_ratio", 1.0),
            reasoning=comp_data.get("reasoning", ""),
        )
        solv_data = data.get("solvability", {})
        solv = SolvabilityResult(
            assignments=solv_data.get("assignments", {}),
            scores=[],
            assignment_scores=solv_data.get("assignment_scores", {}),
        )
        return cls(
            query=data.get("query", ""),
            subtasks=subtasks,
            completeness=comp,
            solvability=solv,
            created_ts_ms=data.get("created_ts_ms", 0),
        )


class AOPCoordinator:
    """
    Meta-agent implementing the 5-step Agent-Oriented Planning cycle.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        performance_store: PerformanceStore,
        estimator: Optional[_Estimator] = None,
        completeness: Optional[CompletenessDetector] = None,
        model: str = "gpt-5-mini",
        max_retries: int = 1,
        memory: Optional[Any] = None,
        action_signals: Optional[re.Pattern] = None,
    ):
        self.registry = registry
        self.store = performance_store
        self.estimator = estimator or self._default_estimator(performance_store)
        self.completeness = completeness or CompletenessDetector(model=model)
        self.model = model
        self.max_retries = max_retries
        self.memory = memory  # ConversationMemory for multi-turn context
        # Regex that matches concrete transaction identifiers in a user query.
        # Override per-vertical via the action_signals constructor argument.
        self._action_signals = action_signals or re.compile(
            r"(order\s*#?\d|transaction\s*#?\d|[A-Z]{3}\s*\d|\$\d)",
            re.IGNORECASE,
        )

    def orchestrate(
        self,
        query: str,
        context: Dict[str, Any],
        trace: Optional[Trace] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full 5-step AOP cycle.

        Returns dict compatible with spine's response format.
        """
        start_ms = _now_ms()
        agent_catalog = self._aop_candidate_catalog()

        if not agent_catalog:
            return {
                "error": "No agents available for delegation.",
                "orchestration_pattern": "hierarchical_delegation",
            }

        # Retrieve conversation history for multi-turn context
        conversation_history: List[Dict[str, Any]] = []
        if self.memory:
            try:
                thread_id = context.get("thread_id", "default")
                conversation_history = self.memory.get_conversation_context(
                    thread_id, limit=5
                )
            except Exception:
                pass

        # ── Step 1: Task Decomposition ──
        subtask_strs = self._decompose(query, agent_catalog, conversation_history)
        if trace:
            trace.add("aop_decompose", subtasks=subtask_strs)

        if not subtask_strs:
            return {
                "error": "Failed to decompose query into subtasks.",
                "orchestration_pattern": "hierarchical_delegation",
            }

        # ── Step 2: Agent Selection (Solvability) ──
        solv_result = self._select_agents(subtask_strs, agent_catalog)
        if trace:
            trace.add("aop_solvability", assignments=solv_result.assignments)

        # Build Subtask objects
        subtasks = []
        for st_str in subtask_strs:
            st = Subtask(
                description=st_str,
                assigned_agent_id=solv_result.assignments.get(st_str),
                solvability_score=solv_result.assignment_scores.get(st_str, 0.0),
            )
            subtasks.append(st)

        # ── Step 3: Completeness Check ──
        # Bypass when the decomposer labeled all subtasks INFORMATIONAL — the
        # completeness LLM would otherwise invent action requirements the user
        # never requested (e.g. "missing: actionable refund initiation").
        if self._all_informational(subtask_strs):
            comp_result = CompletenessResult(
                complete=True,
                missing=[],
                redundant=[],
                coverage_ratio=1.0,
                reasoning="informational query — action coverage not required",
            )
            if trace:
                trace.add(
                    "aop_completeness",
                    complete=True,
                    missing=[],
                    info="informational_bypass",
                )
        else:
            comp_result = self._check_completeness(
                query, subtask_strs, solv_result.assignments
            )
            if trace:
                trace.add(
                    "aop_completeness",
                    complete=comp_result.complete,
                    missing=comp_result.missing,
                )

        # If incomplete and retries remain, re-decompose with hints
        if not comp_result.complete and self.max_retries > 0:
            subtask_strs = self._re_decompose(query, agent_catalog, comp_result.missing)
            if subtask_strs:
                solv_result = self._select_agents(subtask_strs, agent_catalog)
                subtasks = [
                    Subtask(
                        description=st,
                        assigned_agent_id=solv_result.assignments.get(st),
                        solvability_score=solv_result.assignment_scores.get(st, 0.0),
                    )
                    for st in subtask_strs
                ]
                comp_result = self._check_completeness(
                    query, subtask_strs, solv_result.assignments
                )
                if trace:
                    trace.add(
                        "aop_redecompose",
                        subtasks=subtask_strs,
                        complete=comp_result.complete,
                    )

        # ── Step 4: Execution ──
        subtasks = self._execute_subtasks(subtasks, context)
        if trace:
            trace.add(
                "aop_execute",
                results=[
                    {
                        "subtask": st.description,
                        "agent": st.assigned_agent_id,
                        "success": st.success,
                    }
                    for st in subtasks
                ],
            )

        # ── Step 5: Feedback Loop ──
        self._record_feedback(subtasks)

        total_ms = _now_ms() - start_ms

        # ── Assemble Response ──
        return self._assemble_composite_response(
            query, subtasks, comp_result, solv_result, total_ms
        )

    # ── Sequential multi-task helpers ────────────────────────────────

    def plan_only(
        self,
        query: str,
        context: Dict[str, Any],
        trace: Optional[Trace] = None,
    ) -> Optional[AOPPlan]:
        """
        Execute steps 1-3 of the AOP cycle (decompose, solvability, completeness)
        WITHOUT executing subtasks.  Returns an AOPPlan for deferred execution.
        """
        agent_catalog = self._aop_candidate_catalog()
        if not agent_catalog:
            return None

        conversation_history: List[Dict[str, Any]] = []
        if self.memory:
            try:
                thread_id = context.get("thread_id", "default")
                conversation_history = self.memory.get_conversation_context(
                    thread_id, limit=5
                )
            except Exception:
                pass

        # Step 1: Decompose
        subtask_strs = self._decompose(query, agent_catalog, conversation_history)
        if trace:
            trace.add("aop_decompose", subtasks=subtask_strs)
        if not subtask_strs:
            return None

        # Step 2: Solvability
        solv_result = self._select_agents(subtask_strs, agent_catalog)
        if trace:
            trace.add("aop_solvability", assignments=solv_result.assignments)

        subtasks = [
            Subtask(
                description=st_str,
                assigned_agent_id=solv_result.assignments.get(st_str),
                solvability_score=solv_result.assignment_scores.get(st_str, 0.0),
            )
            for st_str in subtask_strs
        ]

        # Step 3: Completeness
        if self._all_informational(subtask_strs):
            comp_result = CompletenessResult(
                complete=True,
                missing=[],
                redundant=[],
                coverage_ratio=1.0,
                reasoning="informational query — action coverage not required",
            )
            if trace:
                trace.add(
                    "aop_completeness",
                    complete=True,
                    missing=[],
                    info="informational_bypass",
                )
        else:
            comp_result = self._check_completeness(
                query, subtask_strs, solv_result.assignments
            )
            if trace:
                trace.add(
                    "aop_completeness",
                    complete=comp_result.complete,
                    missing=comp_result.missing,
                )

        # Re-decompose if incomplete
        if not comp_result.complete and self.max_retries > 0:
            subtask_strs = self._re_decompose(query, agent_catalog, comp_result.missing)
            if subtask_strs:
                solv_result = self._select_agents(subtask_strs, agent_catalog)
                subtasks = [
                    Subtask(
                        description=st,
                        assigned_agent_id=solv_result.assignments.get(st),
                        solvability_score=solv_result.assignment_scores.get(st, 0.0),
                    )
                    for st in subtask_strs
                ]
                comp_result = self._check_completeness(
                    query, subtask_strs, solv_result.assignments
                )
                if trace:
                    trace.add(
                        "aop_redecompose",
                        subtasks=subtask_strs,
                        complete=comp_result.complete,
                    )

        if trace:
            trace.add("aop_plan_ready", subtask_count=len(subtasks))

        return AOPPlan(
            query=query,
            subtasks=subtasks,
            completeness=comp_result,
            solvability=solv_result,
            created_ts_ms=_now_ms(),
        )

    def execute_single_subtask(
        self,
        plan: AOPPlan,
        subtask_index: int,
        context: Dict[str, Any],
        trace: Optional[Trace] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single subtask from a previously prepared plan.
        Returns a response dict for that subtask only.
        """
        if subtask_index < 0 or subtask_index >= len(plan.subtasks):
            return {"error": f"Invalid subtask index: {subtask_index}"}

        st = plan.subtasks[subtask_index]

        # Execute just this one subtask
        executed = self._execute_subtasks([st], context)
        st = executed[0]

        # Record feedback
        self._record_feedback([st])

        if trace:
            trace.add(
                "aop_execute_single",
                subtask=st.description,
                agent=st.assigned_agent_id,
                success=st.success,
                index=subtask_index,
            )

        # Build response
        text = self._extract_readable_text(st.result) if st.result else ""
        remaining = [
            {
                "index": i,
                "subtask": s.description,
                "agent_id": s.assigned_agent_id,
            }
            for i, s in enumerate(plan.subtasks)
            if s.result is None and i != subtask_index
        ]

        return {
            "text": text,
            "answer": text,
            "score": st.solvability_score,
            "orchestration_pattern": "aop_task_result",
            "executed_subtask": {
                "subtask": st.description,
                "agent_id": st.assigned_agent_id,
                "success": st.success,
                "solvability_score": st.solvability_score,
                "latency_ms": st.latency_ms,
                "result": st.result,
            },
            "remaining_subtasks": remaining,
            "plan_query": plan.query,
        }

    # ── Estimator hot-swap ────────────────────────────────────────

    @staticmethod
    def _default_estimator(store: PerformanceStore) -> _Estimator:
        """Return the default estimator — neural if available, TF-IDF fallback."""
        try:
            from app.orchestration.neural_solvability_estimator import (
                NeuralSolvabilityEstimator,
            )

            return NeuralSolvabilityEstimator(store)
        except Exception as exc:
            print(
                f"[AOP] WARNING: Failed to load neural solvability estimator: {exc}\n"
                f"[AOP] Falling back to TF-IDF estimator. "
                f"Install 'sentence-transformers' and 'torch' to enable neural mode."
            )
            return SolvabilityEstimator(store)

    def swap_estimator(self, kind: str) -> str:
        """Hot-swap the solvability estimator at runtime.

        Args:
            kind: ``"neural"``, ``"tfidf"``, or ``"llm"``.

        Returns:
            The estimator kind now active.

        Raises:
            RuntimeError: If the requested estimator cannot be loaded
                (e.g. missing ``torch`` / ``sentence-transformers``).
        """
        if kind == "neural":
            try:
                from app.orchestration.neural_solvability_estimator import (
                    NeuralSolvabilityEstimator,
                )

                self.estimator = NeuralSolvabilityEstimator(self.store)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load neural estimator: {exc}. "
                    "Install 'sentence-transformers' and 'torch' to enable neural mode."
                ) from exc
        elif kind == "llm":
            from app.orchestration.llm_solvability_estimator import (
                LLMSolvabilityEstimator,
            )

            self.estimator = LLMSolvabilityEstimator(self.store)
        else:
            self.estimator = SolvabilityEstimator(self.store)
        return self.active_estimator_kind

    @property
    def active_estimator_kind(self) -> str:
        """Return ``'neural'``, ``'tfidf'``, or ``'llm'`` depending on which estimator is loaded."""
        cls_name = type(self.estimator).__name__
        if "Neural" in cls_name:
            return "neural"
        if "LLM" in cls_name:
            return "llm"
        return "tfidf"

    # ── Step 1: Task Decomposition ──────────────────────────────────

    @staticmethod
    def _clean_subtask(raw: str, agent_catalog: Dict[str, Dict[str, Any]]) -> str:
        """Strip agent-name prefixes the LLM sometimes embeds in subtask text.

        e.g. 'customer_qa_rag - INFORMATIONAL: ...' → 'INFORMATIONAL: ...'
             'customer_qa_rag: ...' → '...'
        """
        s = raw.strip()
        if not s:
            return s
        for aid in agent_catalog:
            # "agent_id - rest" or "agent_id: rest"
            for sep in (" - ", ": ", " — "):
                prefix = aid + sep
                if s.startswith(prefix):
                    s = s[len(prefix) :].strip()
                elif s.lower().startswith(prefix.lower()):
                    s = s[len(prefix) :].strip()
        return s

    def _decompose(
        self,
        query: str,
        agent_catalog: Dict[str, Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Use LLM to decompose a multi-intent query into atomic subtasks."""
        catalog_summary = []
        for aid, meta in agent_catalog.items():
            caps = meta.get("capabilities", [])
            desc = meta.get("description", "")
            catalog_summary.append(
                f"  - {aid}: {desc} (capabilities: {', '.join(caps)})"
            )
        catalog_str = "\n".join(catalog_summary)

        # Build conversation context string for multi-turn awareness
        history_str = ""
        if conversation_history:
            turns = []
            for turn in conversation_history[-5:]:
                q = turn.get("query", "")
                a = turn.get("answer", "")
                if q:
                    turns.append(f"  User: {q}")
                if a:
                    summary = a[:200] + "..." if len(a) > 200 else a
                    turns.append(f"  Assistant: {summary}")
            if turns:
                history_str = (
                    "\n\nConversation history (for context — "
                    "the current query may reference topics from earlier turns):\n"
                    + "\n".join(turns)
                )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task decomposition module for an AOP multi-agent system.\n"
                    "Given a user query and available agents, break the query into atomic subtasks.\n\n"
                    "Rules:\n"
                    "- Return 1-5 subtasks (prefer fewer).\n"
                    "- If conversation history is provided, use it to resolve references.\n"
                    "- Each subtask description must include the specific topic from context.\n\n"
                    "CRITICAL — Keep subtask descriptions SHORT and factual:\n"
                    "- Write subtasks as SIMPLE RETRIEVAL QUERIES or SIMPLE ACTION REQUESTS.\n"
                    "- Do NOT include formatting instructions, lists of what to include, or\n"
                    "  instructions like 'provide customer-facing explanation with A, B, C, D'.\n"
                    "- Good: 'INFORMATIONAL: cancellation policy for bank transactions'\n"
                    "- Bad: 'customer_qa_rag - INFORMATIONAL: cancellation policy...'\n"
                    "- Bad: 'Provide a comprehensive customer-facing explanation of cancellation "
                    "policy including timelines, fees, exceptions, and cite policy references...'\n"
                    "- Do NOT include agent names in subtask descriptions. Agent assignment is separate.\n\n"
                    "CRITICAL — Distinguish INFORMATIONAL vs ACTION queries:\n"
                    "- Questions ABOUT policies, procedures, or how things work = INFORMATIONAL.\n"
                    '  Examples: "tell me about refund policy", "how do I request a refund?",\n'
                    '  "what documents do I need to open an account?"\n'
                    "- Explicit requests to PERFORM an action = ACTION, even without full details.\n"
                    '  Examples: "I want to issue a refund", "process a refund for order #123",\n'
                    '  "I want to open an account", "refund my last transaction"\n'
                    "- The key distinction is USER INTENT: asking for information vs requesting\n"
                    "  an action be performed. A workflow agent can collect missing details\n"
                    "  (order number, amount) itself — do NOT require those for ACTION label.\n"
                    "- When in doubt, prefer INFORMATIONAL.\n\n"
                    'Return STRICT JSON: {"subtasks": ["subtask description 1", ...]}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nAvailable agents:\n{catalog_str}{history_str}",
            },
        ]

        try:
            raw = chat_json(messages=messages, model=self.model)
            subtasks = raw.get("subtasks", [])
            if isinstance(subtasks, list):
                cleaned = [self._clean_subtask(str(s), agent_catalog) for s in subtasks]
                return [s for s in cleaned if s]
            return []
        except Exception as e:
            print(f"[AOP] decompose failed: {e}")
            return []

    def _re_decompose(
        self,
        query: str,
        agent_catalog: Dict[str, Dict[str, Any]],
        missing_aspects: List[str],
    ) -> List[str]:
        """Re-decompose with hints about missing aspects."""
        catalog_summary = []
        for aid, meta in agent_catalog.items():
            caps = meta.get("capabilities", [])
            desc = meta.get("description", "")
            catalog_summary.append(
                f"  - {aid}: {desc} (capabilities: {', '.join(caps)})"
            )
        catalog_str = "\n".join(catalog_summary)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task decomposition module. A previous decomposition was incomplete.\n"
                    "Re-decompose the query, making sure to address the missing aspects.\n\n"
                    "Rules:\n"
                    "- Return 1-5 subtasks MAXIMUM (prefer fewer). Merge related aspects into a single subtask.\n"
                    "- Do NOT create one subtask per missing aspect — group them logically.\n\n"
                    "CRITICAL — Keep subtask descriptions SHORT and factual.\n"
                    "CRITICAL — Distinguish INFORMATIONAL vs ACTION queries:\n"
                    "- Questions ABOUT policies, procedures, or how things work = INFORMATIONAL.\n"
                    "- Explicit requests to PERFORM an action = ACTION, even without full details.\n"
                    "  The workflow agent can collect missing details itself.\n"
                    "- When in doubt, prefer INFORMATIONAL.\n\n"
                    'Return STRICT JSON: {"subtasks": ["subtask description 1", ...]}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Available agents:\n{catalog_str}\n\n"
                    f"Missing aspects from previous attempt:\n"
                    + "\n".join(f"  - {m}" for m in missing_aspects)
                ),
            },
        ]

        try:
            raw = chat_json(messages=messages, model=self.model)
            subtasks = raw.get("subtasks", [])
            if isinstance(subtasks, list):
                result = [str(s).strip() for s in subtasks if str(s).strip()]
                return result[:5]  # Hard cap — never exceed 5 subtasks
            return []
        except Exception:
            return []

    # ── AOP candidate filtering ────────────────────────────────────

    # Agent kinds that are internal (leaf-level) and should never be
    # assigned top-level AOP subtasks.  Only primary agents (RAG,
    # workflow runners) should be candidates.
    _EXCLUDED_KINDS = frozenset({"tool_operator", "guardrails"})

    # ── Description-based heuristic (FALLBACK ONLY) ──────────────
    # Used only when the declarative ``aop_eligible`` attribute is
    # absent (legacy factory specs that predate the attribute).
    _ROUTING_SIGNALS = frozenset(
        {
            "routes them to",
            "route requests to",
            "routes to the appropriate",
            "classifies incoming",
            "classify and route",
        }
    )
    _CUSTOMER_SERVING_SIGNALS = frozenset(
        {
            "customer-facing",
            "user-facing",
            "frontline",
            "answer customer",
            "answers customer",
            "answer question",
            "answers question",
            "orchestrates end-to-end",
            "end-to-end",
            "collect required",
            "from the customer",
        }
    )
    _INTERNAL_SIGNALS = frozenset(
        {
            "compliance",
            "guardrails",
            "validation",
            "audit",
            "blocking",
            "so other agents",
            "for other agents",
            "policy snippets",
        }
    )

    @classmethod
    def _is_aop_eligible(cls, meta: Dict[str, Any]) -> bool:
        """Return True if an agent should be an AOP subtask candidate.

        Primary: uses the declarative ``aop_eligible`` attribute set at
        build time by the planning LLMs (infer_capabilities →
        blueprint_creator → spec_builder).

        Fallback: for legacy specs without the attribute, falls back to
        a description-based heuristic.
        """
        # ── Primary: declarative attribute ──
        flag = meta.get("aop_eligible")
        if isinstance(flag, bool):
            return flag

        # ── Fallback: description-based heuristic (legacy specs) ──
        desc = (meta.get("description") or "").lower()
        caps = " ".join(str(c).lower() for c in (meta.get("capabilities") or []))
        combined = desc + " " + caps

        if any(sig in combined for sig in cls._ROUTING_SIGNALS):
            return False
        if any(sig in combined for sig in cls._CUSTOMER_SERVING_SIGNALS):
            return True
        if any(sig in combined for sig in cls._INTERNAL_SIGNALS):
            return False

        return True

    def _aop_candidate_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Return only primary agents eligible for AOP subtask assignment.

        Filters out:
        - tool_operator / guardrails (leaf agents invoked by workflows)
        - Internal knowledge_rag agents (compliance checkers, not user-facing)
        """
        all_meta = self.registry.all_meta()
        return {
            aid: meta
            for aid, meta in all_meta.items()
            if meta.get("agent_kind") not in self._EXCLUDED_KINDS
            and meta.get("type") not in self._EXCLUDED_KINDS
            and self._is_aop_eligible(meta)
        }

    # ── Step 2: Agent Selection ─────────────────────────────────────

    def _select_agents(
        self, subtasks: List[str], agent_catalog: Dict[str, Dict[str, Any]]
    ) -> SolvabilityResult:
        """Score all (subtask, agent) pairs via SolvabilityEstimator."""
        return self.estimator.estimate(subtasks, agent_catalog)

    # ── Step 3: Completeness Check ──────────────────────────────────

    def _check_completeness(
        self,
        query: str,
        subtasks: List[str],
        assignments: Dict[str, str],
    ) -> CompletenessResult:
        """Audit plan for completeness and non-redundancy."""
        return self.completeness.check(query, subtasks, assignments)

    # ── Step 4: Execution ───────────────────────────────────────────

    def _is_action_agent(self, agent_id: str) -> bool:
        """Return True if the agent requires concrete user-provided transaction context.

        An agent is considered an "action agent" only if it requires user context
        AND is NOT customer-facing.  Customer-facing agents (e.g. FAQ agents with
        ``customer_facing: true``) can answer informational queries even when
        ``requires_user_context`` is set — they retrieve knowledge and ask
        clarifying questions rather than performing irreversible actions.
        """
        meta = self.registry.all_meta().get(agent_id, {})
        if not meta.get("requires_user_context"):
            return False
        # Agents with customer-facing docs can handle informational queries
        # safely without requiring user-provided transaction details.
        if meta.get("has_customer_facing_docs"):
            return False
        return True

    def _all_informational(self, subtask_strs: List[str]) -> bool:
        """Return True when all subtasks are informational (no action needed).

        Detection (in priority order):
        1. LLM label: if ANY subtask has an ACTION label → False.
        2. LLM label: if ALL subtasks have INFORMATIONAL label → True.
        3. Action-signal fallback: if no subtask contains concrete transaction
           identifiers (order#, transaction#, amounts), treat as informational.
        """
        if not subtask_strs:
            return False

        # Check 1: any ACTION label → not all informational
        any_labeled_action = any("ACTION" in s.upper() for s in subtask_strs)
        if any_labeled_action:
            return False

        # Check 2: all INFORMATIONAL labels → informational
        all_labeled_info = all("INFORMATIONAL" in s.upper() for s in subtask_strs)
        if all_labeled_info:
            return True

        # Check 3: no action signals in any subtask → informational
        return not any(self._action_signals.search(s) for s in subtask_strs)

    def _execute_subtasks(
        self,
        subtasks: List[Subtask],
        context: Dict[str, Any],
    ) -> List[Subtask]:
        """Execute subtasks by delegating to assigned agents.

        Independent subtasks run in parallel via ThreadPoolExecutor.
        """
        # Pre-check each subtask: resolve agent, apply guardrails
        runnable: List[Subtask] = []
        for st in subtasks:
            if not st.assigned_agent_id:
                st.success = False
                st.result = {"error": "No agent assigned"}
                continue

            agent = self.registry.get(st.assigned_agent_id)
            if not agent:
                st.success = False
                st.result = {
                    "error": f"Agent {st.assigned_agent_id} not found in registry"
                }
                continue

            if self._is_action_agent(st.assigned_agent_id):
                _label, _ = parse_aop_label(st.description)
                labeled_informational = _label == "INFORMATIONAL"
                labeled_action = _label == "ACTION"

                if not (labeled_action or labeled_informational):
                    original_query = context.get("original_query", "") or ""
                    has_transaction = self._action_signals.search(
                        original_query if original_query else st.description
                    )
                    if not has_transaction:
                        agent_meta = self.registry.all_meta().get(
                            st.assigned_agent_id, {}
                        )
                        msg = agent_meta.get("missing_context_message") or (
                            "To help with this I'll need a few more details — "
                            "could you share your order or transaction reference?"
                        )
                        st.success = False
                        st.result = {"error": "guardrail_blocked", "message": msg}
                        print(
                            f"[AOP-GUARD] Blocked action agent {st.assigned_agent_id} — "
                            f"no transaction details in query"
                        )
                        continue

            runnable.append(st)

        if not runnable:
            return subtasks

        def _run_one(st: Subtask) -> None:
            _, agent_query = parse_aop_label(st.description)
            agent = self.registry.get(st.assigned_agent_id)
            t0 = _now_ms()
            try:
                result = agent.handle(
                    {"query": agent_query, "text": agent_query, "context": context}
                )
                st.result = result
                st.success = not result.get("error")
                try:
                    st.solvability_score = float(
                        result.get("score", st.solvability_score)
                    )
                except (TypeError, ValueError):
                    pass
            except Exception as e:
                st.result = {"error": str(e)}
                st.success = False
            st.latency_ms = _now_ms() - t0

        if len(runnable) == 1:
            _run_one(runnable[0])
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                list(pool.map(_run_one, runnable))

        return subtasks

    # ── Step 5: Feedback Loop ───────────────────────────────────────

    def _record_feedback(self, subtasks: List[Subtask]) -> None:
        """Write execution results to performance store."""
        for st in subtasks:
            if not st.assigned_agent_id:
                continue
            try:
                self.store.append(
                    ExecutionRecord(
                        agent_id=st.assigned_agent_id,
                        subtask=st.description,
                        success=st.success,
                        score=st.solvability_score,
                        latency_ms=st.latency_ms,
                    )
                )
            except Exception as e:
                print(f"[AOP] feedback write failed for {st.assigned_agent_id}: {e}")

    # ── Response Assembly ───────────────────────────────────────────

    @staticmethod
    def _extract_readable_text(result: Dict[str, Any]) -> str:
        """
        Extract a human-readable string from an agent result dict.

        Agents return varied formats:
          - FAQ/RAG: {"answer": "...", "score": ...}
          - Workflow: {"message": "...", "current_state": ..., "slots": ...}
          - Tool: {"text": "...", "tool_result": ...}
          - Generic: {"response": "..."}
        """
        if not result:
            return ""

        # Direct text fields (preferred)
        for key in ("text", "answer", "message", "response"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Workflow-style: build a readable summary
        if "current_state" in result or "workflow_id" in result:
            parts = []
            if result.get("message"):
                parts.append(str(result["message"]))
            if result.get("action"):
                parts.append(f"Action: {result['action']}")
            if result.get("missing_slots"):
                missing = result["missing_slots"]
                if isinstance(missing, list):
                    parts.append(
                        f"Need more info: {', '.join(str(s) for s in missing)}"
                    )
            if result.get("current_state"):
                parts.append(f"(State: {result['current_state']})")
            if parts:
                return " | ".join(parts)

        # Nested result dict
        if isinstance(result.get("result"), dict):
            return AOPCoordinator._extract_readable_text(result["result"])

        # Last resort: skip internal keys, show only short values
        skip = {
            "slots",
            "history",
            "mapper",
            "allowed_events",
            "slot_defs",
            "context",
            "thread_id",
            "workflow_id",
            "request_id",
            "router_plan",
        }
        summary_parts = []
        for k, v in result.items():
            if k in skip:
                continue
            s = str(v)
            if len(s) < 200:
                summary_parts.append(f"{k}: {s}")
        return (
            " | ".join(summary_parts[:5]) if summary_parts else "(no readable content)"
        )

    def _assemble_composite_response(
        self,
        query: str,
        subtasks: List[Subtask],
        completeness: CompletenessResult,
        solvability: SolvabilityResult,
        total_latency_ms: int,
    ) -> Dict[str, Any]:
        """Combine subtask results into a spine-compatible response dict."""
        # Collect individual answers
        answers = []
        for st in subtasks:
            if st.result and not st.result.get("error"):
                text = self._extract_readable_text(st.result)
                answers.append(f"[{st.assigned_agent_id}] {text}")
            elif st.result and st.result.get("error") == "guardrail_blocked":
                # Surface the helpful message from the guardrail, not the error code
                msg = st.result.get("message", "This action requires more details.")
                answers.append(f"[{st.assigned_agent_id}] {msg}")
            else:
                err = (
                    st.result.get("error", "unknown error")
                    if st.result
                    else "no result"
                )
                answers.append(f"[{st.assigned_agent_id}] Unable to complete: {err}")

        combined_text = "\n\n".join(answers)

        # Average score across successful subtasks
        successful = [st for st in subtasks if st.success]
        avg_score = (
            sum(st.solvability_score for st in successful) / len(successful)
            if successful
            else 0.0
        )

        return {
            "text": combined_text,
            "answer": combined_text,
            "score": avg_score,
            "orchestration_pattern": "hierarchical_delegation",
            "subtask_results": [
                {
                    "subtask": st.description,
                    "agent_id": st.assigned_agent_id,
                    "success": st.success,
                    "solvability_score": st.solvability_score,
                    "latency_ms": st.latency_ms,
                    "result": st.result,
                }
                for st in subtasks
            ],
            "completeness": {
                "complete": completeness.complete,
                "missing": completeness.missing,
                "coverage_ratio": completeness.coverage_ratio,
                "reasoning": completeness.reasoning,
            },
            "solvability": {
                "assignments": solvability.assignments,
                "assignment_scores": {
                    k: round(v, 4) for k, v in solvability.assignment_scores.items()
                },
            },
            "total_latency_ms": total_latency_ms,
        }


def _now_ms() -> int:
    return int(time.time() * 1000)
