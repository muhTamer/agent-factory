# app/runtime/domain_agent_engine.py
"""
Domain Agent Engine — ReAct (Reason + Act) loop for autonomous domain specialists.

Each domain agent combines:
  - Knowledge retrieval (RAG via TF-IDF)
  - Tool selection and execution (ITool interface)
  - Policy enforcement (natural language constraints in prompt)
  - LLM-driven reasoning (multi-step observe → think → act loop)

This replaces both knowledge_rag and workflow_runner with a single
agent type that has genuine agency — it decides what to do, not a
state machine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.shared.rag import Index, query_index

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DomainAgentConfig:
    """Configuration for a domain agent's ReAct engine."""

    agent_id: str
    domain: str  # e.g. "refunds", "orders", "accounts"
    goal: str  # e.g. "Help customers with refund requests"
    policies: List[str] = field(default_factory=list)
    max_steps: int = 10
    model: str = "gpt-5-mini"
    temperature: float = (
        1.0  # 1.0 is the only value some models accept (o-series, gpt-5-mini)
    )
    top_k: int = 5
    retrieval_threshold: float = 0.12
    # Dense retrieval (hybrid fusion with TF-IDF)
    enable_dense_retrieval: bool = False
    dense_weight: float = 0.6  # Weight for dense (embedding) scores
    sparse_weight: float = 0.4  # Weight for sparse (TF-IDF) scores


@dataclass
class ReActStep:
    """One step of the ReAct reasoning loop."""

    step_number: int
    thought: str
    action: str  # retrieve_knowledge | call_tool | respond | ask_user | escalate
    action_input: Dict[str, Any]
    observation: str
    timestamp: float = 0.0


@dataclass
class ThreadState:
    """Per-thread reasoning state for multi-turn conversations."""

    thread_id: str = "default"
    step_history: List[ReActStep] = field(default_factory=list)
    accumulated_slots: Dict[str, Any] = field(default_factory=dict)
    pending_question: Optional[str] = None
    turn_count: int = 0
    original_query: Optional[str] = None  # First query that started this thread
    # Cache policy content retrieved on first turn so subsequent turns
    # have the full workflow steps without needing to re-retrieve.
    cached_policy_content: Optional[str] = None


# ---------------------------------------------------------------------------
# Terminal actions
# ---------------------------------------------------------------------------
_TERMINAL_ACTIONS = frozenset({"respond", "ask_user", "escalate"})


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DomainAgentEngine:
    """
    ReAct reasoning engine for domain agents.

    Implements the Observe → Think → Act loop:
      1. OBSERVE: User query + conversation history + retrieved context
      2. THINK:   LLM reasons about what action to take
      3. ACT:     Execute the chosen action
      4. OBSERVE: Result of the action
      5. REPEAT until terminal action or max_steps
    """

    def __init__(
        self,
        config: DomainAgentConfig,
        index: Index,
        tools: Dict[str, Any],  # name → ITool
        llm_fn: Optional[Callable] = None,
        memory: Optional[Any] = None,
        embed_fn: Optional[Callable] = None,
        dense_vecs: Optional[List[List[float]]] = None,
    ) -> None:
        self.config = config
        self.index = index
        self.tools = tools or {}
        self._llm_fn = llm_fn
        self._memory = memory
        self._embed_fn = embed_fn
        self._dense_vecs = dense_vecs
        self._thread_states: Dict[str, ThreadState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        query: str,
        thread_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the ReAct loop for a query. Returns IAgent-compatible response."""
        ctx = context or {}
        state = self._get_or_create_thread(thread_id)
        state.turn_count += 1

        # Remember the original query that started this thread
        if state.original_query is None:
            state.original_query = query

        # Carry over any accumulated slots from spine (cross-agent handoff)
        ext_slots = ctx.get("_accumulated_slots")
        if isinstance(ext_slots, dict):
            state.accumulated_slots.update(ext_slots)

        steps: List[ReActStep] = []

        for step_num in range(1, self.config.max_steps + 1):
            # Build prompt
            messages = self._build_react_prompt(query, state, steps, ctx)

            # LLM THINKS
            llm_response = self._call_llm(messages)
            thought, action, action_input = self._parse_react_response(llm_response)

            # ACT
            observation = self._execute_action(action, action_input, state)

            # Cache policy content from retrieval so subsequent turns
            # have the full workflow without re-retrieving.
            if (
                action == "retrieve_knowledge"
                and observation
                and not state.cached_policy_content
            ):
                if observation.startswith("Retrieved from source(s):"):
                    state.cached_policy_content = observation

            step = ReActStep(
                step_number=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
                timestamp=time.time(),
            )
            steps.append(step)
            state.step_history.append(step)

            # Terminal actions end the loop
            if action in _TERMINAL_ACTIONS:
                break

        return self._build_response(steps, state, query)

    # ------------------------------------------------------------------
    # Thread state
    # ------------------------------------------------------------------

    def _get_or_create_thread(self, thread_id: str) -> ThreadState:
        if thread_id not in self._thread_states:
            self._thread_states[thread_id] = ThreadState(thread_id=thread_id)
        return self._thread_states[thread_id]

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: str, action_input: Dict[str, Any], state: ThreadState
    ) -> str:
        """Execute a ReAct action and return an observation string."""

        if action == "retrieve_knowledge":
            return self._action_retrieve(action_input)

        if action == "call_tool":
            return self._action_call_tool(action_input, state)

        if action == "respond":
            return action_input.get("answer", "")

        if action == "ask_user":
            question = action_input.get("question", "Could you provide more details?")
            state.pending_question = question
            return f"Asking user: {question}"

        if action == "escalate":
            reason = action_input.get("reason", "Unable to resolve")
            return f"Escalating: {reason}"

        return f"Unknown action: {action}"

    # Maximum number of chunks a source can have to qualify for full
    # source-expansion.  Sources with more chunks (e.g. large FAQ CSVs)
    # return only the directly matched chunks sorted by relevance.
    _SOURCE_EXPANSION_LIMIT = 50

    def _action_retrieve(self, action_input: Dict[str, Any]) -> str:
        """Retrieve knowledge from the agent's corpus using hybrid retrieval.

        Uses adaptive source-expansion:
        - Small sources (≤ _SOURCE_EXPANSION_LIMIT chunks, e.g. policy YAMLs):
          expand to full document so the agent sees complete context.
        - Large sources (> limit, e.g. BankFAQs.csv with 1700+ Q&A pairs):
          return only the matched chunks sorted by relevance score.
          Each Q&A pair is independent — expansion would flood the context
          with irrelevant entries and bury the actual matches.
        """
        search_query = action_input.get("query", "")
        if not search_query:
            return "No search query provided."

        # Sparse retrieval (TF-IDF)
        sparse_hits = query_index(self.index, search_query, k=self.config.top_k)

        # Hybrid fusion: combine sparse + dense if available
        if (
            self.config.enable_dense_retrieval
            and self._embed_fn is not None
            and self._dense_vecs is not None
            and self.index.items
        ):
            hits = self._hybrid_retrieve(search_query, sparse_hits)
        else:
            hits = sparse_hits

        if not hits or hits[0][0] < self.config.retrieval_threshold:
            return "No relevant information found in knowledge base."

        # Count chunks per source for expansion decision
        source_chunk_counts: Dict[str, int] = {}
        for corpus_item in self.index.items:
            source_chunk_counts[corpus_item.source] = (
                source_chunk_counts.get(corpus_item.source, 0) + 1
            )

        # Separate matched hits into expandable (small) and direct (large) sources
        small_sources: set = set()  # expand to full document
        direct_hits: List[Any] = []  # return matched chunks only
        seen_texts: set = set()

        for score, item in hits:
            if score < self.config.retrieval_threshold:
                continue
            chunk_count = source_chunk_counts.get(item.source, 0)
            if chunk_count <= self._SOURCE_EXPANSION_LIMIT:
                small_sources.add(item.source)
            else:
                if item.text not in seen_texts:
                    direct_hits.append(item)
                    seen_texts.add(item.text)

        # Expand small sources (full document context)
        expanded: List[Any] = []
        if small_sources:
            for corpus_item in self.index.items:
                if corpus_item.source in small_sources:
                    if corpus_item.text not in seen_texts:
                        expanded.append(corpus_item)
                        seen_texts.add(corpus_item.text)

        # Build passages: expanded first, then direct hits
        all_sources: set = small_sources | {h.source for h in direct_hits}
        passages: List[str] = []
        for item in expanded:
            passages.append(item.text[:1500])
        for item in direct_hits:
            passages.append(item.text[:1500])

        return (
            f"Retrieved from source(s): "
            f"{', '.join(all_sources)}\n\n" + "\n---\n".join(passages)
        )

    def _hybrid_retrieve(
        self,
        query: str,
        sparse_hits: List[Tuple[float, Any]],
    ) -> List[Tuple[float, Any]]:
        """Fuse sparse (TF-IDF) and dense (embedding) scores."""
        try:
            q_vecs = self._embed_fn([query])
            q_vec = q_vecs[0]
        except Exception:
            return sparse_hits  # fallback to sparse only

        # Build sparse lookup: index → score
        sparse_lookup: Dict[int, float] = {}
        items = self.index.items
        for score, item in sparse_hits:
            for idx, corpus_item in enumerate(items):
                if corpus_item is item:
                    sparse_lookup[idx] = score
                    break

        # Fuse scores
        fused: List[Tuple[float, Any]] = []
        for i, (dv, item) in enumerate(zip(self._dense_vecs, items)):
            dense_score = max(0.0, min(1.0, self._dot(q_vec, dv)))
            sparse_score = sparse_lookup.get(i, 0.0)
            combined = (
                self.config.sparse_weight * sparse_score
                + self.config.dense_weight * dense_score
            )
            fused.append((combined, item))

        fused.sort(key=lambda x: x[0], reverse=True)
        return fused[: self.config.top_k]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    def _action_call_tool(
        self, action_input: Dict[str, Any], state: ThreadState
    ) -> str:
        """Call a tool via the ITool interface."""
        tool_name = action_input.get("tool", "")
        tool_args = action_input.get("args", {})

        if not tool_name:
            return "Error: No tool name specified."

        tool = self.tools.get(tool_name)
        if tool is None:
            available = list(self.tools.keys())
            return f"Error: Tool '{tool_name}' not available. Available: {available}"

        try:
            # Merge accumulated slots + explicit args for tool context
            slots = {**state.accumulated_slots, **tool_args}
            result = tool.execute(slots, {"thread_id": state.thread_id})

            # Accumulate any slot updates from tool result
            if isinstance(result, dict):
                state.accumulated_slots.update(result)
                return f"Tool '{tool_name}' returned: {json.dumps(result)}"
            return f"Tool '{tool_name}' returned: {result}"
        except Exception as e:
            return f"Tool '{tool_name}' failed: {str(e)}"

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the LLM and return parsed JSON."""
        if self._llm_fn is None:
            # Fallback: no LLM available, escalate immediately
            return {
                "thought": "No LLM available for reasoning.",
                "action": "escalate",
                "action_input": {"reason": "LLM not configured"},
            }

        try:
            result = self._llm_fn(
                messages=messages,
                model=self.config.model,
                temperature=self.config.temperature,
            )
            if isinstance(result, dict):
                return result
            return {"raw": result}
        except Exception as e:
            return {
                "thought": f"LLM call failed: {e}",
                "action": "escalate",
                "action_input": {"reason": f"LLM error: {e}"},
            }

    def _parse_react_response(
        self, raw: Dict[str, Any]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Parse LLM response into (thought, action, action_input)."""
        thought = str(raw.get("thought", "")).strip() or "No reasoning provided."

        action = str(raw.get("action", "")).strip().lower()
        if action not in {
            "retrieve_knowledge",
            "call_tool",
            "respond",
            "ask_user",
            "escalate",
        }:
            # If the LLM returned an unrecognized action, treat as respond
            # with the thought as the answer (graceful degradation)
            answer = raw.get("action_input", {}).get("answer", thought)
            return thought, "respond", {"answer": answer}

        action_input = raw.get("action_input", {})
        if not isinstance(action_input, dict):
            action_input = {}

        return thought, action, action_input

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_react_prompt(
        self,
        query: str,
        thread_state: ThreadState,
        previous_steps: List[ReActStep],
        context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Build the ReAct prompt messages for the LLM."""

        # Tool descriptions
        tool_lines: List[str] = []
        for name, tool in self.tools.items():
            desc = "No description"
            if hasattr(tool, "describe"):
                d = tool.describe()
                desc = d.get("description", desc) if isinstance(d, dict) else desc
            tool_lines.append(f"  - {name}: {desc}")
        tools_str = "\n".join(tool_lines) if tool_lines else "  (none)"

        # Policies
        policies_str = (
            "\n".join(f"  - {p}" for p in self.config.policies)
            if self.config.policies
            else "  (none)"
        )

        # Conversation history from memory
        history_str = ""
        if self._memory:
            try:
                ctx = self._memory.get_conversation_context(
                    thread_state.thread_id, limit=5
                )
                if ctx:
                    turns = []
                    for t in ctx:
                        turns.append(f"  User: {t.get('query', '')}")
                        ans = t.get("answer", "")
                        if ans:
                            turns.append(f"  Agent: {ans[:200]}")
                    history_str = "\nConversation history:\n" + "\n".join(turns)
            except Exception:
                pass

        # Previous steps in this turn
        steps_str = ""
        if previous_steps:
            parts = []
            for s in previous_steps:
                parts.append(
                    f"Step {s.step_number}:\n"
                    f"  Thought: {s.thought}\n"
                    f"  Action: {s.action}({json.dumps(s.action_input)})\n"
                    f"  Observation: {s.observation[:800]}"
                )
            steps_str = "\n\nPrevious reasoning steps:\n" + "\n\n".join(parts)

        # Accumulated slots
        slots_str = ""
        if thread_state.accumulated_slots:
            slots_str = (
                f"\nInformation gathered so far: "
                f"{json.dumps(thread_state.accumulated_slots)}"
            )

        system_prompt = (
            f"You are a {self.config.domain} domain specialist agent.\n\n"
            f"Goal: {self.config.goal}\n\n"
            "You reason step-by-step using the ReAct framework (Reason + Act).\n"
            "At each step you MUST output a JSON object with exactly three fields:\n"
            '  "thought": your reasoning about what to do next\n'
            '  "action": one of the available actions\n'
            '  "action_input": parameters for that action\n\n'
            "Available actions:\n"
            '  1. retrieve_knowledge({"query": "search terms"})\n'
            "     - Search your knowledge base for relevant information\n"
            '  2. call_tool({"tool": "tool_name", "args": {"key": "value"}})\n'
            "     - Execute a tool. Available tools:\n"
            f"{tools_str}\n"
            '  3. respond({"answer": "your final answer to the user"})\n'
            "     - Provide a final answer. Use this when you have enough information.\n"
            '  4. ask_user({"question": "what you need from the user"})\n'
            "     - Ask the user for information you need but don't have.\n"
            '  5. escalate({"reason": "why this needs human attention"})\n'
            "     - LAST RESORT ONLY. Use when you have NO relevant knowledge AND "
            "no tools to help. There is no human specialist behind this action — "
            "escalation ends the conversation.\n\n"
            "Policy guidance (follow the spirit, not rigidly):\n"
            f"{policies_str}\n"
            + (
                f"\n--- RETRIEVED POLICY (you MUST follow ONLY these steps) ---\n"
                f"{thread_state.cached_policy_content}\n"
                f"--- END OF POLICY ---\n\n"
                if thread_state.cached_policy_content
                else "\n"
            )
            + "Guidelines:\n"
            "- Retrieve knowledge ONCE before answering factual questions.\n"
            "- Do NOT call retrieve_knowledge more than twice per turn. "
            "After retrieving, use the passages you found to respond.\n"
            "- NEVER repeat the same action with the same or similar input.\n"
            "- If a tool fails, explain the issue and suggest next steps.\n\n"
            "CRITICAL — Retrieval quality and honesty:\n"
            "- After retrieving knowledge, CRITICALLY evaluate: do the passages "
            "DIRECTLY answer the user's specific question?\n"
            "- If YES and they all relate to ONE specific topic → use respond() "
            "and base your answer ONLY on what the passages say.\n"
            "- If the retrieved passages cover MULTIPLE DISTINCT topics, products, "
            "or categories (e.g. travel insurance vs home insurance vs forex), "
            "you MUST use ask_user() to ask the user which specific one they mean. "
            "List the options you found. Do NOT try to summarize all of them.\n"
            "- If the passages are about a DIFFERENT topic, or only tangentially "
            "related, they do NOT count as relevant. Do NOT fabricate an answer "
            "by combining unrelated passages.\n"
            "- If the question is vague or ambiguous, use ask_user() to clarify "
            "what specifically the user wants to know, so you can retrieve "
            "more targeted information.\n"
            "- If after retrieval you genuinely have NO matching content for "
            "the user's question, HONESTLY say you don't have that specific "
            "information and offer to connect them with someone who can help. "
            "Use respond() for this — do NOT use escalate.\n"
            "  Example: 'I don't have specific details about that. "
            "Would you like me to connect you with a specialist who can help?'\n"
            "- NEVER make up facts, figures, timelines, or procedures that are "
            "not in the retrieved passages. If it's not in the knowledge base, "
            "you don't know it.\n\n"
            "CRITICAL — Internal information must NEVER be shared with the user:\n"
            "- Policy documents you retrieve are INTERNAL INSTRUCTIONS for you.\n"
            "  They tell YOU how to act. They are NOT for the customer to see.\n"
            "- NEVER mention: policy names, policy IDs, version numbers, "
            "regulatory codes (PSD2, AMLD5, GDPR, PCI-DSS, etc.), "
            "compliance framework names, internal process names, "
            "approval thresholds, rule IDs, section numbers, "
            "or any other internal/operational detail.\n"
            "- NEVER say 'per our policy', 'our refunds policy states', "
            "'according to the policy', 'compliance checks', "
            "'AML/KYC requirements', or similar.\n"
            "- Instead, translate policy requirements into PLAIN, "
            "NATURAL customer-friendly language. For example:\n"
            "  BAD:  'Per our refunds policy, we must complete AML/KYC verification'\n"
            "  GOOD: 'I just need to verify your identity before we proceed'\n"
            "  BAD:  'We follow PSD2/AMLD5/GDPR compliance checks'\n"
            "  GOOD: 'Let me confirm a couple of details for security'\n\n"
            "CRITICAL — Concise, gradual responses:\n"
            "- Ask ONE question at a time. Do NOT list multiple questions "
            "or ask for several pieces of information in a single turn.\n"
            "- Keep responses SHORT — 1 to 3 sentences maximum when asking questions.\n"
            "- Do NOT preview or explain future steps, the full process, "
            "or what you will ask next. Handle one step at a time.\n"
            "- Do NOT add bullet lists, numbered steps, or 'what happens next' sections.\n"
            "- Be warm and conversational, not procedural.\n\n"
            "CRITICAL — Policy grounding:\n"
            "- When you retrieve a policy document, FOLLOW its documented rules "
            "and workflow step by step. Do NOT invent your own procedure.\n"
            "- ONLY perform actions and ask questions that are EXPLICITLY listed "
            "in the policy workflow steps. If a step is not in the policy, "
            "do NOT add it yourself. For example, if the policy does not require "
            "card verification, do NOT ask for the last 4 digits of a card.\n"
            "- Your reasoning (the 'thought' field) should reference policy rules, "
            "but your customer-facing answer must NEVER cite them.\n"
            "- Only ask the user for information that the policy SPECIFICALLY "
            "requires AND that you cannot look up via your tools.\n"
            "- NEVER invent security checks, verification steps, or additional "
            "requirements beyond what the policy document explicitly states.\n\n"
            "CRITICAL — Tool-first approach:\n"
            "- If the user provides a transaction ID, order number, or account "
            "reference, use call_tool to look it up BEFORE asking for more details.\n"
            "- Do NOT ask the user for information you could retrieve via tools "
            "(e.g. transaction details, account status, payment history).\n"
            "- After looking up data via tools, check the results against policy "
            "rules before asking the user for anything else.\n\n"
            'Return STRICT JSON: {"thought": "...", "action": "...", "action_input": {...}}'
        )

        # User content
        user_content = f"User query: {query}{history_str}{slots_str}{steps_str}"

        # If resuming after ask_user, include full conversation context
        if thread_state.pending_question:
            # Build summary of previous turns' reasoning so the LLM
            # remembers the original intent and what it already did.
            prev_steps_str = ""
            prev_steps = [
                s
                for s in thread_state.step_history
                if s not in previous_steps  # exclude current turn's steps
            ]
            if prev_steps:
                parts = []
                for s in prev_steps[-6:]:  # last 6 steps max to avoid bloat
                    # Give retrieval observations more room so policy
                    # workflow steps aren't lost across turns.
                    obs_limit = 600 if s.action == "retrieve_knowledge" else 200
                    parts.append(
                        f"Step {s.step_number}: "
                        f"thought={s.thought[:200]} → "
                        f"{s.action}({json.dumps(s.action_input)}) → "
                        f"{s.observation[:obs_limit]}"
                    )
                prev_steps_str = (
                    "\n\nPrevious reasoning from earlier turns:\n" + "\n".join(parts)
                )

            original_ctx = ""
            if thread_state.original_query and thread_state.original_query != query:
                original_ctx = (
                    f"Original user question: {thread_state.original_query}\n"
                )

            user_content = (
                f"{original_ctx}"
                f'You previously asked the user: "{thread_state.pending_question}"\n'
                f"User responded: {query}"
                f"{slots_str}{prev_steps_str}{steps_str}"
            )
            thread_state.pending_question = None

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    def _build_response(
        self,
        steps: List[ReActStep],
        state: ThreadState,
        query: str,
    ) -> Dict[str, Any]:
        """Build an IAgent-compatible response dict from ReAct steps."""
        last_step = steps[-1] if steps else None
        if not last_step:
            return {"answer": "I could not process your request.", "score": 0.0}

        response: Dict[str, Any] = {
            "intent": self.config.domain,
            "domain": self.config.domain,
            "agent_id": self.config.agent_id,
        }

        if last_step.action == "respond":
            answer = last_step.action_input.get("answer", last_step.observation)
            response["answer"] = answer
            response["text"] = answer

        elif last_step.action == "ask_user":
            question = last_step.action_input.get("question", "")
            response["answer"] = question
            response["text"] = question
            response["needs_input"] = True
            response["domain_agent_clarification"] = True

        elif last_step.action == "escalate":
            reason = last_step.action_input.get("reason", "Unable to resolve")
            response["answer"] = (
                "I don't have enough information to fully answer this question. "
                "Could you try rephrasing or providing more details?"
            )
            response["text"] = response["answer"]
            response["escalation"] = True
            response["escalation_reason"] = reason

        else:
            # Max steps reached without terminal action
            response["answer"] = (
                "I wasn't able to find a complete answer. "
                "Could you try asking in a different way?"
            )
            response["text"] = response["answer"]
            response["escalation"] = True
            response["escalation_reason"] = "Max reasoning steps reached"

        # ---- Explainability / RQ2: Full reasoning trace ----
        response["react_trace"] = [
            {
                "step": s.step_number,
                "thought": s.thought,
                "action": s.action,
                "action_input": s.action_input,
                "observation": s.observation[:600],
            }
            for s in steps
        ]

        # Detailed tool-call results for the frontend trace panel
        tool_results: List[Dict[str, Any]] = []
        for s in steps:
            if s.action == "call_tool":
                tool_results.append(
                    {
                        "step": s.step_number,
                        "tool": s.action_input.get("tool", "unknown"),
                        "args": {
                            k: v
                            for k, v in s.action_input.items()
                            if k not in ("tool",)
                        },
                        "result": s.observation[:800],
                    }
                )
        response["tool_results"] = tool_results

        # Metadata for governance & audit
        response["tools_used"] = list(
            set(
                s.action_input.get("tool", "")
                for s in steps
                if s.action == "call_tool" and s.action_input.get("tool")
            )
        )
        # Check current turn AND thread history — agent may have retrieved
        # knowledge in a prior turn (e.g. before ask_user) that grounds this answer.
        all_steps = list(state.step_history)
        response["knowledge_retrieved"] = any(
            s.action == "retrieve_knowledge" for s in all_steps
        )
        response["step_count"] = len(steps)
        response["slots"] = dict(state.accumulated_slots)
        response["policies_applied"] = list(self.config.policies)

        # ---- Knowledge sources for explainability provenance ----
        def _parse_knowledge_steps(
            step_list: List[ReActStep],
        ) -> List[Dict[str, Any]]:
            ks: List[Dict[str, Any]] = []
            for s in step_list:
                if s.action == "retrieve_knowledge" and s.observation:
                    obs = s.observation
                    source_names: List[str] = []
                    passages: List[str] = []
                    if obs.startswith("Retrieved from source(s):"):
                        header, _, body = obs.partition("\n\n")
                        source_names = [
                            n.strip()
                            for n in header.replace(
                                "Retrieved from source(s):", ""
                            ).split(",")
                            if n.strip()
                        ]
                        if body:
                            passages = [
                                p.strip()[:300]
                                for p in body.split("\n---\n")
                                if p.strip()
                            ]
                    ks.append(
                        {
                            "query": s.action_input.get("query", ""),
                            "sources": source_names,
                            "passages": passages[:5],
                        }
                    )
            return ks

        # Current turn retrieval sources
        knowledge_sources = _parse_knowledge_steps(steps)
        # Fall back to prior turns if current turn had no retrieval
        if not knowledge_sources:
            prior_steps = [s for s in all_steps if s not in steps]
            knowledge_sources = _parse_knowledge_steps(prior_steps)
            # Mark these as from a prior turn so the UI can label them
            for ks in knowledge_sources:
                ks["from_prior_turn"] = True
        response["knowledge_sources"] = knowledge_sources

        # ---- Policy grounding: always show which policy drives decisions ----
        # Extract the specific policy workflow steps that are relevant to
        # the current turn's reasoning, so the UI can show exactly which
        # entries guided the agent's decision.
        if self.config.policies:
            policy_entries: List[str] = []
            if state.cached_policy_content:
                # Extract workflow step blocks from the cached policy.
                # Each step starts with a header like "step_1_...: " and
                # ends before the next step or section boundary.
                import re

                step_blocks = re.split(
                    r"\n(?=\s*step_\d+_)",
                    state.cached_policy_content,
                )
                # Find which steps the agent referenced in its thoughts
                # on this turn.
                for block in step_blocks:
                    block = block.strip()
                    if not block:
                        continue
                    # Extract step name (e.g. "step_1_collect_transaction_reference")
                    step_match = re.match(r"(step_\d+\w*)", block)
                    if not step_match:
                        continue
                    step_name = step_match.group(1)
                    # Check if any of this turn's thoughts reference this step
                    for s in steps:
                        thought_lower = (s.thought or "").lower()
                        # Match step name or step number references
                        step_num = re.search(r"step_(\d+)", step_name)
                        if step_num:
                            num = step_num.group(1)
                            if (
                                step_name.lower() in thought_lower
                                or f"step {num}" in thought_lower
                                or f"step_{num}" in thought_lower
                            ):
                                # Extract the description from the block
                                desc_match = re.search(
                                    r"description:\s*>?\s*\n?\s*(.+?)(?:\n\s*\w+:|$)",
                                    block,
                                    re.DOTALL,
                                )
                                desc = (
                                    desc_match.group(1).strip()
                                    if desc_match
                                    else block[:200]
                                )
                                policy_entries.append(f"{step_name}: {desc[:300]}")
                                break

            response["policy_sources"] = [
                {
                    "name": p,
                    "type": "workflow_policy",
                    "active_entries": policy_entries,
                }
                for p in self.config.policies
            ]

        return response
