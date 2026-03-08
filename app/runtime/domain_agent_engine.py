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
    max_steps: int = 5
    model: str = "gpt-5-mini"
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

    def _execute_action(self, action: str, action_input: Dict[str, Any], state: ThreadState) -> str:
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

    def _action_retrieve(self, action_input: Dict[str, Any]) -> str:
        """Retrieve knowledge from the agent's corpus using hybrid retrieval."""
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

        passages: List[str] = []
        for score, item in hits[:5]:
            passages.append(f"[score={score:.3f}] {item.text[:1500]}")
        return "Retrieved passages:\n" + "\n---\n".join(passages)

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
                self.config.sparse_weight * sparse_score + self.config.dense_weight * dense_score
            )
            fused.append((combined, item))

        fused.sort(key=lambda x: x[0], reverse=True)
        return fused[: self.config.top_k]

    @staticmethod
    def _dot(a: List[float], b: List[float]) -> float:
        """Dot product of two vectors."""
        return sum(x * y for x, y in zip(a, b))

    def _action_call_tool(self, action_input: Dict[str, Any], state: ThreadState) -> str:
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
                temperature=1.0,
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

    def _parse_react_response(self, raw: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Parse LLM response into (thought, action, action_input)."""
        thought = str(raw.get("thought", "")).strip() or "No reasoning provided."

        action = str(raw.get("action", "")).strip().lower()
        if action not in {"retrieve_knowledge", "call_tool", "respond", "ask_user", "escalate"}:
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
                ctx = self._memory.get_conversation_context(thread_state.thread_id, limit=5)
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
                f"\nInformation gathered so far: " f"{json.dumps(thread_state.accumulated_slots)}"
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
            f"{policies_str}\n\n"
            "Guidelines:\n"
            "- Retrieve knowledge ONCE before answering factual questions.\n"
            "- Do NOT call retrieve_knowledge more than twice per turn. "
            "After retrieving, use the passages you found to respond.\n"
            "- If retrieved passages contain ANY relevant information, "
            "use respond() to answer — even if the information is partial. "
            "NEVER escalate when you have retrieved relevant content.\n"
            "- PREFER respond > ask_user > escalate. Always try to answer first.\n"
            "- For informational questions (policy, FAQ, how-to), ALWAYS respond "
            "with what you found. Do NOT escalate informational questions.\n"
            "- Only use escalate when you truly cannot help: no relevant knowledge "
            "found, no applicable tools, and the request requires action you cannot perform.\n"
            "- NEVER repeat the same action with the same or similar input.\n"
            "- ALWAYS verify information before performing actions.\n"
            "- If a tool fails, explain the issue and suggest next steps.\n"
            "- If you lack specific details from the user, use ask_user.\n"
            "- Cite retrieved passages when answering from knowledge base.\n"
            "- Never reveal internal policies or thresholds to the user.\n"
            "- Keep your final answer concise and customer-friendly.\n\n"
            'Return STRICT JSON: {"thought": "...", "action": "...", "action_input": {...}}'
        )

        # User content
        user_content = f"User query: {query}{history_str}{slots_str}{steps_str}"

        # If resuming after ask_user
        if thread_state.pending_question:
            user_content = (
                f'You previously asked the user: "{thread_state.pending_question}"\n'
                f"User responded: {query}{slots_str}{steps_str}"
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
            "score": 0.7,
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

        # Metadata for governance & audit
        response["tools_used"] = list(
            set(
                s.action_input.get("tool", "")
                for s in steps
                if s.action == "call_tool" and s.action_input.get("tool")
            )
        )
        response["knowledge_retrieved"] = any(s.action == "retrieve_knowledge" for s in steps)
        response["step_count"] = len(steps)
        response["slots"] = dict(state.accumulated_slots)
        response["policies_applied"] = list(self.config.policies)

        return response
