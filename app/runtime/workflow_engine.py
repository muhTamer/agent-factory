# app/runtime/workflow_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time


class WorkflowExecutionError(Exception):
    pass


@dataclass
class WorkflowResult:
    workflow_id: str
    agent_id: str
    current_state: str
    terminal: bool
    slots: Dict[str, Any]
    history: List[Dict[str, Any]]
    output: Dict[str, Any]


class GenericWorkflowEngine:
    """
    Generic FSM execution engine for workflow_runner agents.
    Designed to be driven by the generated agent wrapper.
    """

    def __init__(
        self,
        agent_id: str,
        workflow_spec: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
        logger=print,
    ):
        self.agent_id = agent_id
        self.spec = workflow_spec
        self.context = context or {}
        self.logger = logger

        # Tools can be injected from runtime later.
        self.tools = tools or {}

        self.workflow_id = str(workflow_spec["id"])
        self.description = workflow_spec.get("description", "")
        self.slot_defs = workflow_spec.get("slots", {}) or {}
        self.slots: Dict[str, Any] = {k: None for k in self.slot_defs.keys()}

        self.states = self._normalize_states(workflow_spec["states"])
        self.initial_state = workflow_spec["initial_state"]
        self.current_state = self.initial_state

        if self.initial_state not in self.states:
            raise WorkflowExecutionError(
                f"Initial state '{self.initial_state}' not found in states"
            )

        self.history: List[Dict[str, Any]] = []

        self.logger(f"[WF] init agent={self.agent_id} workflow={self.workflow_id}")
        self._enter_state(self.current_state)

    # ------------------------------------------------------------------
    # 🔄 Normalization
    # ------------------------------------------------------------------
    def _normalize_states(self, raw_states: Any) -> Dict[str, Dict[str, Any]]:
        """
        Normalize states into a canonical internal format:
        {
          state_name: {
            description: str,
            on: {event: next_state},
            actions: [str],
            terminal: bool
          }
        }
        """
        normalized: Dict[str, Dict[str, Any]] = {}

        # Dict form (LLM-friendly)
        if isinstance(raw_states, dict):
            for name, spec in raw_states.items():
                normalized[name] = self._normalize_state(name, spec)

        # Array form
        elif isinstance(raw_states, list):
            for item in raw_states:
                name = item["name"]
                normalized[name] = self._normalize_state(name, item)

        else:
            raise WorkflowExecutionError("Invalid states format")

        return normalized

    def _normalize_state(self, name: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        on: Dict[str, str] = {}

        if isinstance(spec.get("on"), dict):
            on.update({str(k): str(v) for k, v in spec["on"].items()})

        # transitions list can be either {event,target} or {event,next_state}
        if isinstance(spec.get("transitions"), list):
            for t in spec["transitions"]:
                if not isinstance(t, dict):
                    continue
                ev = t.get("event")
                tgt = t.get("target") or t.get("next_state")
                if ev and tgt:
                    on[str(ev)] = str(tgt)

        actions: List[str] = []

        # actions list
        if isinstance(spec.get("actions"), list):
            actions.extend([str(a) for a in spec["actions"] if a is not None])

        # single action
        if isinstance(spec.get("action"), str) and spec["action"].strip():
            actions.append(spec["action"].strip())

        # on_enter can be string or list
        if "on_enter" in spec:
            oe = spec["on_enter"]
            if isinstance(oe, list):
                actions.extend([str(a) for a in oe if a is not None])
            elif isinstance(oe, str) and oe.strip():
                actions.append(oe.strip())

        terminal = bool(
            spec.get("terminal") or str(spec.get("type", "")).lower() in {"final", "terminal"}
        )

        return {
            "name": name,
            "description": str(spec.get("description", "")),
            "on": on,
            "actions": actions,
            "terminal": terminal,
        }

    # ------------------------------------------------------------------
    # ▶ Execution
    # ------------------------------------------------------------------
    def _enter_state(self, state_name: str) -> None:
        state = self.states[state_name]
        self.logger(f"[WF] → enter state={state_name} terminal={state['terminal']}")

        self.history.append(
            {
                "state": state_name,
                "timestamp": time.time(),
                "actions": list(state["actions"]),
            }
        )

        for action in state["actions"]:
            self._execute_action(action)

    def _execute_action(self, action: str) -> None:
        """
        Minimal action executor:
        - call:tool_name    -> invokes injected tool
        - action:xyz        -> symbolic (no-op v1)
        - log:xyz           -> logs
        - notify:xyz        -> symbolic (no-op v1)
        - any other string  -> symbolic (no-op v1)
        """
        a = (action or "").strip()
        if not a:
            return

        self.logger(f"[WF]   action={a}")

        # Tool invocation
        if a.startswith("call:"):
            tool_name = a.split("call:", 1)[1].strip()
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger(f"[WF][WARN] tool not found: {tool_name}")
                return
            try:
                # Tool contract (v1): tool(slots, context) -> optional dict updates
                res = tool(self.slots, self.context)
                if isinstance(res, dict):
                    self.slots.update(res)
            except Exception as e:
                raise WorkflowExecutionError(f"tool '{tool_name}' failed: {e}") from e
            return

        # Logging
        if a.startswith("log:"):
            msg = a.split("log:", 1)[1].strip()
            self.logger(f"[WF][LOG] {msg}")
            return

        # Everything else is symbolic for now
        # (Later we can add LLM slot extraction, templates, etc.)
        return

    # ------------------------------------------------------------------
    # 🧠 Public API used by generated agents
    # ------------------------------------------------------------------
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic request handler invoked by generated workflow agents.

        Supported input:
        - event: optional str
        - slots: optional dict slot updates
        - query: optional str (for logging/debugging)
        - mapper: optional dict (debug/explainability from upstream mapper)

        Behavior:
        - apply slot updates
        - if event is missing/None and workflow is not terminal:
            return a request_clarification payload listing missing required slots
        - otherwise attempt transition by event
        - return structured result
        """
        if not isinstance(request, dict):
            request = {"query": str(request)}

        event = request.get("event")
        slot_updates = request.get("slots")
        query = request.get("query")

        # Apply slot updates
        if isinstance(slot_updates, dict):
            self.slots.update(slot_updates)

        # If terminal, just return state
        if self.states[self.current_state]["terminal"]:
            res = self._result()
            res["action"] = "terminal"
            return res

        # If no event was provided, we should ask for missing required info
        if event is None or (isinstance(event, str) and not event.strip()):
            missing_required = self._missing_required_slots()
            res = self._result()
            res["action"] = "request_clarification"
            res["missing_slots"] = missing_required
            res["message"] = self._clarification_message(missing_required)
            if query:
                res["query"] = query
            # Pass through mapper/debug if present (optional)
            if isinstance(request.get("mapper"), dict):
                res["mapper"] = request["mapper"]
            return res

        # Otherwise, attempt transition
        if isinstance(event, str):
            event = event.strip()

        self._handle_event(event)
        res = self._result()
        if query:
            res["query"] = query
        if isinstance(request.get("mapper"), dict):
            res["mapper"] = request["mapper"]
        return res

    def _missing_required_slots(self) -> List[str]:
        missing = []
        for slot, meta in (self.slot_defs or {}).items():
            try:
                is_required = bool(meta.get("required"))
            except Exception:
                is_required = False
            if is_required and not self.slots.get(slot):
                missing.append(slot)
        return missing

    def _clarification_message(self, missing_required: List[str]) -> str:
        if not missing_required:
            # This can happen if the workflow needs an event but doesn't require slots.
            allowed = list((self.states.get(self.current_state, {}).get("on") or {}).keys())
            if allowed:
                return "I need a bit more information to proceed. Please clarify what you want to do next."
            return "I need a bit more information to proceed."
        return "To continue, please provide: " + ", ".join(missing_required) + "."

    def _handle_event(self, event: str) -> None:
        state = self.states[self.current_state]

        if state["terminal"]:
            self.logger("[WF] already terminal; ignoring event")
            return

        if event not in state["on"]:
            self.logger(f"[WF][WARN] event '{event}' ignored in state '{self.current_state}'")
            return

        next_state = state["on"][event]
        self.current_state = next_state
        self._enter_state(next_state)

    def metadata(self) -> Dict[str, Any]:
        return {
            "id": self.agent_id,
            "type": "workflow_runner",
            "ready": True,
            "description": self.description,
            "workflow_id": self.workflow_id,
            "state": self.current_state,
            "terminal": self.states[self.current_state]["terminal"],
            "capabilities": [],  # optionally fill later from blueprint/spec
            "context": {
                "docs": len(self.context.get("docs", []) or []),
                "policies": len(self.context.get("policies", []) or []),
                "tools": len(self.context.get("tools", []) or []),
            },
        }

    def _result(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "current_state": self.current_state,
            "terminal": self.states[self.current_state]["terminal"],
            "slots": self.slots,
            "history": self.history,
            "context": self.context,
        }
