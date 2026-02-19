# app/runtime/tools/interface.py
"""
ITool — the single contract every pluggable tool must satisfy.

Contract:
    execute(slots, context) -> dict
        slots  : current FSM slot values (read, may use for params)
        context: runtime context (thread_id, policies, etc.)
        returns: dict of slot keys/values to merge back into the FSM

All adapters (Stub, Http, Sql, ...) implement this interface so the
ToolRegistry can expose them to the workflow engine as plain callables
without the engine knowing which backend is behind each tool.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class ITool(ABC):
    """Base interface for all pluggable workflow tools."""

    @abstractmethod
    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool and return slot updates.

        Args:
            slots:   Current FSM slot values.  Tools may read any slot they need.
            context: Runtime context (thread_id, vertical, policies, …).

        Returns:
            A dict of slot key → value updates that will be merged into the FSM.
            Return an empty dict if there are no updates.
        """
        ...

    def describe(self) -> Dict[str, Any]:
        """
        Human/machine-readable description of the tool.
        Override in subclasses to provide richer schema info.
        """
        return {"name": type(self).__name__, "type": "unknown"}

    # ------------------------------------------------------------------
    # Convenience: make instances directly callable so they can be passed
    # to GenericWorkflowEngine(tools={"name": tool_instance}) unchanged.
    # ------------------------------------------------------------------
    def __call__(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return self.execute(slots, context)
