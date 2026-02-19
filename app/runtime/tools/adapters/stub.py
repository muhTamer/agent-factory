# app/runtime/tools/adapters/stub.py
"""
StubTool — wraps any plain callable in the ITool interface.

Use this to register existing stub functions (or lambdas) without
rewriting them:

    from app.runtime.tools.adapters.stub import StubTool
    from app.runtime.tools.stub_tools import _initiate_refund

    tool = StubTool("initiate_refund", _initiate_refund)
    registry.register("initiate_refund", tool)

StubTool also wraps a static response dict for even simpler cases:

    tool = StubTool.from_response("verify_identity", {"kyc_status": "verified"})
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from app.runtime.tools.interface import ITool


class StubTool(ITool):
    """Wraps a plain callable (or a static dict) as an ITool."""

    def __init__(
        self,
        name: str,
        fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        description: str = "",
    ) -> None:
        self.name = name
        self._fn = fn
        self._description = description or f"Stub implementation of '{name}'"

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_response(
        cls,
        name: str,
        response: Dict[str, Any],
        description: str = "",
    ) -> "StubTool":
        """Create a StubTool that always returns a fixed response dict."""
        return cls(name, lambda slots, ctx: dict(response), description)

    # ------------------------------------------------------------------
    # ITool
    # ------------------------------------------------------------------
    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        result = self._fn(slots, context)
        return result if isinstance(result, dict) else {}

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "stub",
            "description": self._description,
        }
