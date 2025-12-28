# app/runtime/interfaces.py
from __future__ import annotations
from typing import Protocol, Dict, Any


class IAgent(Protocol):
    """Minimal contract that all domain agents (generated or handwritten) must satisfy."""

    def load(self, spec: Dict[str, Any]) -> None:
        """Initialize the agent from its slice of the factory spec (paths, params, etc.)."""
        ...

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an inference request.
        Must return JSON-safe dict (no objects) matching the agent's declared contract.
        """
        ...

    def metadata(self) -> Dict[str, Any]:
        """Return lightweight info (id, type, version, ready flags) for /health & debugging."""
        ...
