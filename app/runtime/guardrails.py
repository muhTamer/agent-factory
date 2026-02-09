# app/runtime/guardrails.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class GuardResult:
    allowed: bool = True
    reason: str = ""
    mutated_query: Optional[str] = None
    mutated_context: Optional[Dict[str, Any]] = None
    mutated_response: Optional[Dict[str, Any]] = None


class Guardrails(Protocol):
    def pre(self, query: str, context: Dict[str, Any]) -> GuardResult: ...
    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult: ...


class NoOpGuardrails:
    def pre(self, query: str, context: Dict[str, Any]) -> GuardResult:
        return GuardResult(allowed=True)

    def post(self, response: Dict[str, Any], context: Dict[str, Any]) -> GuardResult:
        return GuardResult(allowed=True)
