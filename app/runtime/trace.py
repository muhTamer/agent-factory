# app/runtime/trace.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TraceEvent:
    ts_ms: int
    stage: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    request_id: str
    started_ts_ms: int
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)

    @staticmethod
    def start(
        query: str, request_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None
    ) -> "Trace":
        rid = request_id or str(uuid.uuid4())
        return Trace(
            request_id=rid,
            started_ts_ms=_now_ms(),
            query=query,
            context=context or {},
            events=[],
        )

    def add(self, stage: str, **data: Any) -> None:
        self.events.append(TraceEvent(ts_ms=_now_ms(), stage=stage, data=data))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_ts_ms": self.started_ts_ms,
            "query": self.query,
            "context": self.context,
            "events": [{"ts_ms": e.ts_ms, "stage": e.stage, "data": e.data} for e in self.events],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
