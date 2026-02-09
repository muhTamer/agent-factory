# app/runtime/audit_writer.py
from __future__ import annotations

from pathlib import Path

from app.runtime.trace import Trace


class JsonlAuditWriter:
    def __init__(self, audit_dir: str = ".factory/audit", filename: str = "runtime_traces.jsonl"):
        self.audit_dir = Path(audit_dir)
        self.path = self.audit_dir / filename
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def write(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trace.to_json() + "\n")
