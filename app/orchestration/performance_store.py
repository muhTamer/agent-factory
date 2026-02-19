# app/orchestration/performance_store.py
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionRecord:
    """A single historical record of an agent executing a subtask."""

    agent_id: str
    subtask: str
    success: bool
    score: float  # Quality score 0.0-1.0
    latency_ms: int
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PerformanceStore:
    """
    Append-only JSON file store for agent execution history.

    Implements the feedback loop from Li et al. (2024) AOP framework:
    after each subtask execution, results are recorded so future
    solvability estimations can incorporate historical performance.

    Path default: .factory/performance_history.json
    Format: JSON array of ExecutionRecord dicts.
    """

    def __init__(self, path: str = ".factory/performance_history.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_all(self, records: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def append(self, record: ExecutionRecord) -> None:
        """Append a single execution record."""
        records = self._read_all()
        records.append(asdict(record))
        self._write_all(records)

    def query(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[ExecutionRecord]:
        """
        Read records, optionally filtered by agent_id.
        Returns most recent first, up to limit.
        """
        raw = self._read_all()
        if agent_id:
            raw = [r for r in raw if r.get("agent_id") == agent_id]
        raw.sort(key=lambda r: r.get("timestamp", 0), reverse=True)
        raw = raw[:limit]
        return [
            ExecutionRecord(
                agent_id=r.get("agent_id", ""),
                subtask=r.get("subtask", ""),
                success=bool(r.get("success", False)),
                score=float(r.get("score", 0.0)),
                latency_ms=int(r.get("latency_ms", 0)),
                timestamp=float(r.get("timestamp", 0.0)),
                metadata=r.get("metadata", {}),
            )
            for r in raw
        ]

    def agent_success_rate(self, agent_id: str) -> float:
        """
        Returns success rate for a given agent (0.0-1.0).
        Returns 0.5 (neutral prior) if no history exists.
        """
        records = self.query(agent_id=agent_id)
        if not records:
            return 0.5
        return sum(1 for r in records if r.success) / len(records)

    def agent_avg_score(self, agent_id: str) -> float:
        """
        Returns average quality score for a given agent.
        Returns 0.5 (neutral prior) if no history exists.
        """
        records = self.query(agent_id=agent_id)
        if not records:
            return 0.5
        return sum(r.score for r in records) / len(records)
