# app/runtime/memory.py
"""
Conversation Memory Module — the "M" in PMPA (Wang et al. 2024).

Thread-scoped conversation memory with:
  - Turn storage (query, response, state, slots, policy decisions)
  - FSM state snapshots for audit trail
  - Simplified context retrieval for LLM prompt injection

Uses a dict-based in-memory backend (POC). The MemoryBackend Protocol
allows swapping to Redis/Postgres later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class MemoryTurn:
    """A single conversation turn."""

    turn_id: int
    timestamp: float
    query: str
    response: Dict[str, Any]
    agent_id: Optional[str] = None
    fsm_state: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    policy_decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """Snapshot of FSM state at a point in time."""

    timestamp: float
    state: str
    slots: Dict[str, Any]
    agent_id: str
    trigger: str  # "state_entry" | "policy_decision" | "delegation"


@runtime_checkable
class MemoryBackend(Protocol):
    """Swappable backend protocol (dict -> Redis -> Postgres)."""

    def store_turn(self, thread_id: str, turn: MemoryTurn) -> None: ...

    def get_turns(self, thread_id: str, limit: int = 20) -> List[MemoryTurn]: ...

    def store_snapshot(self, thread_id: str, snapshot: StateSnapshot) -> None: ...

    def get_snapshots(self, thread_id: str, limit: int = 10) -> List[StateSnapshot]: ...

    def get_last_turn(self, thread_id: str) -> Optional[MemoryTurn]: ...

    def clear(self, thread_id: str) -> None: ...


class DictMemoryBackend:
    """In-memory dict-based implementation (POC, matches THREAD_CTX pattern)."""

    def __init__(self) -> None:
        self._turns: Dict[str, List[MemoryTurn]] = {}
        self._snapshots: Dict[str, List[StateSnapshot]] = {}

    def store_turn(self, thread_id: str, turn: MemoryTurn) -> None:
        self._turns.setdefault(thread_id, []).append(turn)

    def get_turns(self, thread_id: str, limit: int = 20) -> List[MemoryTurn]:
        turns = self._turns.get(thread_id, [])
        return turns[-limit:] if limit else turns

    def store_snapshot(self, thread_id: str, snapshot: StateSnapshot) -> None:
        self._snapshots.setdefault(thread_id, []).append(snapshot)

    def get_snapshots(self, thread_id: str, limit: int = 10) -> List[StateSnapshot]:
        snaps = self._snapshots.get(thread_id, [])
        return snaps[-limit:] if limit else snaps

    def get_last_turn(self, thread_id: str) -> Optional[MemoryTurn]:
        turns = self._turns.get(thread_id, [])
        return turns[-1] if turns else None

    def clear(self, thread_id: str) -> None:
        self._turns.pop(thread_id, None)
        self._snapshots.pop(thread_id, None)


class ConversationMemory:
    """
    Thread-scoped conversation memory manager.

    The "M" in PMPA (Wang et al. 2024): each agent state transition
    records context, enabling multi-turn reasoning and audit trails.
    """

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._backend: MemoryBackend = backend or DictMemoryBackend()
        self._turn_counters: Dict[str, int] = {}

    def record_turn(
        self,
        thread_id: str,
        query: str,
        response: Dict[str, Any],
        agent_id: Optional[str] = None,
        fsm_state: Optional[str] = None,
        slots: Optional[Dict[str, Any]] = None,
        policy_decisions: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryTurn:
        """Record a conversation turn with full context."""
        count = self._turn_counters.get(thread_id, 0) + 1
        self._turn_counters[thread_id] = count

        turn = MemoryTurn(
            turn_id=count,
            timestamp=time.time(),
            query=query,
            response=response,
            agent_id=agent_id,
            fsm_state=fsm_state,
            slots=dict(slots) if slots else {},
            policy_decisions=list(policy_decisions) if policy_decisions else [],
            metadata=dict(metadata) if metadata else {},
        )
        self._backend.store_turn(thread_id, turn)
        return turn

    def record_state_snapshot(
        self,
        thread_id: str,
        state: str,
        slots: Dict[str, Any],
        agent_id: str,
        trigger: str = "state_entry",
    ) -> StateSnapshot:
        """Record an FSM state snapshot for PMPA audit trail."""
        snapshot = StateSnapshot(
            timestamp=time.time(),
            state=state,
            slots=dict(slots),
            agent_id=agent_id,
            trigger=trigger,
        )
        self._backend.store_snapshot(thread_id, snapshot)
        return snapshot

    def get_conversation_context(self, thread_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return turns as simple dicts for LLM context injection."""
        turns = self._backend.get_turns(thread_id, limit=limit)
        return [
            {
                "turn_id": t.turn_id,
                "query": t.query,
                "agent_id": t.agent_id,
                "fsm_state": t.fsm_state,
                "answer": (
                    t.response.get("answer")
                    or t.response.get("text")
                    or t.response.get("message")
                    or ""
                ),
            }
            for t in turns
        ]

    def get_turns(self, thread_id: str, limit: int = 20) -> List[MemoryTurn]:
        """Retrieve recent turns for a thread."""
        return self._backend.get_turns(thread_id, limit=limit)

    def get_snapshots(self, thread_id: str, limit: int = 10) -> List[StateSnapshot]:
        """Retrieve recent FSM state snapshots."""
        return self._backend.get_snapshots(thread_id, limit=limit)

    def get_last_turn(self, thread_id: str) -> Optional[MemoryTurn]:
        """Get the most recent turn for a thread."""
        return self._backend.get_last_turn(thread_id)

    def clear(self, thread_id: str) -> None:
        """Clear all memory for a thread."""
        self._turn_counters.pop(thread_id, None)
        self._backend.clear(thread_id)
