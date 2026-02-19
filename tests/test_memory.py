# tests/test_memory.py
from __future__ import annotations

from app.runtime.memory import (
    ConversationMemory,
    DictMemoryBackend,
    MemoryBackend,
)


def test_store_and_retrieve_turn():
    mem = ConversationMemory()
    turn = mem.record_turn("t1", query="Hello", response={"answer": "Hi"})
    assert turn.turn_id == 1
    assert turn.query == "Hello"
    turns = mem.get_turns("t1")
    assert len(turns) == 1
    assert turns[0].query == "Hello"


def test_turns_ordered_by_turn_id():
    mem = ConversationMemory()
    mem.record_turn("t1", query="First", response={"answer": "A"})
    mem.record_turn("t1", query="Second", response={"answer": "B"})
    mem.record_turn("t1", query="Third", response={"answer": "C"})
    turns = mem.get_turns("t1")
    assert [t.turn_id for t in turns] == [1, 2, 3]
    assert [t.query for t in turns] == ["First", "Second", "Third"]


def test_turn_limit_respected():
    mem = ConversationMemory()
    for i in range(10):
        mem.record_turn("t1", query=f"Q{i}", response={"answer": f"A{i}"})
    turns = mem.get_turns("t1", limit=3)
    assert len(turns) == 3
    # Should return the 3 most recent
    assert turns[0].turn_id == 8
    assert turns[2].turn_id == 10


def test_store_and_retrieve_snapshot():
    mem = ConversationMemory()
    snap = mem.record_state_snapshot(
        "t1", state="evaluate_policy", slots={"amount": 100}, agent_id="refund_agent"
    )
    assert snap.state == "evaluate_policy"
    snaps = mem.get_snapshots("t1")
    assert len(snaps) == 1
    assert snaps[0].slots == {"amount": 100}


def test_snapshot_limit_respected():
    mem = ConversationMemory()
    for i in range(8):
        mem.record_state_snapshot("t1", state=f"s{i}", slots={"i": i}, agent_id="agent")
    snaps = mem.get_snapshots("t1", limit=3)
    assert len(snaps) == 3
    assert snaps[0].state == "s5"


def test_get_last_turn_returns_most_recent():
    mem = ConversationMemory()
    mem.record_turn("t1", query="First", response={"answer": "A"})
    mem.record_turn("t1", query="Second", response={"answer": "B"})
    last = mem.get_last_turn("t1")
    assert last is not None
    assert last.query == "Second"
    assert last.turn_id == 2


def test_get_last_turn_empty_returns_none():
    mem = ConversationMemory()
    assert mem.get_last_turn("nonexistent") is None


def test_clear_removes_all_data():
    mem = ConversationMemory()
    mem.record_turn("t1", query="Q", response={"answer": "A"})
    mem.record_state_snapshot("t1", state="s", slots={}, agent_id="a")
    mem.clear("t1")
    assert mem.get_turns("t1") == []
    assert mem.get_snapshots("t1") == []
    assert mem.get_last_turn("t1") is None


def test_conversation_context_format():
    mem = ConversationMemory()
    mem.record_turn(
        "t1",
        query="What is the refund policy?",
        response={"answer": "30-day window"},
        agent_id="faq_agent",
        fsm_state="RESPOND",
    )
    ctx = mem.get_conversation_context("t1")
    assert len(ctx) == 1
    assert ctx[0]["query"] == "What is the refund policy?"
    assert ctx[0]["answer"] == "30-day window"
    assert ctx[0]["agent_id"] == "faq_agent"
    assert ctx[0]["fsm_state"] == "RESPOND"
    assert ctx[0]["turn_id"] == 1


def test_multiple_threads_isolated():
    mem = ConversationMemory()
    mem.record_turn("thread_a", query="A query", response={"answer": "A answer"})
    mem.record_turn("thread_b", query="B query", response={"answer": "B answer"})
    turns_a = mem.get_turns("thread_a")
    turns_b = mem.get_turns("thread_b")
    assert len(turns_a) == 1
    assert len(turns_b) == 1
    assert turns_a[0].query == "A query"
    assert turns_b[0].query == "B query"


def test_policy_decisions_stored_in_turn():
    mem = ConversationMemory()
    decisions = [
        {"type": "eligibility", "result": True, "reason": "KYC verified"},
        {"type": "amount_check", "result": True, "reason": "Under threshold"},
    ]
    turn = mem.record_turn(
        "t1",
        query="Refund me",
        response={"answer": "Done"},
        policy_decisions=decisions,
    )
    assert len(turn.policy_decisions) == 2
    assert turn.policy_decisions[0]["type"] == "eligibility"
    retrieved = mem.get_turns("t1")
    assert len(retrieved[0].policy_decisions) == 2


def test_turn_counter_increments():
    mem = ConversationMemory()
    t1 = mem.record_turn("t1", query="Q1", response={})
    t2 = mem.record_turn("t1", query="Q2", response={})
    t3 = mem.record_turn("t1", query="Q3", response={})
    assert t1.turn_id == 1
    assert t2.turn_id == 2
    assert t3.turn_id == 3


def test_memory_backend_protocol():
    backend = DictMemoryBackend()
    assert isinstance(backend, MemoryBackend)


def test_turn_with_all_fields():
    mem = ConversationMemory()
    turn = mem.record_turn(
        "t1",
        query="I need a refund",
        response={"answer": "Processing", "score": 0.9},
        agent_id="refund_agent",
        fsm_state="evaluate_policy",
        slots={"amount": 500, "currency": "EUR"},
        policy_decisions=[{"type": "eligibility", "result": True}],
        metadata={"source": "chat_api"},
    )
    assert turn.agent_id == "refund_agent"
    assert turn.fsm_state == "evaluate_policy"
    assert turn.slots == {"amount": 500, "currency": "EUR"}
    assert turn.metadata == {"source": "chat_api"}


def test_snapshot_trigger_types():
    mem = ConversationMemory()
    mem.record_state_snapshot("t1", state="s1", slots={}, agent_id="a", trigger="state_entry")
    mem.record_state_snapshot("t1", state="s2", slots={}, agent_id="a", trigger="policy_decision")
    mem.record_state_snapshot("t1", state="s3", slots={}, agent_id="a", trigger="delegation")
    snaps = mem.get_snapshots("t1")
    assert [s.trigger for s in snaps] == ["state_entry", "policy_decision", "delegation"]
