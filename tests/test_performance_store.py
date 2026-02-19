# tests/test_performance_store.py
from __future__ import annotations

from app.orchestration.performance_store import ExecutionRecord, PerformanceStore


def test_append_and_query(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    rec = ExecutionRecord(
        agent_id="agent_a",
        subtask="process refund",
        success=True,
        score=0.9,
        latency_ms=120,
    )
    store.append(rec)

    results = store.query()
    assert len(results) == 1
    assert results[0].agent_id == "agent_a"
    assert results[0].success is True
    assert results[0].score == 0.9


def test_query_filter_by_agent(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    store.append(ExecutionRecord("agent_a", "task1", True, 0.8, 100))
    store.append(ExecutionRecord("agent_b", "task2", False, 0.3, 200))
    store.append(ExecutionRecord("agent_a", "task3", True, 0.9, 150))

    results_a = store.query(agent_id="agent_a")
    assert len(results_a) == 2
    assert all(r.agent_id == "agent_a" for r in results_a)

    results_b = store.query(agent_id="agent_b")
    assert len(results_b) == 1


def test_query_limit(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    for i in range(10):
        store.append(ExecutionRecord("agent_a", f"task{i}", True, 0.5, 100))

    results = store.query(limit=3)
    assert len(results) == 3


def test_query_most_recent_first(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    store.append(ExecutionRecord("agent_a", "old", True, 0.5, 100, timestamp=1000.0))
    store.append(ExecutionRecord("agent_a", "new", True, 0.9, 50, timestamp=2000.0))

    results = store.query()
    assert results[0].subtask == "new"
    assert results[1].subtask == "old"


def test_agent_success_rate(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    store.append(ExecutionRecord("agent_a", "t1", True, 0.8, 100))
    store.append(ExecutionRecord("agent_a", "t2", False, 0.2, 200))
    store.append(ExecutionRecord("agent_a", "t3", True, 0.7, 150))
    store.append(ExecutionRecord("agent_a", "t4", True, 0.9, 120))

    rate = store.agent_success_rate("agent_a")
    assert rate == 0.75  # 3/4


def test_agent_success_rate_no_history(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    rate = store.agent_success_rate("nonexistent")
    assert rate == 0.5  # neutral prior


def test_agent_avg_score(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    store.append(ExecutionRecord("agent_a", "t1", True, 0.6, 100))
    store.append(ExecutionRecord("agent_a", "t2", True, 0.8, 150))

    avg = store.agent_avg_score("agent_a")
    assert avg == 0.7  # (0.6 + 0.8) / 2


def test_agent_avg_score_no_history(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    avg = store.agent_avg_score("nonexistent")
    assert avg == 0.5  # neutral prior


def test_empty_file_returns_empty(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    results = store.query()
    assert results == []


def test_corrupted_file_returns_empty(tmp_path):
    path = tmp_path / "perf.json"
    path.write_text("not valid json", encoding="utf-8")
    store = PerformanceStore(path=str(path))
    results = store.query()
    assert results == []


def test_timestamp_auto_set(tmp_path):
    store = PerformanceStore(path=str(tmp_path / "perf.json"))
    rec = ExecutionRecord("agent_a", "task", True, 0.5, 100)
    assert rec.timestamp > 0
    store.append(rec)

    results = store.query()
    assert results[0].timestamp > 0
