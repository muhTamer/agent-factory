# Evaluation Framework

The evaluation system measures the Meta-Agent Factory against two baselines (AutoGen, LangGraph) across 60 ground-truth scenarios spanning five categories. It supports the four thesis research questions (RQ1–RQ4).

---

## Scenario Design

### Ground Truth (`evaluation/scenarios/ground_truth.json`)

60 scenarios across five categories:

| Category                 | N  | Description |
|--------------------------|----|-------------|
| `informational_routing`  | 14 | Simple FAQ / knowledge retrieval queries |
| `actionable_routing`     | 10 | Tool-calling workflows (refunds, lookups) |
| `hierarchical_delegation`| 24 | Multi-intent queries requiring AOP decomposition |
| `hitl_escalation`        | 6  | Human-in-the-loop escalation triggers |
| `graceful_degradation`   | 6  | Out-of-scope / unsolvable query handling |

Each scenario specifies:
```json
{
  "id": "b77_refund_01",
  "category": "actionable_routing",
  "description": "Customer requests refund for order TXN-12345",
  "query": "I want a refund for order TXN-12345",
  "expected_pattern": "single_agent",
  "expected_agent": "agent_refunds",
  "expected_tools": ["lookup_payment", "initiate_refund"],
  "answer_keywords": ["refund", "processed"],
  "solvable": true
}
```

### Metrics

| Metric | Definition |
|--------|------------|
| **Orchestration Accuracy** | Correct pattern selected (single_agent, sequential, parallel, hierarchical, HITL) |
| **Agent Accuracy** | Correct agent(s) invoked for the query |
| **Outcome Accuracy** | Answer contains expected keywords and matches expected behavior |
| **Soft Pass** | Routing correct but LLM non-deterministically skipped tool calls |
| **Latency** | Wall-clock time per scenario (ms) |

---

## RQ1 — Framework Comparison

### Meta-Agent Factory (main harness)
```powershell
python -m evaluation.harness
```
Runs all 60 scenarios through the RuntimeSpine pipeline.

### AutoGen Baseline
```powershell
python -m evaluation.autogen_baseline
```
Builds a `SelectorGroupChat` with equivalent domain agents and runs the same 60 scenarios. Note: AutoGen sends queries to ALL agents simultaneously, causing ~3x API calls per turn.

### LangGraph Baseline
```powershell
python -m evaluation.langgraph_baseline
```
Builds a LangGraph Supervisor graph with the same agent set.

### Retry Timeout Failures
Azure S0 tier rate limits cause timeouts during baseline runs. The unified retry script re-runs failed scenarios with extended timeouts and progressive delay:

```powershell
# Retry AutoGen timeouts
python -m evaluation.retry_eval_timeouts --framework autogen --delay 20 --timeout 600

# Retry LangGraph timeouts
python -m evaluation.retry_eval_timeouts --framework langgraph --delay 20 --timeout 600
```

Options:
- `--delay` — Seconds between retries (default: 20, increases by 5 each round)
- `--timeout` — Per-scenario timeout in seconds (default: 600)
- `--max-rounds` — Maximum retry rounds (default: 10)

The retry system automatically:
- Backs up original results to `*_pre_retry.json` before first modification
- Identifies timeout failures by error messages, empty agent fields, or latency ≥ 290s
- Merges retry results into the original result set
- Recomputes summary metrics after each round
- Loops until zero timeout failures remain (up to max rounds)

---

## RQ2 — Pattern Effectiveness

```powershell
python -m evaluation.rq2_harness
```

Evaluates pattern selection accuracy across the five orchestration patterns:
- `single_agent` — Direct routing to one specialist
- `sequential_handoff` — Multi-step pipeline across agents
- `parallel_fan_out` — Concurrent execution with result aggregation
- `hierarchical_delegation` — AOP decomposition into subtasks
- `human_in_the_loop` — Escalation to human operator

Also measures router confidence calibration and AOP decomposition quality.

---

## RQ3 — Governance Trade-offs

```powershell
python -m evaluation.run_governance_comparison
```

Runs all 60 scenarios at three governance levels (LOW, MEDIUM, HIGH) and measures the accuracy–safety–latency trade-off:
- **LOW** — No guardrails active
- **MEDIUM** — PII redaction + hallucination check
- **HIGH** — Full guardrail suite (PII, intent blocking, hallucination, tone, citation)

---

## Solvability Comparison

```powershell
python -m evaluation.run_solvability_comparison
python -m evaluation.run_solvability_comparison --detailed   # per-scenario output
python -m evaluation.run_solvability_comparison --dry-run    # 5 scenarios only
```

Three-way comparison of TF-IDF, Neural (MiniLM + MLP), and LLM (gpt-5-mini) solvability estimators on 45 ground-truth subtask scenarios. Measures classification accuracy, per-category breakdown, McNemar's paired significance tests, and latency.

**Latest results (v4, 2026-04-11):**

| Estimator | Accuracy | Standard (n=21) | Lexical Gap (n=24) | Latency |
|-----------|:--------:|:---------------:|:------------------:|:-------:|
| TF-IDF | 73.3% (33/45) | **90.5%** | 58.3% | **0.9 ms** |
| Neural | 75.6% (34/45) | 76.2% | 75.0% | 145.8 ms |
| LLM | **86.7% (39/45)** | 76.2% | **95.8%** | 20,459 ms |

McNemar's tests: No pairwise comparison reaches statistical significance at p<0.05 (TF-IDF vs Neural p=1.0, LLM vs TF-IDF p=0.24, LLM vs Neural p=0.33).

---

## Results

Results are saved to `evaluation/results/`:

```
evaluation/results/
├── rq1/                              # Meta-Agent Factory results
├── autogen_baseline/                 # AutoGen results + *_pre_retry.json backups
├── langgraph_baseline/               # LangGraph results + *_pre_retry.json backups
├── rq2/                              # Pattern effectiveness results
├── rq3/                              # Governance comparison results
└── solvability/                      # TF-IDF vs Neural comparison
```

Each directory contains:
- `*_results.json` — Per-scenario detailed results
- `*_summary.json` — Aggregated metrics (overall + by category)

### Smoke Test

Quick connectivity check before running full evaluations:
```powershell
python -m evaluation.smoke_test
```

---

## Soft-Pass Mechanism

A scenario receives `soft_pass=True` when:
1. The router selects the correct orchestration pattern
2. The correct agent is invoked
3. But the LLM non-deterministically skips expected tool calls or produces slightly different output

This captures inherent LLM variance without penalizing correct orchestration decisions. Soft-passes are tracked separately from strict passes in all summary metrics.
