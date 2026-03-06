# RQ2 Case Studies — Summary

## Overview

Seven detailed case study narratives demonstrating how explainability mechanisms and IEEE standards compliance operate across the four orchestration patterns in the Meta-Agent Factory. Each case study traces a customer query from intake through orchestration, governance, and response delivery, with full IEEE compliance assessment against 23 requirements from three standards.

**Research Question:** *"What governance mechanisms enable auditability and transparency in meta-agent decision-making, and how do they align with IEEE standards for explainable AI?"*

---

## Case Study Matrix

| # | ID | File | Pattern | Complexity | IEEE Compliance | Key Finding |
|---|-----|------|---------|------------|-----------------|-------------|
| 1 | simple_01 | [simple_01.md](simple_01.md) | Direct | Low | 91% (21/23) | Provenance enables trust; simplest pattern has highest baseline |
| 2 | simple_03 | [simple_03.md](simple_03.md) | Direct | Low | 91% (21/23) | Info-rich responses need structured comparison format |
| 3 | fsm_02 | [fsm_02.md](fsm_02.md) | FSM Workflow | Medium | 87% (20/23) | State machines produce richest audit trails; confidence gap for deterministic systems |
| 4 | deleg_01 | [deleg_01.md](deleg_01.md) | AOP Delegation | High | 96% (22/23) | Multi-agent coordination maximises compliance; synthesis remains challenging |
| 5 | hitl_01 | [hitl_01.md](hitl_01.md) | HITL Escalation | High | 100% (23/23) | Doing least achieves most compliance — knowing limits is transparent |
| 6 | deleg_04 | [deleg_04.md](deleg_04.md) | AOP Delegation | High | 96% (22/23) | Hallucination prevention is content-shaping governance; dual-layer defence |
| 7 | fsm_04 | [fsm_04.md](fsm_04.md) | FSM Workflow | Medium | 83% (19/23) | Error handling tests boundaries; structural slot guarantees prevent premature action |

---

## Compliance by Orchestration Pattern

| Pattern | Cases | Mean Compliance | Range | Best IEEE Standard |
|---------|-------|-----------------|-------|-------------------|
| Direct Routing | 2 | 91% | 91%–91% | P3394 (100%) |
| FSM Workflow | 2 | 85% | 83%–87% | P3394 (95%) |
| AOP Delegation | 2 | 96% | 96%–96% | P3394, 2894 (100%) |
| HITL Escalation | 1 | 100% | 100% | All (100%) |
| **Overall** | **7** | **92%** | **83%–100%** | |

---

## Compliance by IEEE Standard

| Standard | Requirements | Mean Compliance | Common Gaps |
|----------|-------------|-----------------|-------------|
| IEEE P3394 (Message Format) | 10 | 97% | R10 (minimal agent chains in simple workflows) |
| IEEE 2894-2024 (Explainability) | 7 | 96% | R6 (confidence scores missing for deterministic FSMs) |
| IEEE 3152-2024 (Transparency) | 6 | 83% | R4 (capabilities not discoverable), R6 (escalation not evidenced in non-HITL) |

---

## Aggregate Findings

### Common Strengths (All 7 Cases)

1. **Full provenance tracking** — Every response traces back to a specific source, agent, and timestamp
2. **Agent identity disclosure** — All responses include the responsible agent ID (IEEE 3152-R2)
3. **Multi-level explanations** — Summary, detailed, and full audit levels generated for every scenario
4. **PII-aware guardrails** — Financial figures correctly distinguished from personal data
5. **Structured message format** — IEEE P3394 envelope present in all cases

### Common Gaps (5+ Cases)

1. **Capability discovery (3152-R4)** — 4 of 7 cases fail; the system does not expose what *other* agents could handle the query
2. **Escalation evidence (3152-R6)** — 5 of 7 cases fail; escalation capability exists but is not evidenced in trace for non-HITL scenarios
3. **Confidence for deterministic systems (2894-R6)** — 2 of 7 cases fail; FSM workflows are deterministic but IEEE expects probabilistic uncertainty reporting

### Pattern-Specific Insights

| Pattern | Unique Strength | Unique Challenge |
|---------|----------------|-----------------|
| Direct | Simplicity = auditability | Limited complexity demonstration |
| FSM | Deterministic state = verifiable decisions | Hidden from user; no transition reasoning |
| AOP | Richest governance data (decompose + score + completeness) | Multi-agent explanation synthesis |
| HITL | Perfect compliance via honest limitation disclosure | No feedback loop from human resolution |

---

## Governance Mechanisms — Cross-Case Evidence

| Mechanism | Demonstrated In | Key Evidence |
|-----------|----------------|--------------|
| 1. Planning traces | deleg_01, deleg_04, hitl_01 | AOP decomposition with labelled subtasks |
| 2. Agent selection logs | All 7 cases | Solvability scores, intent-aware routing |
| 3. Compliance checkpoints | fsm_02, deleg_04, hitl_01 | Policy evaluation, hallucination detection, risk assessment |
| 4. Reasoning provenance | All 7 cases | Source citations, agent IDs, retrieval scores |
| 5. Escalation triggers | hitl_01 | Multi-factor risk score exceeding threshold |
| 6. Decision rollback | deleg_01, deleg_04, fsm_04 | Action blocking, hallucination mutation, slot enforcement |

---

## RQ2 Answer

### Part A — Governance Mechanisms

All six governance mechanisms are demonstrated across the seven case studies:

- **Planning traces:** AOP decomposition creates auditable task plans (deleg_01, deleg_04)
- **Agent selection logs:** Solvability scoring with intent-aware bonuses provides transparent delegation decisions (all cases)
- **Compliance checkpoints:** Pre/post guardrails, policy evaluation, and risk assessment enforce governance (fsm_02, deleg_04, hitl_01)
- **Reasoning provenance:** Every response is traceable to specific sources, agents, and processing steps (all cases)
- **Escalation triggers:** Multi-factor risk assessment triggers human-in-the-loop when automated handling would be unsafe (hitl_01)
- **Decision rollback:** Action blocking, response mutation, and slot enforcement prevent premature or incorrect actions (deleg_01, deleg_04, fsm_04)

### Part B — IEEE Standards Alignment

Across 7 scenarios and 23 requirements per scenario (161 total checks):

- **Overall compliance:** 92% (148/161 checks passing)
- **MUST requirements:** 96% compliance (structural requirements consistently met)
- **SHOULD requirements:** 84% compliance (gaps primarily in capability discovery and escalation evidence)
- **Best pattern:** HITL escalation (100%) — honest limitation disclosure satisfies all transparency requirements
- **Lowest pattern:** FSM error handling (83%) — incomplete workflows produce less trace data

### Key Contribution

This is the **first empirical operationalisation of IEEE P3394, 2894-2024, and 3152-2024 in a multi-agent banking system**. The case studies demonstrate that:

1. **High compliance is achievable** — 92% average across diverse orchestration patterns
2. **Gaps are addressable** — the three common gaps (capability discovery, escalation evidence, deterministic confidence) require instrumentation changes, not architectural redesign
3. **The paradox of compliance** — the scenario where the AI does the least (HITL escalation) achieves the most compliance, suggesting that IEEE standards implicitly reward systems that know their limits

---

*Generated: 2026-03-06 | Agent Factory v2.3 | 7 case studies across 4 orchestration patterns*
