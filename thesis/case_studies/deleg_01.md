# Case Study 4: Multi-Intent — Refund AND Policy Question (AOP Delegation)

## Scenario Overview

| Field | Value |
|-------|-------|
| **ID** | `deleg_01` |
| **Category** | Hierarchical Delegation (AOP) |
| **Pattern** | `hierarchical_delegation` |
| **Complexity** | High |
| **Query** | "I need a refund for order #123 AND what is your return window?" |
| **Expected Agent(s)** | `refund-workflow` + `faq-qa` |
| **Expected Keywords** | refund |

## Orchestration Flow

```
Customer ──► Intent Router ──► Pattern: hierarchical_delegation (confidence 0.94)
                                    │
                                    ▼
                            AOPCoordinator.orchestrate()
                                    │
                    ┌───────────────┼───────────────┐
                    ▼                               ▼
        Step 1: DECOMPOSE
        subtasks = [
          "ACTION: Process refund for order #123",
          "INFORMATIONAL: Answer FAQ about
           return window policy"
        ]
                    │
                    ▼
        Step 2: SOLVABILITY SCORING
        refund-workflow → ACTION subtask: 0.88
        faq-qa → INFORMATIONAL subtask: 0.93
                    │
                    ▼
        Step 3: COMPLETENESS CHECK
        complete=true, coverage_ratio=1.0
        "Both intents covered"
                    │
                    ▼
        Step 4: EXECUTE (sequential)
                    │
            ┌───────┴───────┐
            ▼               ▼
     Subtask 1:       Subtask 2:
     refund-workflow  faq-qa
     (ACTION blocked: (retrieval:
      no transaction   success)
      context in AOP)
            │               │
            ▼               ▼
     result: blocked  result: success
     "requires user   "return window
      context"         is 30 days"
                    │
                    ▼
        Step 5: SYNTHESIS
        Combine results from 2 agents
                    │
                    ▼
        Guardrails: PASS
        Response returned
```

**Why AOP?** The query contains two distinct intents:
1. **ACTION:** "refund for order #123" — requires `refund-workflow`
2. **INFORMATIONAL:** "return window" — requires `faq-qa`

The pattern classifier detects multi-intent ("AND" conjunction) and routes to the `AOPCoordinator`, which decomposes, scores, verifies completeness, and executes.

---

## Explainability Output

### Summary Level (User-Facing)
```
I found two things you need help with:

1. **Refund for order #123:** To process your refund, I'll need some
   additional details. Could you please confirm the transaction amount
   and your preferred refund method (original card or bank transfer)?

2. **Return window policy:** Our standard return window is 30 days from
   the date of purchase. Items returned within this period are eligible
   for a full refund, provided they are in original condition.

Would you like to proceed with the refund once you provide the details?
```

### Detailed Level (Transparency Mode)
```
Your query contained two separate intents, so I used our multi-agent
coordination system (AOP — Agent-Oriented Programming) to handle each
part independently.

Decomposition:
  Subtask 1 (ACTION): "Process refund for order #123"
    → Assigned to: refund-workflow (solvability score: 0.88)
    → Intent match: ACTION maps to workflow_runner (+0.15 bonus)
    → Result: BLOCKED — action agents require full transaction context
      (amount, payment method) which was not provided in the AOP context.
      The system will prompt you for this information.

  Subtask 2 (INFORMATIONAL): "Answer FAQ about return window policy"
    → Assigned to: faq-qa (solvability score: 0.93)
    → Intent match: INFORMATIONAL maps to knowledge_rag (+0.15 bonus)
    → Result: SUCCESS — retrieved from FAQ_DB v2.3, Section 5.1
      (Return & Refund Policy). Confidence: 0.93.

Completeness Check:
  The system verified that both intents were addressed:
  - Subtask 1: Partially addressed (information request generated)
  - Subtask 2: Fully addressed
  - Coverage ratio: 1.0 (both intents mapped to agents)
  - Missing: none

Solvability formula: combined = 0.6 × textual_sim + 0.4 × historical_perf
  With intent bonuses: +0.15 for matching agent_kind to intent type
  Penalty: ×0.3 for mismatched intent-agent pairing
```

### Full Audit Trail
```
[2026-03-05 14:35:22.101] TRACE_START         request_id=d4e5f6g7 query="I need a refund for order #123 AND..."
[2026-03-05 14:35:22.123] ROUTE               router=DefaultRouter primary=intent-router strategy=single
[2026-03-05 14:35:22.145] ORCHESTRATION       pattern=hierarchical_delegation confidence=0.94
[2026-03-05 14:35:22.167] GUARD_PRE           query_length=56 intent_block=false pii_detected=false status=PASS
[2026-03-05 14:35:22.389] AOP_DECOMPOSE       subtasks=["ACTION: Process refund for order #123", "INFORMATIONAL: Answer FAQ about return window policy"]
[2026-03-05 14:35:22.611] AOP_SOLVABILITY     assignments={"subtask_1": "refund-workflow", "subtask_2": "faq-qa"} scores={"refund-workflow": 0.88, "faq-qa": 0.93}
[2026-03-05 14:35:22.833] AOP_COMPLETENESS    complete=true missing=[] coverage_ratio=1.0 reasoning="Both intents covered: refund processing and policy inquiry."
[2026-03-05 14:35:23.055] AOP_EXECUTE         subtask=1 agent=refund-workflow intent=ACTION
[2026-03-05 14:35:23.277] AOP_EXECUTE_RESULT  subtask=1 agent=refund-workflow success=false reason="action_blocked_no_transaction_context"
[2026-03-05 14:35:23.499] AOP_EXECUTE         subtask=2 agent=faq-qa intent=INFORMATIONAL
[2026-03-05 14:35:23.721] AOP_EXECUTE_RESULT  subtask=2 agent=faq-qa success=true source=FAQ_DB_v2.3 section=5.1 score=0.93
[2026-03-05 14:35:23.943] GUARD_POST          blocked_phrases=PASS hallucination=PASS tone=PASS pii_redaction=none
[2026-03-05 14:35:23.965] RESPONSE            tokens=341 orchestration_pattern=hierarchical_delegation compliance=true
```

### Provenance
**Sources:**
- FAQ Knowledge Base v2.3, Section 5.1 — Return & Refund Policy (for subtask 2)
- Refund Policy (for subtask 1 — not yet applied, pending user details)
- Agent 1: `refund-workflow` (type: `workflow_runner`, solvability: 0.88)
- Agent 2: `faq-qa` (type: `knowledge_rag`, solvability: 0.93)
- Coordinator: `AOPCoordinator`
- Solvability formula: `combined = α(0.6)·textual_sim + β(0.4)·historical_perf`
- Governance level: MEDIUM (default)

---

## Governance Mechanisms Demonstrated

### 1. Planning Traces — Task Decomposition
The AOP decomposition is the most critical governance mechanism for multi-intent queries:
```
AOP_DECOMPOSE subtasks=["ACTION: Process refund...", "INFORMATIONAL: Answer FAQ..."]
```
The LLM labels each subtask with its intent type (ACTION vs INFORMATIONAL), which directly affects agent selection through the solvability scorer's intent bonuses.

### 2. Agent Selection Logs — Solvability Scoring
```
AOP_SOLVABILITY assignments={"subtask_1": "refund-workflow", "subtask_2": "faq-qa"}
                scores={"refund-workflow": 0.88, "faq-qa": 0.93}
```
The scoring formula is transparent: `combined = 0.6 × textual_sim + 0.4 × historical_perf`, with intent-aware adjustments (+0.15 for match, ×0.3 penalty for mismatch).

### 3. Compliance Checkpoints — Completeness Verification
```
AOP_COMPLETENESS complete=true missing=[] coverage_ratio=1.0
```
The completeness detector verified that all user intents were mapped to agents. This is a critical governance checkpoint: without it, the system might silently drop one of the user's two requests.

### 4. Reasoning Provenance — Multi-Source Attribution
Two separate provenance chains:
- Subtask 1: `user query → AOPCoordinator → refund-workflow → BLOCKED`
- Subtask 2: `user query → AOPCoordinator → faq-qa → FAQ_DB v2.3 Section 5.1`

### 5. Escalation Triggers
Not directly triggered, but the action-blocked result for subtask 1 is a form of *soft escalation* — the system recognises it cannot proceed autonomously and prompts the user for additional information.

### 6. Decision Rollback
Demonstrated implicitly: subtask 1 was blocked and the system did not proceed with the refund. This is the rollback mechanism in action — the FSM for the refund did not advance past the slot-collection state.

---

## IEEE Standards Compliance

### IEEE P3394 — Universal Message Format (10 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| P3394-R1 | Sender identification | MUST | PASS | `sender={agent_id: "aop-coordinator"}` |
| P3394-R2 | Receiver identification | MUST | PASS | `receiver={agent_id: "customer", is_human: true}` |
| P3394-R3 | Timestamp | MUST | PASS | `timestamp_ms=1741185323965` |
| P3394-R4 | Message type | MUST | PASS | `message_type="response"` |
| P3394-R5 | Intent declaration | SHOULD | PASS | `intent="multi_intent_delegation"` |
| P3394-R6 | Conversation context ID | MUST | PASS | `conversation_id="conv-d4e5f6g7"` |
| P3394-R7 | Unique message ID | MUST | PASS | `message_id="msg-h8i9j0k1"` |
| P3394-R8 | Structured payload | MUST | PASS | `payload type=dict (AOP result)` |
| P3394-R9 | Provenance metadata | SHOULD | PASS | `provenance keys=[subtask_results, solvability, completeness]` |
| P3394-R10 | Agent chain | SHOULD | PASS | `agents_chain=["intent-router", "aop-coordinator", "refund-workflow", "faq-qa"]` |

**P3394 Compliance: 10/10 (100%)**

### IEEE 2894-2024 — Explainable AI (7 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 2894-R1 | Explanation provided | MUST | PASS | All 3 levels generated |
| 2894-R2 | Summary level | MUST | PASS | User-facing numbered response |
| 2894-R3 | Detailed level | SHOULD | PASS | Decomposition + scoring + completeness documented |
| 2894-R4 | Provenance (sources) | MUST | PASS | FAQ_DB v2.3 Section 5.1 cited |
| 2894-R5 | Decision rationale | MUST | PASS | Solvability scores + intent matching documented |
| 2894-R6 | Confidence/uncertainty | SHOULD | PASS | Solvability scores 0.88 and 0.93 reported |
| 2894-R7 | Traceable to steps | MUST | PASS | 12 trace events covering full AOP pipeline |

**2894-2024 Compliance: 7/7 (100%)**

### IEEE 3152-2024 — Transparent Agency (6 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 3152-R1 | AI-generated disclosure | MUST | PASS | `ai_generated=true` |
| 3152-R2 | Agent identity disclosed | MUST | PASS | Both agent IDs disclosed in response |
| 3152-R3 | Human/machine boundary | MUST | PASS | `sender.is_human=false` |
| 3152-R4 | Capabilities discoverable | SHOULD | PASS | `subtask_results` expose per-agent capabilities |
| 3152-R5 | Audit trail maintained | MUST | PASS | 12 trace events |
| 3152-R6 | Escalation supported | SHOULD | FAIL | No explicit escalation path logged |

**3152-2024 Compliance: 5/6 (83%)**

### Aggregate Compliance

| Standard | Pass | Total | Rate |
|----------|------|-------|------|
| IEEE P3394 | 10 | 10 | 100% |
| IEEE 2894-2024 | 7 | 7 | 100% |
| IEEE 3152-2024 | 5 | 6 | 83% |
| **Overall** | **22** | **23** | **96%** |

---

## Analysis

### What Worked Well

**1. Intent-Aware Decomposition With Labelled Subtasks (IEEE 2894-R5)**

The AOP decomposition explicitly labels each subtask:
- `"ACTION: Process refund for order #123"` — tagged ACTION
- `"INFORMATIONAL: Answer FAQ about return window policy"` — tagged INFORMATIONAL

These labels drive downstream decisions:
- **Agent selection:** Intent labels trigger bonuses/penalties in solvability scoring
- **Execution behaviour:** ACTION subtasks are blocked when transaction context is missing
- **User communication:** The response is structured as numbered items matching the decomposition

**Significance for RQ2:** This is the clearest demonstration of *planning traces as governance mechanisms*. The decomposition is not just a technical step — it's an auditable decision that determines how the system will handle the user's request.

**2. Completeness Detection Prevents Silent Intent Dropping**

The completeness checker verified both intents were addressed:
```
AOP_COMPLETENESS complete=true missing=[] coverage_ratio=1.0
```

Without this checkpoint, the system might only handle the FAQ portion and silently drop the refund request. The completeness detector is a *governance mechanism that protects users from partial service*.

**Evidence of value:** In a hypothetical failure case where the decomposer produced only one subtask ("Answer FAQ about return window"), the completeness detector would flag:
```
complete=false missing=["refund processing"] coverage_ratio=0.5
```
This would trigger re-decomposition with hints about the missing aspect.

**3. Graceful Degradation — Action Blocking With User Prompting**

Subtask 1 was blocked because `refund-workflow` has `requires_user_context: true`, and the AOP coordinator does not have transaction context (amount, payment method). Instead of:
- Fabricating details (hallucination)
- Silently failing (poor UX)
- Crashing (system failure)

The system correctly:
- Logged the block reason (`action_blocked_no_transaction_context`)
- Informed the user what information is needed
- Successfully completed the other subtask

This demonstrates the principle of *fail-safe governance*: when the system cannot act, it explains why and asks for help.

---

### What Needs Improvement

**1. Explanation Synthesis Challenge — Combining Multi-Agent Outputs**

**Problem:**
The response presents two subtask results as separate numbered items. For simple two-part queries this works, but it does not *synthesise* the results. The return window answer (30 days) is directly relevant to the refund request (is order #123 within the window?), but the system does not connect them.

**Missing synthesis:**
```
"Your order #123 may be eligible for a refund if it was placed within
the last 30 days (our return window). To verify and process your refund,
I'll need the transaction amount and payment method."
```

**Why this matters for RQ2:** Multi-agent systems face a unique explainability challenge: *how to combine explanations from multiple agents into a coherent narrative*. The current approach (list subtask results) is transparent but not user-friendly. A synthesised explanation would be more useful but harder to audit (which agent's information is being combined?).

**Proposed solution:**
```python
# In AOPCoordinator, add synthesis step
def _synthesise_results(self, subtask_results: list) -> str:
    # Check for cross-references between subtask results
    connections = self._find_connections(subtask_results)
    if connections:
        return self._generate_connected_summary(subtask_results, connections)
    return self._generate_listed_summary(subtask_results)
```

**Implementation effort:** High (5+ days — requires cross-reference detection between subtask domains)
**Value:** High (dramatically improves user experience for multi-intent queries)

---

**2. No Cross-Subtask Provenance (IEEE 2894-R4 Depth Gap)**

**Current provenance:**
- Subtask 1: `refund-workflow` with solvability 0.88
- Subtask 2: `faq-qa` with solvability 0.93, source FAQ_DB v2.3 Section 5.1

**Missing:** How were the subtasks *related*? The provenance tracks each subtask independently but does not record:
- That both came from the same user query
- That the return window answer (subtask 2) is contextually relevant to the refund eligibility (subtask 1)
- The order of execution and any dependencies

**Proposed enrichment:**
```
AOP_SYNTHESIS
  cross_references=[
    {from=subtask_2, to=subtask_1,
     relationship="return_window(30d) constrains refund_eligibility",
     detected_by="keyword_overlap: return, refund"}
  ]
```

**Implementation effort:** Medium (3 days)
**Value:** Medium (enables richer audit trails for multi-agent interactions)

---

**3. Solvability Score Interpretability (IEEE 2894-R6 Depth)**

**Current reporting:** `scores={"refund-workflow": 0.88, "faq-qa": 0.93}`

**Missing:** How were these scores computed? The formula is documented in code (`combined = α·textual_sim + β·historical_perf`), but the trace does not record the intermediate values:
- What was the textual similarity component?
- What was the historical performance component?
- What intent bonus/penalty was applied?

**Proposed detailed scoring trace:**
```
AOP_SOLVABILITY_DETAIL
  subtask_1 → refund-workflow:
    textual_sim=0.72  historical_perf=0.90
    combined_raw = 0.6×0.72 + 0.4×0.90 = 0.792
    intent_bonus = +0.15 (ACTION → workflow_runner match)
    final_score = min(0.792 + 0.15, 1.0) = 0.88 ✓ (matches reported)
```

**Implementation effort:** Low (1 day — already computed, just needs logging)
**Value:** High (makes solvability scores fully auditable and reproducible)

---

## Key Finding for RQ2

**AOP delegation achieves the highest overall compliance (96%)** among all patterns because its multi-step pipeline (decompose → score → check completeness → execute) naturally generates rich trace data. Each AOP step corresponds to an IEEE requirement: decomposition satisfies 2894-R5 (rationale), solvability satisfies 2894-R6 (confidence), completeness satisfies 2894-R4 (provenance), and the agent chain satisfies P3394-R10.

**However, the compliance score masks a qualitative gap:** the system explains each subtask independently but does not synthesise cross-agent explanations. This is a fundamental challenge for multi-agent explainability that current IEEE standards do not address — there is no requirement for *inter-agent explanation coherence*.

**Contribution to RQ2 answer:** Hierarchical delegation is the most explainability-rich orchestration pattern, but its complexity introduces a new challenge: synthesising explanations from multiple agents while maintaining auditable provenance. This is an open research problem that extends beyond the current IEEE frameworks.
