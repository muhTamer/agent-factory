# Case Study 3: Refund Workflow — Customer Provides Transaction Details

## Scenario Overview

| Field | Value |
|-------|-------|
| **ID** | `fsm_02` |
| **Category** | FSM Workflow |
| **Pattern** | `fsm_workflow` |
| **Complexity** | Medium |
| **Query** | "I need a refund for order number 12345, charged EUR 150 on my card" |
| **Expected Agent** | `refund-workflow` |
| **Expected Keywords** | refund |

## Orchestration Flow

```
Customer ──► Intent Router ──► Pattern: direct (initial classify)
                                    │
                               Action signal detected:
                               /order\s*#?\d/ matches "order number 12345"
                                    │
                                    ▼
                          refund-workflow agent
                          (GenericWorkflowEngine)
                                    │
                      ┌─────────────┼─────────────┐
                      ▼             ▼              ▼
              collect_customer  validate_     eligibility_
              _info             eligibility   decision
              (slot filling)    (policy check) (approve)
                                    │
                                    ▼
                          determine_approval_path
                          (€150 < €5,000 auto-approval)
                                    │
                                    ▼
                          execute_refund ──► terminal_success
                                    │
                                    ▼
                          Pre-guardrails: PASS
                          Post-guardrails: PASS
                                    │
                                    ▼
                              Response returned
```

**Why FSM?** The query contains an action signal (`order number 12345`) and financial keywords (`refund`, `EUR 150`). The `refund-workflow` agent activates the `refund_reversal_fsm_v1` state machine, which manages the multi-step refund process through defined states.

---

## Explainability Output

### Summary Level (User-Facing)
```
I've started processing your refund for order #12345 (EUR 150). Based on
our records, this transaction is within the 30-day refund window and the
amount qualifies for automatic approval. Your refund will be credited
back to your card within 5–7 business days. A confirmation reference
number will be sent to your registered email.
```

### Detailed Level (Transparency Mode)
```
Your refund request was processed through our structured refund workflow
(refund_reversal_fsm_v1). Here is how each step was evaluated:

Step 1 — Information Collection:
  Extracted from your message: order_id=12345, amount=EUR 150,
  payment_method=card, reason=order cancellation.
  Customer ID resolved from session context: CUST-002.

Step 2 — Eligibility Validation:
  Policy check performed against refund policy v3.1:
  - Transaction date: within 30-day window ✓
  - Amount: EUR 150 (within eligible range €1–€50,000) ✓
  - Payment method: card (eligible for automated reversal) ✓
  - No prior refund on this transaction ✓
  Result: ELIGIBLE (policy_reference: REFUND-POL-3.1-S2)

Step 3 — Approval Path:
  Amount EUR 150 < auto-approval threshold €5,000.
  Route: automatic approval (no manual review required).

Step 4 — Execution:
  Refund initiated via card processor. Expected credit: 5–7 business days.

Source: Refund Policy v3.1, Section 2 (Eligibility Criteria)
Agent: refund-workflow (type: workflow_runner, requires_user_context: true)
Workflow: refund_reversal_fsm_v1
```

### Full Audit Trail
```
[2026-03-05 14:30:05.101] TRACE_START        request_id=c3d4e5f6 query="I need a refund for order number 12345..."
[2026-03-05 14:30:05.123] ROUTE              router=DefaultRouter primary=refund-workflow strategy=single
[2026-03-05 14:30:05.145] ORCHESTRATION      pattern=fsm_workflow confidence=0.96
[2026-03-05 14:30:05.167] GUARD_PRE          query_length=65 intent_block=false pii_detected=false status=PASS
[2026-03-05 14:30:05.189] STATE_TRANSITION   from=INIT to=collect_customer_info
[2026-03-05 14:30:05.211] SLOT_FILLED        key=customer_id value=CUST-002 source=session_context
[2026-03-05 14:30:05.233] SLOT_FILLED        key=transaction_id value=12345 source=user_message
[2026-03-05 14:30:05.255] SLOT_FILLED        key=amount value=150 source=user_message
[2026-03-05 14:30:05.277] SLOT_FILLED        key=payment_method value=card source=user_message
[2026-03-05 14:30:05.299] SLOT_FILLED        key=reason value="order cancellation" source=user_message
[2026-03-05 14:30:05.321] STATE_TRANSITION   from=collect_customer_info to=validate_eligibility
[2026-03-05 14:30:05.543] POLICY_EVALUATED   eligible=true policy_ref=REFUND-POL-3.1-S2 reason="within_30_days"
[2026-03-05 14:30:05.565] STATE_TRANSITION   from=validate_eligibility to=eligibility_decision
[2026-03-05 14:30:05.587] STATE_TRANSITION   from=eligibility_decision to=determine_approval_path
[2026-03-05 14:30:05.609] APPROVAL_PATH      amount=150 threshold=5000 path=auto_approval
[2026-03-05 14:30:05.631] STATE_TRANSITION   from=determine_approval_path to=execute_refund
[2026-03-05 14:30:05.853] EXECUTE            agent=refund-workflow workflow=refund_reversal_fsm_v1 result=success
[2026-03-05 14:30:05.875] STATE_TRANSITION   from=execute_refund to=terminal_success terminal=true
[2026-03-05 14:30:05.897] GUARD_POST         blocked_phrases=PASS hallucination=PASS tone=PASS pii_redaction=none
[2026-03-05 14:30:05.919] RESPONSE           tokens=245 orchestration_pattern=fsm_workflow compliance=true
```

### Provenance
**Sources:**
- Refund Policy v3.1, Section 2 — Eligibility Criteria
- Customer record: CUST-002 (from session context)
- Transaction record: order #12345 (from user message)
- Agent: `refund-workflow` (type: `workflow_runner`)
- Workflow: `refund_reversal_fsm_v1`
- Policy reference: REFUND-POL-3.1-S2
- Governance level: MEDIUM (default)

---

## Governance Mechanisms Demonstrated

### 1. Planning Traces — State Machine as Explicit Plan
The FSM *is* the plan. Unlike AOP decomposition (which is dynamic), the FSM's states are pre-defined:
```
INIT → collect_customer_info → validate_eligibility → eligibility_decision
     → determine_approval_path → execute_refund → terminal_success
```
Each state transition is logged, creating a complete decision trace.

### 2. Agent Selection Logs
```
[14:30:05.123] ROUTE primary=refund-workflow strategy=single
```
The action signal (`order #12345`) triggered routing to the `refund-workflow` agent. This agent has `requires_user_context: true`, which is satisfied because the user provided transaction details.

### 3. Compliance Checkpoints
- **Pre-guardrails:** Query within length limit. No blocked intents. No PII detected (order number is a reference, not personal data).
- **Post-guardrails:** Hallucination check is critical here — the system must NOT claim "refund processed" unless the workflow actually reached `terminal_success`. The hallucination regex `/refund.*(?:initiated|processed|approved)/` passes because the workflow *did* complete successfully.

### 4. Reasoning Provenance
The FSM provides step-by-step provenance:
- **Slot provenance:** Each slot records its source (`session_context` vs `user_message`)
- **Policy provenance:** Eligibility check cites specific policy reference (REFUND-POL-3.1-S2)
- **Approval provenance:** Auto-approval threshold (€5,000) explicitly logged

### 5. Escalation Triggers
None triggered. Amount (€150) is below all escalation thresholds. If the amount had been €8,000+, the FSM would have transitioned to `request_approval` instead of auto-approving.

### 6. Decision Rollback
Not triggered in this scenario, but the FSM supports it: if `validate_eligibility` returned `eligible=false`, the state machine would transition to `terminal_ineligible` instead of proceeding.

---

## IEEE Standards Compliance

### IEEE P3394 — Universal Message Format (10 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| P3394-R1 | Sender identification | MUST | PASS | `sender={agent_id: "refund-workflow", agent_type: "workflow_runner"}` |
| P3394-R2 | Receiver identification | MUST | PASS | `receiver={agent_id: "customer", is_human: true}` |
| P3394-R3 | Timestamp | MUST | PASS | `timestamp_ms=1741185005919` |
| P3394-R4 | Message type | MUST | PASS | `message_type="response"` |
| P3394-R5 | Intent declaration | SHOULD | PASS | `intent="refund_request"` |
| P3394-R6 | Conversation context ID | MUST | PASS | `conversation_id="conv-c3d4e5f6"` |
| P3394-R7 | Unique message ID | MUST | PASS | `message_id="msg-g7h8i9j0"` |
| P3394-R8 | Structured payload | MUST | PASS | `payload type=dict (workflow result)` |
| P3394-R9 | Provenance metadata | SHOULD | PASS | `provenance keys=[policy_ref, workflow, agent, slots]` |
| P3394-R10 | Agent chain | SHOULD | PASS | `agents_chain=["intent-router", "refund-workflow"]` |

**P3394 Compliance: 10/10 (100%)**

### IEEE 2894-2024 — Explainable AI (7 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 2894-R1 | Explanation provided | MUST | PASS | Summary + Detailed + Full levels generated |
| 2894-R2 | Summary level | MUST | PASS | User-facing refund confirmation with timeline |
| 2894-R3 | Detailed level | SHOULD | PASS | 4-step workflow breakdown with policy citations |
| 2894-R4 | Provenance (sources) | MUST | PASS | Refund Policy v3.1, Section 2 cited |
| 2894-R5 | Decision rationale | MUST | PASS | Each state transition explains why it occurred |
| 2894-R6 | Confidence/uncertainty | SHOULD | FAIL | No explicit confidence score — FSM is deterministic |
| 2894-R7 | Traceable to steps | MUST | PASS | 20 trace events with state transitions |

**2894-2024 Compliance: 6/7 (86%)**

### IEEE 3152-2024 — Transparent Agency (6 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 3152-R1 | AI-generated disclosure | MUST | PASS | `ai_generated=true` |
| 3152-R2 | Agent identity disclosed | MUST | PASS | `agent_id="refund-workflow"` |
| 3152-R3 | Human/machine boundary | MUST | PASS | `sender.is_human=false` |
| 3152-R4 | Capabilities discoverable | SHOULD | FAIL | Workflow states not exposed to user |
| 3152-R5 | Audit trail maintained | MUST | PASS | 20 trace events including all state transitions |
| 3152-R6 | Escalation supported | SHOULD | FAIL | No escalation path evidenced in trace |

**3152-2024 Compliance: 4/6 (67%)**

### Aggregate Compliance

| Standard | Pass | Total | Rate |
|----------|------|-------|------|
| IEEE P3394 | 10 | 10 | 100% |
| IEEE 2894-2024 | 6 | 7 | 86% |
| IEEE 3152-2024 | 4 | 6 | 67% |
| **Overall** | **20** | **23** | **87%** |

---

## Analysis

### What Worked Well

**1. State Machine as Self-Documenting Audit Trail (IEEE 2894-R7)**

The FSM produces the richest audit trail of any orchestration pattern in the system. With 20 trace events (vs 8 for direct routing), every decision point is recorded:

- **6 state transitions** document the workflow progression
- **5 slot-fill events** trace where each input came from
- **1 policy evaluation** cites the exact policy section
- **1 approval path decision** logs the threshold comparison

**Why this matters for RQ2:** An FSM-based workflow is inherently *process-transparent*. Unlike LLM-based reasoning (which is opaque), the FSM's deterministic state transitions create a verifiable sequence of decisions. An auditor can reconstruct exactly what happened and why, without needing to interpret LLM reasoning.

**2. Slot Provenance — Input Attribution (IEEE 2894-R4)**

Each slot records its source explicitly:
```
SLOT_FILLED key=customer_id value=CUST-002 source=session_context
SLOT_FILLED key=transaction_id value=12345 source=user_message
```

This dual-source attribution (session context vs user message) is important for auditability:
- If the customer_id is wrong, the auditor knows it came from session context (system error)
- If the transaction_id is wrong, the auditor knows the user provided it (user error)

**3. Policy Enforcement Transparency (Governance Mechanism 3)**

The eligibility check cites a specific policy reference:
```
POLICY_EVALUATED eligible=true policy_ref=REFUND-POL-3.1-S2 reason="within_30_days"
```

This satisfies a core RQ2 requirement: *compliance checkpoints with verifiable policy citations*. The policy is not a black box — it's a citable document section that can be independently verified.

---

### What Needs Improvement

**1. No Confidence Score for Deterministic Workflows (IEEE 2894-R6 Gap)**

**Problem:**
The FSM is deterministic — it either succeeds or fails based on policy rules. The system does not report a confidence score because "confidence" is not meaningful for rule-based decisions.

**However, IEEE 2894-R6 still expects uncertainty information.** The gap is in *framing*: even deterministic systems have uncertainty:
- **Data uncertainty:** Is the transaction_id correct? Is the amount accurate?
- **Policy uncertainty:** Is the policy still current? Has it been updated?
- **Execution uncertainty:** Will the card processor successfully reverse the charge?

**Proposed solution — report certainty dimensions:**
```json
{
  "certainty": {
    "input_validation": 1.0,    // All required slots filled
    "policy_match": 1.0,        // Eligibility clearly met
    "policy_currency": 0.85,    // Policy last updated 49 days ago
    "execution_success": 0.95   // Card reversal success rate
  },
  "overall_certainty": 0.95,
  "note": "Deterministic workflow; uncertainty from data and execution factors"
}
```

**Implementation effort:** Medium (3 days — requires tracking policy update dates and execution success rates)
**Value:** High (converts 2894-R6 from FAIL to PASS, adds meaningful uncertainty info)

---

**2. FSM States Hidden From User (IEEE 3152-R4 Gap)**

**Current behaviour:**
The user sees "I've started processing your refund..." but has no visibility into the workflow stages. They cannot discover:
- What steps the system will take
- Where in the process they currently are
- What could go wrong at each step

**Missing capability — progress disclosure:**
```
Your request is being processed through our refund workflow:
  ✓ Step 1/4: Information collected (order #12345, EUR 150)
  ✓ Step 2/4: Eligibility verified (within 30-day policy window)
  ✓ Step 3/4: Approved (automatic — amount under €5,000)
  → Step 4/4: Executing refund (5–7 business days)
```

**Proposed solution:**
```python
# In GenericWorkflowEngine, add user-facing progress
def _progress_summary(self) -> str:
    completed = [s for s in self.history if s["terminal"] is False]
    total = len(self.workflow_def["states"]) - 1  # exclude terminal
    return f"Step {len(completed)}/{total}: {self.current_state}"
```

**Implementation effort:** Medium (2 days)
**Value:** High (transforms opaque workflow into transparent process)

---

**3. No Transition Reasoning in State Machine (Explainability Depth)**

**Current trace:**
```
STATE_TRANSITION from=eligibility_decision to=determine_approval_path
```

**Missing:** *Why* this transition and not another? The FSM's `on` events map transitions, but the trace does not record which event triggered each transition.

**Proposed enriched trace:**
```
STATE_TRANSITION from=eligibility_decision to=determine_approval_path
                 event=eligible trigger="policy_check.result==true"
                 alternatives=[{event=ineligible, target=terminal_ineligible, rejected="eligible==true"}]
```

**Implementation effort:** Low (1 day — add event data to trace)
**Value:** Medium (enables contrastive explanations for FSM decisions)

---

## Key Finding for RQ2

**FSM workflows produce the most auditable traces of any orchestration pattern** because their deterministic nature eliminates LLM reasoning opacity. However, this determinism creates a paradoxical compliance gap: IEEE 2894-R6 (confidence/uncertainty) assumes probabilistic systems and does not account for deterministic workflows. This suggests that IEEE standards may need to differentiate between *epistemic uncertainty* (what the model doesn't know) and *aleatoric uncertainty* (what the process cannot control) for mixed AI/rule-based systems.

**Contribution to RQ2 answer:** FSM-based governance achieves 87% IEEE compliance — slightly below direct routing (91%) — but produces a qualitatively richer audit trail. The compliance gap is a *standards mismatch*, not an implementation failure. This finding suggests that IEEE compliance frameworks should accommodate deterministic sub-systems within broader AI governance architectures.
