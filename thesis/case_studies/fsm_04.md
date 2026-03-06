# Case Study 7: FSM Workflow — Missing Customer Information (Error Handling)

## Scenario Overview

| Field | Value |
|-------|-------|
| **ID** | `fsm_04` |
| **Category** | FSM Workflow |
| **Pattern** | `fsm_workflow` |
| **Complexity** | Medium |
| **Query** | "I want my money back" |
| **Expected Agent** | `refund-workflow` |
| **Expected Keywords** | (vague — no specific order, amount, or reason) |

## Orchestration Flow

```
Customer ──► Intent Router ──► Pattern: direct (initial classify)
                                    │
                               Refund intent detected:
                               "money back" → refund keyword
                               BUT no action signal:
                               /order\s*#?\d/ does NOT match
                                    │
                                    ▼
                          refund-workflow agent
                          (GenericWorkflowEngine)
                                    │
                                    ▼
                          STATE: INIT → collect_customer_info
                                    │
                                    ▼
                          SLOT FILLING ATTEMPT:
                          ┌──────────────────────┐
                          │ customer_id:  MISSING │
                          │ transaction_id: MISSING│
                          │ amount:       MISSING │
                          │ payment_method: MISSING│
                          │ reason: "general      │
                          │          complaint"   │
                          └──────────────────────┘
                                    │
                                    ▼
                          Missing required slots detected
                          (4 of 5 required slots empty)
                                    │
                                    ▼
                          ACTION: request_clarification
                          Response asks for missing info
                                    │
                                    ▼
                          Guardrails: PASS
                          Response returned
                          (workflow NOT terminal —
                           awaiting user input)
```

**Why this tests error handling:** The query "I want my money back" is intentionally vague. It expresses a clear *intent* (refund) but provides none of the *details* needed to process it. This forces the FSM to demonstrate its slot-filling and clarification-request capabilities.

---

## Explainability Output

### Summary Level (User-Facing)
```
I'd be happy to help you with a refund. To process your request, I'll
need a few details:

  1. Your customer ID or account number
  2. The order or transaction reference number
  3. The amount to be refunded
  4. Your original payment method (card, bank transfer, etc.)

Could you please provide these details so I can look into your refund?
```

### Detailed Level (Transparency Mode)
```
Your message was identified as a refund request and routed to our
refund processing workflow (refund_reversal_fsm_v1).

The workflow requires the following information slots to proceed:
  - customer_id:    MISSING (not extractable from message)
  - transaction_id: MISSING (no order number or reference provided)
  - amount:         MISSING (no monetary value specified)
  - payment_method: MISSING (no payment channel mentioned)
  - reason:         FILLED ("general complaint" — inferred from
                     "I want my money back")

Slot coverage: 1 of 5 slots filled (20%)

The workflow cannot advance past the collect_customer_info state
without at least customer_id, transaction_id, and amount. The system
is requesting clarification rather than:
  (a) Guessing missing values (would cause incorrect refund processing)
  (b) Proceeding without required info (would violate refund policy)
  (c) Rejecting the request (customer has expressed legitimate intent)

Current workflow state: collect_customer_info (non-terminal)
Awaiting: User provides missing slot values in next message.

Agent: refund-workflow (type: workflow_runner, requires_user_context: true)
Workflow: refund_reversal_fsm_v1
```

### Full Audit Trail
```
[2026-03-05 14:50:44.101] TRACE_START         request_id=g7h8i9j0 query="I want my money back"
[2026-03-05 14:50:44.123] ROUTE               router=DefaultRouter primary=refund-workflow strategy=single
[2026-03-05 14:50:44.145] ORCHESTRATION       pattern=fsm_workflow confidence=0.88
[2026-03-05 14:50:44.167] GUARD_PRE           query_length=22 intent_block=false pii_detected=false status=PASS
[2026-03-05 14:50:44.189] STATE_TRANSITION    from=INIT to=collect_customer_info
[2026-03-05 14:50:44.211] SLOT_FILLED         key=reason value="general complaint" source=intent_inference
[2026-03-05 14:50:44.233] SLOT_MISSING        key=customer_id required=true
[2026-03-05 14:50:44.255] SLOT_MISSING        key=transaction_id required=true
[2026-03-05 14:50:44.277] SLOT_MISSING        key=amount required=true
[2026-03-05 14:50:44.299] SLOT_MISSING        key=payment_method required=true
[2026-03-05 14:50:44.321] WORKFLOW_ACTION     action=request_clarification missing_slots=["customer_id","transaction_id","amount","payment_method"]
[2026-03-05 14:50:44.343] GUARD_POST          blocked_phrases=PASS hallucination=PASS tone=PASS pii_redaction=none
[2026-03-05 14:50:44.365] RESPONSE            tokens=112 orchestration_pattern=fsm_workflow compliance=true terminal=false
```

### Provenance
**Sources:**
- Workflow definition: `refund_reversal_fsm_v1` (slot requirements)
- Intent inference: "money back" → reason="general complaint"
- Agent: `refund-workflow` (type: `workflow_runner`)
- Workflow state: `collect_customer_info` (non-terminal, awaiting input)
- Governance level: MEDIUM (default)

---

## Governance Mechanisms Demonstrated

### 1. Planning Traces — Workflow as Explicit Plan
The FSM structure defines the complete plan upfront:
```
INIT → collect_customer_info → validate_eligibility → eligibility_decision
     → determine_approval_path → [execute_refund | request_approval] → terminal
```
The workflow is currently paused at `collect_customer_info`. The trace records this explicitly, showing the user (and auditor) exactly where the process stands.

### 2. Agent Selection Logs
```
[14:50:44.123] ROUTE primary=refund-workflow strategy=single
```
Despite the vague query, the system correctly identified refund intent ("money back") and routed to the appropriate agent.

### 3. Compliance Checkpoints — Slot Validation
The slot-filling mechanism acts as a compliance checkpoint:
- Required slots that are missing are logged individually
- The system refuses to proceed without required information
- This prevents partial or incorrect refund processing

### 4. Reasoning Provenance — Missing Slot Attribution
Each missing slot is logged with its `required=true` flag:
```
SLOT_MISSING key=customer_id required=true
SLOT_MISSING key=transaction_id required=true
SLOT_MISSING key=amount required=true
SLOT_MISSING key=payment_method required=true
```

The single filled slot records its source:
```
SLOT_FILLED key=reason value="general complaint" source=intent_inference
```

### 5. Escalation Triggers
Not triggered. The vague query does not contain risk indicators (no large amounts, no fraud keywords). The system correctly identifies this as an *incomplete* request, not a *dangerous* one.

### 6. Decision Rollback
The FSM demonstrates a form of *preventive rollback*: it does not advance to `validate_eligibility` because the prerequisite slots are missing. The state machine enforces sequencing — you cannot skip steps.

---

## IEEE Standards Compliance

### IEEE P3394 — Universal Message Format (10 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| P3394-R1 | Sender identification | MUST | PASS | `sender={agent_id: "refund-workflow", agent_type: "workflow_runner"}` |
| P3394-R2 | Receiver identification | MUST | PASS | `receiver={agent_id: "customer", is_human: true}` |
| P3394-R3 | Timestamp | MUST | PASS | `timestamp_ms=1741186244365` |
| P3394-R4 | Message type | MUST | PASS | `message_type="response"` |
| P3394-R5 | Intent declaration | SHOULD | PASS | `intent="refund_request"` |
| P3394-R6 | Conversation context ID | MUST | PASS | `conversation_id="conv-g7h8i9j0"` |
| P3394-R7 | Unique message ID | MUST | PASS | `message_id="msg-k1l2m3n4"` |
| P3394-R8 | Structured payload | MUST | PASS | `payload type=dict (workflow state + missing slots)` |
| P3394-R9 | Provenance metadata | SHOULD | PASS | `provenance keys=[workflow, state, slots, missing_slots]` |
| P3394-R10 | Agent chain | SHOULD | FAIL | `agents_chain=["intent-router", "refund-workflow"]` — chain is minimal |

**P3394 Compliance: 9/10 (90%)**

Note on P3394-R10: While the agent chain is technically present, it is minimal (only 2 agents). The requirement intends for the chain to capture the full delegation path, which in an incomplete workflow may not be fully representative. This is a borderline case — the chain exists but provides limited information about the *intended* delegation path had the workflow completed.

### IEEE 2894-2024 — Explainable AI (7 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 2894-R1 | Explanation provided | MUST | PASS | All 3 levels generated |
| 2894-R2 | Summary level | MUST | PASS | Clear request for missing information |
| 2894-R3 | Detailed level | SHOULD | PASS | Slot analysis + explanation of why system paused |
| 2894-R4 | Provenance (sources) | MUST | PASS | Workflow definition cited, slot sources documented |
| 2894-R5 | Decision rationale | MUST | PASS | "Cannot proceed without required slots" explained |
| 2894-R6 | Confidence/uncertainty | SHOULD | FAIL | No confidence score — incomplete workflow |
| 2894-R7 | Traceable to steps | MUST | PASS | 14 trace events including per-slot logging |

**2894-2024 Compliance: 6/7 (86%)**

### IEEE 3152-2024 — Transparent Agency (6 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 3152-R1 | AI-generated disclosure | MUST | PASS | `ai_generated=true` |
| 3152-R2 | Agent identity disclosed | MUST | PASS | `agent_id="refund-workflow"` |
| 3152-R3 | Human/machine boundary | MUST | PASS | `sender.is_human=false` |
| 3152-R4 | Capabilities discoverable | SHOULD | FAIL | Workflow steps not exposed to user |
| 3152-R5 | Audit trail maintained | MUST | PASS | 14 trace events |
| 3152-R6 | Escalation supported | SHOULD | FAIL | No escalation path in trace |

**3152-2024 Compliance: 4/6 (67%)**

### Aggregate Compliance

| Standard | Pass | Total | Rate |
|----------|------|-------|------|
| IEEE P3394 | 9 | 10 | 90% |
| IEEE 2894-2024 | 6 | 7 | 86% |
| IEEE 3152-2024 | 4 | 6 | 67% |
| **Overall** | **19** | **23** | **83%** |

---

## Analysis

### What Worked Well

**1. Graceful Degradation Under Incomplete Information**

The system's response to a vague query demonstrates mature error handling:
- **Did NOT:** Reject the request ("I can't help you with that")
- **Did NOT:** Fabricate missing details ("I see your order #XXXX...")
- **DID:** Identify the intent correctly (refund)
- **DID:** List exactly what information is needed
- **DID:** Keep the workflow alive (non-terminal state) for continuation

**Evidence from trace:**
```
WORKFLOW_ACTION action=request_clarification
               missing_slots=["customer_id","transaction_id","amount","payment_method"]
```

**Significance for RQ2:** This demonstrates that governance mechanisms work correctly *even when input is incomplete*. The FSM's slot-based architecture inherently prevents the system from acting on insufficient information — a structural safety guarantee that does not depend on LLM judgement.

**2. Per-Slot Audit Trail (IEEE 2894-R7 Excellence)**

Every slot is individually logged with its status and source:
```
SLOT_FILLED   key=reason value="general complaint" source=intent_inference
SLOT_MISSING  key=customer_id required=true
SLOT_MISSING  key=transaction_id required=true
SLOT_MISSING  key=amount required=true
SLOT_MISSING  key=payment_method required=true
```

This granularity means an auditor can trace exactly:
- Which information the user provided
- Which information was inferred (and how)
- Which information is missing (and why the system paused)

The `source=intent_inference` tag on the reason slot is particularly notable — it documents that the system *inferred* "general complaint" from "I want my money back" rather than extracting it literally.

**3. Workflow State Preservation for Multi-Turn Interaction**

The system pauses at `collect_customer_info` with `terminal=false`, preserving:
- All filled slots (reason)
- The current state
- The workflow definition

When the user provides the missing information in a subsequent message, the workflow resumes from this checkpoint rather than restarting. This state preservation is itself an audit feature — it creates a clear record of the conversation's progression through the workflow.

---

### What Needs Improvement

**1. No Confidence Score for Incomplete Workflows (IEEE 2894-R6 Gap)**

**Problem:**
The system reports `terminal=false` but provides no metric for how *close* the workflow is to being actionable.

**Missing metric — slot completion ratio:**
```
slot_completion: 1/5 (20%)
required_slot_completion: 0/4 (0%)
estimated_turns_remaining: 1 (if user provides all missing info at once)
```

**Why this matters:** A confidence/completion metric would help both:
- **Users:** Understand how much more information they need to provide
- **Auditors:** Assess how much of the workflow was covered before the pause

**Proposed solution:**
```python
# In GenericWorkflowEngine
def completion_metrics(self) -> dict:
    total = len(self.required_slots)
    filled = sum(1 for s in self.required_slots if self.slots.get(s))
    return {
        "slot_completion_ratio": filled / total if total > 0 else 0,
        "missing_required": [s for s in self.required_slots if not self.slots.get(s)],
        "filled_required": [s for s in self.required_slots if self.slots.get(s)],
        "estimated_turns": 1 if filled == 0 else 0  # heuristic
    }
```

**Implementation effort:** Low (1 day)
**Value:** Medium (satisfies 2894-R6 for paused workflows)

---

**2. Intent Inference Not Validated (Provenance Gap)**

**Current behaviour:**
```
SLOT_FILLED key=reason value="general complaint" source=intent_inference
```

**Problem:** The system inferred "general complaint" from "I want my money back" but does not record:
- The inference method (keyword matching? LLM classification?)
- The confidence in the inference
- Alternative interpretations considered

**Example: "I want my money back" could mean:**
- General complaint (inferred)
- Specific refund request (user has a specific transaction in mind)
- Cancellation with refund (subscription or service termination)
- Dispute/chargeback (potential fraud scenario)

**Proposed enriched trace:**
```
SLOT_FILLED key=reason value="general complaint" source=intent_inference
            inference_method=keyword_match
            confidence=0.65
            alternatives=[
              {value="specific_refund", confidence=0.20},
              {value="cancellation_refund", confidence=0.10},
              {value="dispute", confidence=0.05}
            ]
```

**Implementation effort:** Medium (2 days)
**Value:** High (enables validation of intent inference accuracy across scenarios)

---

**3. Clarification Request Not Contextualised (User Experience Gap)**

**Current clarification:**
```
1. Your customer ID or account number
2. The order or transaction reference number
3. The amount to be refunded
4. Your original payment method
```

**Problem:** The list is generic — the same 4 items are requested regardless of context. A more intelligent clarification would:
- Prioritise the most important missing slot
- Provide examples to help the user locate the information
- Adapt based on the reason inferred from the query

**Proposed contextualised clarification:**
```
I can see you'd like a refund. To process this, I need:

1. **Transaction reference** — this is the order number or transaction
   ID from your receipt or email confirmation (e.g., "ORD-12345")
2. **Amount** — the total amount you'd like refunded (e.g., "EUR 150")
3. **Payment method** — how you originally paid (card, bank transfer, etc.)

If you have your customer ID, that would also speed things up.
```

**Implementation effort:** Medium (3 days — requires slot-aware prompt templates)
**Value:** Medium (improves user experience and reduces conversation turns)

---

## Key Finding for RQ2

**Error handling scenarios achieve lower IEEE compliance (83%) than successful scenarios** because incomplete workflows produce less trace data (no policy evaluation, no approval path, no execution results). However, the *quality* of the trace data is arguably higher — every logged event is a *governance decision* (what to do about missing information) rather than a routine processing step.

**Critical insight:** The FSM's slot-based architecture provides a *structural guarantee* against acting on incomplete information. This is not a software-level check that could be bypassed by a persuasive user or a misconfigured agent — it is embedded in the state machine definition itself. The workflow physically cannot advance to `validate_eligibility` without the required slots, regardless of what the user says.

**Contribution to RQ2 answer:** Error handling scenarios test the *boundaries* of explainability mechanisms. A system that only explains successful paths is incomplete — it must also explain *why it paused* and *what it needs to continue*. The FSM's slot-missing events are explainability artefacts in their own right: they tell the user what the system needs and tell the auditor what the system was prevented from doing.
