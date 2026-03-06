# Case Study 2: FAQ — Premium Current Account Features

## Scenario Overview

| Field | Value |
|-------|-------|
| **ID** | `simple_03` |
| **Category** | Simple Routing (Direct) |
| **Pattern** | Direct |
| **Complexity** | Low |
| **Query** | "How is M&N Bank Premium CA different from a regular CA?" |
| **Expected Agent** | `faq-qa` |
| **Expected Keywords** | premium, account |

## Orchestration Flow

```
Customer ──► Intent Router ──► Pattern: direct (confidence 0.97)
                                    │
                                    ▼
                              faq-qa agent
                              (TF-IDF retrieval)
                                    │
                                    ▼
                          Pre-guardrails: PASS
                          Post-guardrails: PASS
                                    │
                                    ▼
                              Response returned
```

**Why Direct?** Comparative informational query — the user asks for a feature comparison, not an action. The classifier detects no action signals (`/order\s*#?\d|transaction\s*#?\d/` does not match) and routes to `faq-qa` for knowledge retrieval.

---

## Explainability Output

### Summary Level (User-Facing)
```
The M&N Bank Premium Current Account offers several advantages over the
regular Current Account: a higher daily transaction limit (€50,000 vs
€25,000), a dedicated relationship manager, priority customer service,
free international wire transfers (up to 5 per month), and a waived
annual maintenance fee when maintaining a minimum balance of €10,000.
```

### Detailed Level (Transparency Mode)
```
I retrieved the product comparison data from our FAQ knowledge base by
matching "Premium CA" and "regular CA" against our product catalogue
entries. The TF-IDF retrieval matched Section 2.1 (Current Account
Product Tiers) with a cosine similarity of 0.91.

Key differences identified:
1. Daily transaction limit: €50,000 (Premium) vs €25,000 (Regular)
2. Relationship manager: Dedicated (Premium) vs shared pool (Regular)
3. Customer service: Priority queue (Premium) vs standard queue (Regular)
4. International transfers: 5 free/month (Premium) vs €15/transfer (Regular)
5. Annual maintenance fee: Waived at €10,000 balance (Premium) vs €120/year (Regular)
6. Overdraft facility: Pre-approved up to €5,000 (Premium) vs application required (Regular)

Source: FAQ_DB v2.3, Section 2.1 — Current Account Product Tiers
Agent: faq-qa (type: knowledge_rag)
Retrieval confidence: 0.91

Note: This response contains multiple data points from a single FAQ section.
The information density is higher than typical single-fact FAQ queries,
which increases the risk of partial inaccuracy if the source document is
outdated. Last verified: 2026-01-15.
```

### Full Audit Trail
```
[2026-03-05 14:25:12.201] TRACE_START     request_id=b2c3d4e5 query="How is M&N Bank Premium CA different..."
[2026-03-05 14:25:12.223] ROUTE           router=DefaultRouter primary=faq-qa strategy=single candidates=[]
[2026-03-05 14:25:12.245] ORCHESTRATION   pattern=direct confidence=0.97
[2026-03-05 14:25:12.267] GUARD_PRE       query_length=58 intent_block=false pii_detected=false status=PASS
[2026-03-05 14:25:12.489] EXECUTE         agent=faq-qa retrieval=success source=FAQ_DB_v2.3 section=2.1 score=0.91
[2026-03-05 14:25:12.511] SELECT          agent_confirmed=faq-qa
[2026-03-05 14:25:12.533] GUARD_POST      blocked_phrases=PASS hallucination=PASS tone=PASS pii_redaction=none
[2026-03-05 14:25:12.555] RESPONSE        tokens=312 orchestration_pattern=direct compliance=true
```

### Provenance
**Sources:**
- FAQ Knowledge Base v2.3, Section 2.1 — Current Account Product Tiers
- Agent: `faq-qa` (type: `knowledge_rag`, AOP-eligible: true)
- Retrieval score: 0.91 (TF-IDF cosine similarity)
- Retrieved: 2026-03-05 14:25:12.489 UTC
- Governance level: MEDIUM (default)

---

## Governance Mechanisms Demonstrated

### 1. Planning Traces
Not applicable — direct routing bypasses AOP decomposition. The decision *not* to decompose is logged at the orchestration stage.

### 2. Agent Selection Logs
```
[14:25:12.223] ROUTE primary=faq-qa strategy=single candidates=[]
```
Single-agent selection. The `DefaultRouter` matched the informational intent to `faq-qa`.

### 3. Compliance Checkpoints
- **Pre-guardrails:** Query within length limit (58 chars). No blocked intents. No PII.
- **Post-guardrails:** All checks PASS. The response contains financial figures (€50,000, €25,000) but these are product features, not customer data — PII redactor correctly does not flag them.

### 4. Reasoning Provenance
Chain: `query → DefaultRouter → faq-qa → FAQ_DB v2.3 Section 2.1 → response`. The retrieval score (0.91) is slightly lower than simple_01 (0.95) because the comparative query matches more broadly across multiple product entries.

### 5. Escalation Triggers
None triggered. Informational query with no risk markers.

### 6. Decision Rollback
Not applicable.

---

## IEEE Standards Compliance

### IEEE P3394 — Universal Message Format (10 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| P3394-R1 | Sender identification | MUST | PASS | `sender={agent_id: "faq-qa", agent_type: "knowledge_rag", is_human: false}` |
| P3394-R2 | Receiver identification | MUST | PASS | `receiver={agent_id: "customer", is_human: true}` |
| P3394-R3 | Timestamp | MUST | PASS | `timestamp_ms=1741184712555` |
| P3394-R4 | Message type | MUST | PASS | `message_type="response"` |
| P3394-R5 | Intent declaration | SHOULD | PASS | `intent="informational_query"` |
| P3394-R6 | Conversation context ID | MUST | PASS | `conversation_id="conv-b2c3d4e5"` |
| P3394-R7 | Unique message ID | MUST | PASS | `message_id="msg-f6g7h8i9"` |
| P3394-R8 | Structured payload | MUST | PASS | `payload type=dict` |
| P3394-R9 | Provenance metadata | SHOULD | PASS | `provenance keys=[source, agent, score, timestamp]` |
| P3394-R10 | Agent chain | SHOULD | PASS | `agents_chain=["intent-router", "faq-qa"]` |

**P3394 Compliance: 10/10 (100%)**

### IEEE 2894-2024 — Explainable AI (7 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 2894-R1 | Explanation provided | MUST | PASS | Summary + Detailed + Full levels generated |
| 2894-R2 | Summary level | MUST | PASS | Feature comparison in plain language |
| 2894-R3 | Detailed level | SHOULD | PASS | 6-point comparison with source citation |
| 2894-R4 | Provenance (sources) | MUST | PASS | FAQ_DB v2.3, Section 2.1 cited |
| 2894-R5 | Decision rationale | MUST | PASS | Direct routing chosen for informational query |
| 2894-R6 | Confidence/uncertainty | SHOULD | PASS | TF-IDF score 0.91 reported |
| 2894-R7 | Traceable to steps | MUST | PASS | 8 trace events recorded |

**2894-2024 Compliance: 7/7 (100%)**

### IEEE 3152-2024 — Transparent Agency (6 requirements)

| Req ID | Description | Severity | Status | Evidence |
|--------|-------------|----------|--------|----------|
| 3152-R1 | AI-generated disclosure | MUST | PASS | `ai_generated=true` in envelope |
| 3152-R2 | Agent identity disclosed | MUST | PASS | `agent_id="faq-qa"` in response |
| 3152-R3 | Human/machine boundary | MUST | PASS | `sender.is_human=false` |
| 3152-R4 | Capabilities discoverable | SHOULD | FAIL | No candidates list in direct routing |
| 3152-R5 | Audit trail maintained | MUST | PASS | 8 trace events recorded |
| 3152-R6 | Escalation supported | SHOULD | FAIL | No escalation evidence in trace |

**3152-2024 Compliance: 4/6 (67%)**

### Aggregate Compliance

| Standard | Pass | Total | Rate |
|----------|------|-------|------|
| IEEE P3394 | 10 | 10 | 100% |
| IEEE 2894-2024 | 7 | 7 | 100% |
| IEEE 3152-2024 | 4 | 6 | 67% |
| **Overall** | **21** | **23** | **91%** |

---

## Analysis

### What Worked Well

**1. Multi-Fact Retrieval With Single-Source Provenance (IEEE 2894-R4)**

Unlike simple_01 which returned a single fact, this query required a *comparative* answer with 6 distinct data points. All 6 trace back to a single, citable source:

- **Source:** FAQ_DB v2.3, Section 2.1 — one document, one section
- **All facts from same retrieval event:** timestamp 14:25:12.489
- **Consistency guarantee:** Because all comparison points come from the same source, they are internally consistent (e.g., the €50,000 limit and the €10,000 minimum balance are from the same product tier definition)

**Significance for RQ2:** Information-rich responses are harder to audit because each data point could potentially come from a different source. By maintaining single-source provenance, the system simplifies the audit task — an auditor need only verify one document section to validate the entire response.

**2. Appropriate Handling of Financial Figures in Post-Guardrails**

The response contains multiple monetary values (€50,000, €25,000, €10,000, €120, €5,000, €15). The PII redactor correctly distinguishes between:
- **Product features** (not PII): €50,000 daily limit, €10,000 minimum balance
- **Customer data** (PII): account numbers, balances, transaction amounts

This selective non-redaction is important: an overly aggressive PII filter would strip the financial figures and render the comparison useless. The guardrail system demonstrates *precision* in addition to *recall*.

**3. Self-Documenting Confidence Degradation**

The retrieval confidence (0.91) is lower than simple_01 (0.95). The detailed explanation explicitly notes this:
> "The information density is higher than typical single-fact FAQ queries, which increases the risk of partial inaccuracy."

This self-awareness satisfies IEEE 2894-R6 (confidence/uncertainty) at a deeper level — the system not only reports a confidence number but contextualises why it is lower than expected.

---

### What Needs Improvement

**1. No Structured Comparison Format (Information Density Risk)**

**Problem identified:**
The response contains 6 comparison dimensions rendered as prose. For an info-rich response like this, a structured format would improve both user comprehension and machine auditability.

**Current output format:**
```
"...higher daily transaction limit (€50,000 vs €25,000), a dedicated
relationship manager, priority customer service..."
```

**Evidence of risk:** If the FAQ section is updated to add a 7th comparison dimension, the prose format may not surface it clearly. A structured format makes additions and omissions visible.

**Proposed format:**
```json
{
  "comparison": [
    {"dimension": "Daily transaction limit", "premium": "€50,000", "regular": "€25,000"},
    {"dimension": "Relationship manager", "premium": "Dedicated", "regular": "Shared pool"},
    {"dimension": "Customer service", "premium": "Priority queue", "regular": "Standard queue"}
  ],
  "source": "FAQ_DB v2.3, Section 2.1",
  "completeness": "6/6 dimensions from source"
}
```

**Implementation effort:** Medium (3 days — requires structured output parsing in faq-qa agent)
**Value:** High (enables automated completeness verification: does the response cover all dimensions in the source?)

---

**2. No Staleness Detection for Retrieved Content (IEEE 2894-R4 Enhancement)**

**Current behaviour:**
- FAQ_DB v2.3 was last updated 2026-01-15
- The system reports this date but does not evaluate whether the content might be stale
- Product features can change (rate adjustments, new tiers) without the system detecting it

**Missing mechanism:**
A staleness threshold that flags content older than N days and appends a caveat to the response.

**Example of what should happen:**
```
[14:25:12.489] EXECUTE agent=faq-qa source=FAQ_DB_v2.3 section=2.1
               last_updated=2026-01-15 days_since_update=49
               staleness_warning=true threshold=30_days
```

**Impact on explainability:** Without staleness detection, the provenance metadata creates *false confidence* — the user sees "FAQ v2.3, Section 2.1" and trusts the answer, not knowing the source might be outdated.

**Proposed solution:**
```python
# In faq-qa agent retrieval
STALENESS_THRESHOLD_DAYS = 30

def _check_staleness(self, source_metadata: dict) -> Optional[str]:
    last_updated = source_metadata.get("last_updated")
    if last_updated:
        days_old = (datetime.now() - last_updated).days
        if days_old > STALENESS_THRESHOLD_DAYS:
            return (f"Note: This information was last verified {days_old} days ago. "
                    f"Please confirm current product details with your branch.")
    return None
```

**Implementation effort:** Low (1 day)
**Value:** High (prevents confidently incorrect answers from stale data)

---

**3. Same 3152-R4 and R6 Gaps as simple_01**

The same two structural gaps (capability discovery, escalation evidence) persist. This confirms these are *systemic* issues in the direct-routing pattern, not one-off omissions. See simple_01 for proposed solutions.

---

## Key Finding for RQ2

**Information-rich FAQ responses stress-test provenance tracking.** While simple_01 proves provenance works for single-fact queries, simple_03 demonstrates that multi-fact comparative responses require additional structure (comparison format, completeness counts) to maintain the same audit quality. The 91% compliance score is identical to simple_01, but the *depth* of compliance is shallower because auditing 6 data points from prose is harder than auditing 1 data point.

**Contribution to RQ2 answer:** Compliance rates alone are insufficient to measure explainability quality. Two scenarios can have identical IEEE compliance scores (91%) while differing significantly in practical auditability. This motivates the need for *information density metrics* alongside binary pass/fail compliance checks.
