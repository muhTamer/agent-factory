# Governance, Explainability, and Guardrails

This document covers the IEEE standards compliance layer, multi-level explainability engine, and safety guardrails.

---

## IEEE Standards Compliance

The system implements compliance checking against three IEEE standards. Each response is evaluated and a compliance report is attached.

### IEEE P3394 — Universal Message Format (UMF)

**Purpose:** Standardizes the structure of messages exchanged between AI agents and humans.

**Requirements checked:**

| ID | Requirement | Severity | What Is Checked |
|----|-------------|----------|-----------------|
| R1 | Sender identification | Must | `sender` field in message envelope |
| R2 | Receiver identification | Must | `receiver` field in message envelope |
| R3 | Timestamp | Must | ISO 8601 timestamp present |
| R4 | Message type declaration | Must | `message_type` field (e.g., "chat_response") |
| R5 | Intent declaration | Should | `intent` field in message metadata |
| R6 | Conversation context ID | Must | `conversation_id` / `thread_id` present |
| R7 | Unique message ID | Must | `message_id` / `request_id` present |
| R8 | Structured payload | Must | JSON-structured response body |
| R9 | Provenance metadata | Should | Source attribution, agent chain |
| R10 | Agent chain | Should | List of agents involved in producing the response |

### IEEE 2894-2024 — Guide for Explainable AI

**Purpose:** Ensures AI outputs are accompanied by appropriate explanations at multiple levels.

**Requirements checked:**

| ID | Requirement | Severity | What Is Checked |
|----|-------------|----------|-----------------|
| R1 | Explanation exists | Must | At least one explanation level present |
| R2 | Summary level | Must | User-appropriate explanation exists |
| R3 | Detailed level | Should | Auditor-appropriate explanation exists |
| R4 | Provenance | Must | Data sources and citations documented |
| R5 | Decision rationale | Should | Key decisions have recorded reasoning |
| R6 | Confidence/uncertainty | Should | Scores or confidence metrics present |
| R7 | Traceability | Should | Steps traceable to specific processing stages |

### IEEE 3152-2024 — Transparent Human/Machine Agency

**Purpose:** Ensures transparency about the AI nature of the system and clear human/machine boundaries.

**Requirements checked:**

| ID | Requirement | Severity | What Is Checked |
|----|-------------|----------|-----------------|
| R1 | AI disclosure | Must | Response discloses AI-generated nature |
| R2 | Agent identity | Must | Responding agent identified by name/type |
| R3 | Human/machine boundary | Must | Clear distinction between AI and human actions |
| R4 | Capabilities disclosure | Should | Agent capabilities/limitations discoverable |
| R5 | Audit trail | Must | All actions logged for audit |
| R6 | Escalation support | Should | Human agent escalation available |

### Compliance Report

```python
@dataclass
class ComplianceReport:
    results: List[ComplianceResult]

    @property
    def compliance_rate(self) -> float       # Overall 0.0 – 1.0
    @property
    def by_standard(self) -> Dict[str, float]  # Per standard
    @property
    def by_severity(self) -> Dict[str, float]  # Per severity level
```

Each result includes:
```python
@dataclass
class ComplianceResult:
    requirement: IEEERequirement    # Standard, ID, description, severity
    compliant: bool                 # Pass/fail
    evidence: str                   # What was found
    gap: str                        # What is missing (if non-compliant)
```

---

## Multi-Level Explainability

The `ExplainabilityEngine` (`app/governance/explainability.py`) generates explanations at three levels, each serving a different audience.

### Summary Level (User-Facing)

**Audience:** End users
**Purpose:** Plain language description of what happened
**Example:**
> "Your query was handled by the Refund Specialist agent. It retrieved information from the refund policy and processed your request using 3 tools."

**Content:**
- Which agent handled the query
- What orchestration pattern was used (direct vs multi-agent)
- Number of subtasks (if AOP)
- High-level outcome

### Detailed Level (Auditor)

**Audience:** Compliance auditors, QA teams
**Purpose:** Decision details with scores and policy references
**Example:**
> "Router selected 'refunds_agent' with score 0.92 (intent: ACTIONABLE). Agent executed 4 ReAct steps: retrieve_knowledge → call_tool(lookup_payment) → call_tool(initiate_refund) → respond. Guardrails: pre-check passed, post-check passed."

**Content:**
- Routing decision with scores and reasons
- Agent assignment rationale
- Step-by-step decision log
- Guardrail interventions
- Completeness metrics (if AOP)
- Confidence scores

### Full Level (Developer)

**Audience:** Developers, system administrators
**Purpose:** Complete event log for forensic analysis
**Content:**
- Every trace event with timestamps and `delta_ms` from request start
- Raw data for each decision point
- Memory snapshots
- Tool call arguments and results
- LLM prompt/response pairs (if logged)

### How Explanations Are Generated

Explanations are extracted structurally from `Trace` events — **no additional LLM calls** are needed:

```python
class ExplainabilityEngine:
    def generate_all_levels(
        self, trace: Trace, response: Dict[str, Any]
    ) -> Dict[str, Explanation]
```

Returns:
```python
{
    "summary": Explanation(level=SUMMARY, narrative="...", ...),
    "detailed": Explanation(level=DETAILED, narrative="...", ...),
    "full": Explanation(level=FULL, narrative="...", ...),
}
```

Each `Explanation` contains:
```python
@dataclass
class Explanation:
    level: ExplanationLevel
    narrative: str                          # Human-readable text
    agents_involved: List[str]             # Agent IDs
    decisions: List[Dict[str, Any]]        # Key decisions with rationale
    provenance: List[Dict[str, Any]]       # Data sources and citations
    metrics: Dict[str, Any]                # Scores, latency, counts
```

---

## Policy Guardrails

The `PolicyGuardrails` class (`app/runtime/policy_guardrails.py`) enforces safety rules before and after agent execution.

### Pre-Guardrails (Before Agent Execution)

Applied to the user's query before it reaches any agent.

#### 1. Query Length Check
Rejects queries exceeding `max_query_chars` (default: 5000).

#### 2. Intent Blocking
Policy-based rules that block certain query intents. Configured via `PolicyPack`.

#### 3. PII Redaction
Detects and redacts:
- **Email addresses** — `user@example.com` → `[EMAIL_REDACTED]`
- **Phone numbers** — `+1-234-567-8900` → `[PHONE_REDACTED]`
- **Credit card numbers** — `4111-1111-1111-1111` → `[CARD_REDACTED]`

The redacted query is passed to the agent; the original is not stored.

### Post-Guardrails (Before Response Delivery)

Applied to the agent's response before the user sees it.

#### 1. Blocked Phrase Enforcement
Rejects responses containing internal jargon or policy-specific terms that should not be customer-facing.

#### 2. Hallucination Detection

Detects false action claims using pattern matching:

```regex
refund (has been|was|is) (initiated|processed|approved|completed)
refund_id
successfully refunded
```

**Blocking conditions:**
- Pattern matches AND no transaction context (no order ID, amount, or payment reference in the original query)
- Exception: Multi-turn workflows with accumulated slots are legitimate (e.g., user confirmed "Yes, proceed" after the agent collected payment details)

**Pass-through conditions:**
- `knowledge_retrieved: true` with no tools used → informational response (can't hallucinate actions)
- `needs_input: true` or `domain_agent_clarification: true` → agent is asking questions, not claiming actions

#### 3. Tone Control

Strips patterns that expose internal system behavior:

| Category | Patterns Stripped |
|----------|-------------------|
| Urgency questions | "Is this urgent?", "Would you consider this urgent?" |
| Async promises | "I've forwarded your case", "I've escalated this" |
| Follow-up promises | "I will get back to you", "I'll follow up", "I'll notify you" |
| Internal jargon | "workflow", "FSM", "slots", "router", "guardrail", "pipeline" |
| File references | Any `*.csv`, `*.yaml`, `*.json`, `*.txt`, `*.md` references |

### Configurable Guardrail Rules

Guardrail rules are data-driven and can be toggled at runtime without restarting the system.

#### GuardrailRule Dataclass

```python
@dataclass
class GuardrailRule:
    id: str            # Unique identifier (e.g., "hallucination_refund")
    label: str         # Display name (e.g., "Hallucinated Refund Action")
    description: str   # What this rule checks
    category: str      # "safety" | "tone" | "internal" | "privacy"
    severity: str      # "high" | "medium" | "low"
    enabled: bool      # Active flag — toggled at runtime
    patterns: List[str] # Regex patterns for detection
```

Rules are stored in `PolicyPack.guardrail_rules` and persisted to `spec/base_policy_pack.yaml`. Each rule maps to one or more regex patterns used by pre- or post-guardrails.

#### Admin API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/guardrails` | List all rules with current enabled state, policy pack name, version |
| `PATCH` | `/guardrails/{rule_id}` | Toggle a rule on/off (`{"enabled": true/false}`) |

Toggling a rule:
1. Updates `PolicyPack.guardrail_rules[rule_id].enabled` in memory
2. Persists the change to `spec/base_policy_pack.yaml` on disk
3. Calls `_rebuild_guardrails()` to hot-swap `spine.guardrails` — no restart needed
4. Takes effect on the next request immediately

#### Frontend Admin Panel

The `GuardrailsAdminPanel` component (see [Frontend docs](frontend.md#guardrailsadminpanel)) provides a UI for toggling rules. It appears in the Safety & Compliance section of the explainability sidebar.

### Governance-Level Guardrails

`GovernanceGuardrails` (`app/runtime/governance_guardrails.py`) wraps `PolicyGuardrails` with governance-level-aware feature toggles for RQ3 trade-off evaluation:

| Governance Level | Pre-Guardrails | Post-Guardrails | Explainability |
|-----------------|----------------|-----------------|----------------|
| Minimal | Basic (length only) | Basic (blocked phrases) | Summary only |
| Standard | Full (PII + intent) | Full (hallucination + tone) | Summary + Detailed |
| Strict | Full + enhanced | Full + enhanced | All three levels |

---

## Audit Trail

### Trace System

Every request creates a `Trace` object that records events through the pipeline:

```python
class Trace:
    request_id: str
    start_ts: float
    events: List[TraceEvent]

    def add(stage: str, data: Dict) -> None
    def elapsed_ms() -> int
```

**Trace events recorded:**
- `route` — Router decision, candidates, scores
- `guard_pre` — Pre-guardrail result (allowed/blocked, mutations)
- `execute` — Agent execution start/end
- `react_step` — Each ReAct iteration (thought, action, observation)
- `tool_call` — Tool execution with args and result
- `guard_post` — Post-guardrail result
- `voice` — Voice rendering
- `governance` — Compliance check results

### JSONL Audit Writer

All traces are written to `.factory/audit/runtime_traces.jsonl` as newline-delimited JSON:

```json
{"request_id": "abc-123", "timestamp": "2024-01-15T10:30:00Z", "query": "...", "agent_id": "refunds_agent", "trace": [...], "response": {...}, "governance": {...}}
```

### Message Envelope (UMF)

Each response is wrapped in a UMF envelope per IEEE P3394:

```json
{
  "message_id": "uuid",
  "timestamp": "2024-01-15T10:30:00Z",
  "sender": {
    "type": "ai_agent",
    "id": "refunds_agent",
    "system": "agent-factory"
  },
  "receiver": {
    "type": "human",
    "id": "user"
  },
  "message_type": "chat_response",
  "conversation_id": "thread-abc",
  "provenance": {
    "knowledge_sources": ["refunds_policy.yaml"],
    "tools_used": ["lookup_payment", "initiate_refund"]
  },
  "agents_chain": ["router", "refunds_agent", "voice"],
  "ai_generated": true
}
```

---

## Frontend Governance Visualization

The frontend renders governance data through several panels:

## Evaluation Methodology

### RQ2 — Explainability Quality (LLM-as-Judge)

The RQ2 evaluation (`evaluation/rq2_harness.py`) assesses explanation quality using two complementary approaches:

**Structural checks (deterministic):**
- IEEE compliance rate across P3394, 2894-2024, and 3152-2024 standards
- Explainability coverage (presence of summary, detailed, and full explanations)
- Provenance and agent identity disclosure

**LLM-as-judge (non-deterministic):**

The `RQ2ExplanationJudge` (`evaluation/rq2_judge.py`) evaluates each explanation level against the execution trace on three dimensions:

| Dimension | What It Measures | Scale |
|-----------|-----------------|-------|
| **Faithfulness** | Does the explanation accurately describe what the trace shows happened? | 1–5 |
| **Completeness** | Does the explanation cover all significant routing, agent, and guardrail decisions? | 1–5 |
| **Clarity** | Is the explanation appropriate for its intended audience level? | 1–5 |

The judge receives the original query, execution trace (ground truth), raw system response, and generated explanation. It uses `gpt-5-mini` via `app.llm_client.chat_json` with temperature=1.0 (Azure minimum). One judge call is made per explanation level per scenario (3 calls × 30 scenarios = 90 judge evaluations per run).

This addresses the circularity problem in pure structural checks: IEEE compliance verifies that required fields *exist*, but the judge evaluates whether the explanation content is *faithful to what actually happened*, *complete in covering decisions*, and *clear for its audience*.

**CLI usage:**
```bash
python -m evaluation.rq2_harness                    # With judge (default)
python -m evaluation.rq2_harness --no-judge          # Structural checks only
```

### RQ3 — Governance Trade-Off Evaluation

The RQ3 evaluation (`evaluation/run_governance_comparison.py`) measures the trade-off between governance strictness and task completion by running 28 scenarios at three governance levels:

| Governance Level | Pre-Guardrails | Post-Guardrails | Explainability |
|-----------------|----------------|-----------------|----------------|
| LOW (minimal) | Disabled (no checks) | Disabled (no blocked phrases, hallucination, or tone) | Summary only |
| MEDIUM (standard) | Full (query length ≤4000 + intent blocking) | Full (hallucination + tone + blocked phrases) | Summary + Detailed |
| HIGH (strict) | Strict (query length ≤2000 + intent blocking) | Full (same as MEDIUM) | All three levels |

**Scenario categories (28 total):**

| Category | Count | Governance Sensitivity |
|----------|-------|----------------------|
| safe_request | 12 | None — should pass at all levels |
| input_validation | 5 | Deterministic — query length triggers pre-guardrail blocks |
| compliance_violation | 4 | Indeterminate — depends on whether LLM generates blocked phrases |
| hallucination_risk | 4 | Indeterminate — depends on whether LLM hallucinates actions |
| tone_violation | 3 | Indeterminate — depends on whether LLM uses jargon |

**Metrics captured per level:**
- **Task completion rate** — Fraction of scenarios where the agent produced a correct response
- **Autonomy score** — `1 − intervention_rate`; how often the system completes without governance blocking
- **Intervention rate** — Fraction of scenarios where guardrails blocked or mutated the response
- **Governance action accuracy** — Correctness of governance decisions on deterministic scenarios only (62 out of 84 total data points). Computed by `expected_governance_action()` which derives the expected block/allow decision from the scenario category and governance config — no hardcoding needed.
- **Per-category breakdown** — Task completion reported per scenario category at each level
- **Latency** — Average wall-clock time per scenario

**Separating governance correctness from LLM variance:**

The evaluation cleanly separates three layers of the trade-off:
1. **Governance action accuracy** (deterministic) — Did governance block/allow correctly? Expected ~100%. At LOW, all 28 scenarios are deterministic (all post-checks disabled). At MEDIUM/HIGH, 17 are deterministic and 11 are indeterminate (depending on LLM output).
2. **Autonomy/intervention** (deterministic) — Perfectly monotonic: stricter levels intervene more.
3. **Task completion** (non-deterministic) — Subject to LLM variance at temperature=1.0. Not expected to decline monotonically because governance post-processing (tone control, hallucination detection) can *improve* response quality.

**Multi-run averaging:** The `--runs N` flag runs the full evaluation N times and reports mean ± std dev for task completion, reducing the impact of LLM variance.

**CLI usage:**
```bash
python -m evaluation.run_governance_comparison --output evaluation/results/rq3/
python -m evaluation.run_governance_comparison --runs 3    # Average over 3 runs
python -m evaluation.run_governance_comparison --dry-run   # Quick test with 3 scenarios
```

---

### GovernancePanel

Three tabs showing compliance data:

1. **Compliance** — Overall rate, per-standard bars with clickable IEEE links
2. **Explainability** — Summary/Detailed/Full tabs with narrative, decisions, metrics
3. **Envelope** — UMF metadata (AI disclosure, sender, receiver, agent chain)

### SourcesPanel

Shows knowledge provenance:
- Policy file badges with active workflow steps
- Retrieved passages (full text, no truncation)
- "Retrieved in an earlier turn" labels for cached sources
- RAG citations with source attribution

### ReActTracePanel

Shows the complete reasoning chain:
- Color-coded action types (blue=retrieval, amber=tool, green=respond, purple=ask, red=escalate)
- Expandable thought/observation sections
- Tool call arguments and results

### PolicyCheckPanel

Simple pass/block indicator:
- Green shield + "Passed" when guardrails allowed the response
- Red shield + "Blocked" with reason when guardrails intervened
