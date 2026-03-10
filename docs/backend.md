# Backend Components Reference

Detailed reference for all backend components in `app/`.

---

## API Layer (`app/main.py`)

Minimal FastAPI application serving as the HTTP entry point.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Agent status, loaded agents, dry_run mode |
| `GET` | `/version` | API version metadata |
| `POST` | `/chat` | Process a user query through the full pipeline |

**Chat Request:**
```json
{
  "query": "I want a refund",
  "thread_id": "optional-thread-id",
  "request_id": "optional-request-id",
  "context": {}
}
```

---

## RuntimeSpine (`app/runtime/spine.py`)

The orchestration backbone implementing the **B3.5 intent-aware pipeline**. Every request flows through the same sequence of stages.

### Constructor

```python
class RuntimeSpine:
    def __init__(
        self,
        registry: AgentRegistry,
        router: Router,
        guardrails: Guardrails | None = None,
        audit_writer: JsonlAuditWriter | None = None,
        aop_coordinator: Optional[AOPCoordinator] = None,
        memory: Optional[ConversationMemory] = None,
        governance_enabled: bool = True,
    )
```

### Pipeline Stages

```
handle_chat(query, thread_id, context)
  │
  ├─ 1. Trace.start() → create request_id, timestamp
  ├─ 2. _classify_orchestration_pattern(query)
  │      → "direct" or "hierarchical_delegation"
  ├─ 3. _guard_pre(query, context)
  │      → PII redaction, intent blocking, length check
  ├─ 4. Execute:
  │      ├─ Direct: _route() → _execute_candidates() → _select_best()
  │      └─ Hierarchical: AOPCoordinator.orchestrate()
  ├─ 5. _guard_post(response, context)
  │      → hallucination detection, tone control, blocked phrases
  ├─ 6. voice.render(query, thread_id, vertical, response)
  │      → customer-facing messages + quick_replies
  ├─ 7. memory.record_turn()
  ├─ 8. _enrich_governance(trace, response, context)
  │      → IEEE compliance, explainability, UMF envelope
  └─ 9. Return response
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `handle_chat(query, thread_id, context)` | Main entry point |
| `_classify_orchestration_pattern(query)` | Detect single vs multi-intent |
| `_accumulate_aop_slots()` | Carry forward slot data across AOP turns |
| `_match_aop_task_selection()` | Match user input to pending AOP tasks |
| `_enrich_governance()` | Attach compliance, explainability, UMF envelope |

### Thread Context (`THREAD_CTX`)

The spine stores per-thread orchestration state in an in-memory dict:

```python
THREAD_CTX[thread_id] = {
    "aop_plan": AOPPlan,          # Pending multi-intent plan
    "accumulated_slots": dict,     # Cross-turn collected data
    "original_query": str,         # First query in thread
}
```

---

## LLM Router (`app/runtime/router.py`)

Intent-aware agent selection using LLM classification.

### Routing Logic

1. **Classify Intent** — INFORMATIONAL (seeking knowledge) vs ACTIONABLE (request to perform action) vs MIXED
2. **Match Agents** — For INFORMATIONAL: prefer FAQ/knowledge-base agents. For ACTIONABLE: prefer agents with action tools
3. **Score & Rank** — Return primary candidate + ranked alternatives with scores (0.0 – 1.0)

### Data Structures

```python
@dataclass
class RouteCandidate:
    id: str          # Agent ID
    score: float     # Confidence (0.0 – 1.0)
    reason: str      # Why this agent was selected

@dataclass
class RoutePlan:
    primary: str                      # Top-ranked agent ID
    candidates: List[RouteCandidate]  # All candidates, sorted by score
    strategy: str                     # "single" or "fanout"
```

### Fallback

If the LLM router fails, `DefaultRouter` returns all agents with equal scores.

---

## Domain Agent Engine (`app/runtime/domain_agent_engine.py`)

The core ReAct (Reason + Act) reasoning loop for domain specialist agents.

### Configuration

```python
@dataclass
class DomainAgentConfig:
    agent_id: str                          # e.g., "refunds_agent"
    domain: str                            # e.g., "refunds"
    goal: str                              # e.g., "Help with refund requests"
    policies: List[str]                    # Natural language constraints
    max_steps: int = 8                     # Max ReAct iterations
    model: str = "gpt-5-mini"             # LLM model
    temperature: float = 1.0               # LLM temperature
    top_k: int = 5                         # RAG retrieval count
    retrieval_threshold: float = 0.12      # Min relevance score
    enable_dense_retrieval: bool = False   # Hybrid TF-IDF + dense
    dense_weight: float = 0.6             # Dense retrieval weight
    sparse_weight: float = 0.4            # TF-IDF weight
```

### ReAct Loop

Each turn executes up to `max_steps` iterations:

```
For each step (1..max_steps):
    1. OBSERVE: Build prompt with query + history + context
    2. THINK:   LLM outputs JSON: {thought, action, action_input}
    3. ACT:     Execute the chosen action
    4. OBSERVE: Record observation from action result
    5. If terminal action → exit loop
```

### Available Actions

| Action | Purpose | Terminal? |
|--------|---------|-----------|
| `retrieve_knowledge` | Search RAG index | No |
| `call_tool` | Execute a registered tool | No |
| `respond` | Return final answer to user | Yes |
| `ask_user` | Request clarification from user | Yes |
| `escalate` | Hand off (ends conversation) | Yes |

### Retrieval Strategy

**Source-aware expansion:**
- Small sources (≤ 50 chunks): expand matched chunk to full document
- Large sources (> 50 chunks): return only matched chunks
- Prevents context overload while ensuring complete policy coverage

**Hybrid retrieval (optional):**
```
fused_score = sparse_weight * tfidf_score + dense_weight * dense_cosine_score
```

### Policy Caching

On first retrieval, the full policy content is cached in `ThreadState.cached_policy_content`. On subsequent turns, this cached content is injected verbatim into the system prompt:

```
--- RETRIEVED POLICY (you MUST follow ONLY these steps) ---
{cached_policy_content}
--- END OF POLICY ---
```

This prevents the agent from hallucinating workflow steps not in the actual policy.

### Response Format

```python
{
    "answer": str,                    # Agent's response text
    "score": float,                   # Confidence score
    "agent_id": str,                  # Agent identifier
    "react_trace": List[Dict],        # Full reasoning chain
    "knowledge_sources": List[Dict],  # Retrieved documents
    "tool_results": List[Dict],       # Tool call results
    "policy_sources": Dict,           # Policy grounding info
    "needs_input": bool,              # True if ask_user action
}
```

---

## AOP Coordinator (`app/orchestration/aop_coordinator.py`)

Meta-agent implementing Agent-Oriented Planning for multi-intent queries.

### 5-Step AOP Cycle

```
1. DECOMPOSE     → LLM breaks query into 1-5 atomic subtasks
2. SCORE AGENTS  → SolvabilityEstimator scores (subtask, agent) pairs
3. COMPLETENESS  → CompletenessDetector audits plan coverage
4. EXECUTE       → Delegate each subtask to assigned agent
5. FEEDBACK      → Record results in PerformanceStore
```

### Subtask Classification

Each subtask is labeled:
- **INFORMATIONAL** — Questions about policies, procedures, how things work
- **ACTION** — Explicit requests to perform actions (even without full details)

### Sequential Task Menu

For multi-intent queries, the AOP coordinator presents a numbered menu:

```
"I found 2 things to help with:
 1. Explain the refund policy
 2. Process your refund for TXN-12345
 Which would you like to start with?"
```

The user selects a task (by number, ordinal, or description), and the spine dispatches it to the assigned agent.

### Data Structures

```python
@dataclass
class Subtask:
    description: str
    assigned_agent_id: Optional[str]
    solvability_score: float          # 0.0 – 1.0
    result: Optional[Dict[str, Any]]
    success: bool
    latency_ms: int

@dataclass
class AOPPlan:
    query: str
    subtasks: List[Subtask]
    completeness: Optional[CompletenessResult]
    solvability: Optional[SolvabilityResult]
    created_ts_ms: int

    def pending_subtasks() -> List[Subtask]     # Unexecuted tasks
    def to_serializable() -> Dict               # For thread storage
```

### AOP Eligibility

Agents are eligible for AOP if:
1. `aop_eligible: true` flag is set in blueprint metadata (primary)
2. Fallback heuristic: description-based signals (customer-serving → include, internal → exclude)
3. Always excluded: `tool_operator` and `guardrails` agents

---

## Policy Guardrails (`app/runtime/policy_guardrails.py`)

Safety enforcement via `PolicyPack`. Prevents PII leakage, hallucination, and policy violations.

### Pre-Guardrails (Input)

| Check | Description |
|-------|-------------|
| Query length | Reject if > `max_query_chars` (default 5000) |
| Intent blocking | Block queries matching blocked intent patterns |
| PII redaction | Detect and redact emails, phone numbers, credit cards |

### Post-Guardrails (Output)

| Check | Description |
|-------|-------------|
| Blocked phrases | Reject responses containing internal jargon |
| Hallucination detection | Block false action claims without transaction context |
| Tone control | Strip urgency questions, async promises, file references |

### Hallucination Detection

Detects patterns like "refund has been initiated" or "successfully refunded" when:
- No transaction context exists (no order ID, amount, or payment reference)
- Not a multi-turn workflow with accumulated slots (which is legitimate)

### Tone Strip Patterns

| Pattern | Example | Replacement |
|---------|---------|-------------|
| Urgency questions | "Is this urgent?" | Stripped |
| Async promises | "I've forwarded your case" | Stripped |
| Internal jargon | "workflow", "FSM", "slots" | Stripped |
| File references | "BankFAQs.csv" | Stripped |

---

## Conversation Memory (`app/runtime/memory.py`)

Thread-scoped conversation memory implementing the "M" in PMPA (Wang et al. 2024).

### Architecture

```python
class MemoryBackend(Protocol):
    def store_turn(...): ...
    def get_turns(...): ...
    def store_snapshot(...): ...
    def get_snapshots(...): ...

class DictMemoryBackend:
    # In-memory implementation (swappable to Redis/Postgres)

class ConversationMemory:
    def record_turn(thread_id, query, response, agent_id, ...) -> MemoryTurn
    def get_conversation_context(thread_id, limit=10) -> List[Dict]
```

### Memory Turn

```python
@dataclass
class MemoryTurn:
    turn_id: int
    timestamp: float
    query: str
    response: Dict[str, Any]
    agent_id: Optional[str]
    fsm_state: Optional[str]
    slots: Dict[str, Any]
    policy_decisions: List[Dict[str, Any]]
```

### LLM Context Retrieval

`get_conversation_context()` returns only safe fields (queries, answers, agent IDs) — never internal policy decisions or slot details.

---

## Voice Renderer (`app/runtime/voice.py`)

Generates customer-facing chat messages from structured agent output.

### Sanitization

Before the LLM sees the response, these internal keys are removed:
- `mapper`, `history`, `policy_config` (FSM internals)
- `policies`, `tools`, `_accumulated_slots` (runtime internals)

### Rendering Rules

- Short, friendly, WhatsApp-like messages (1-5 messages max)
- Ask at most ONE question per turn
- No internal words (workflow, state, slots, tools)
- No file references (BankFAQs.csv → "our FAQ")
- No false promises ("I will get back to you")
- Quick replies: 2-4 relevant suggestions

### Pattern-Specific Behavior

| Response Pattern | Voice Behavior |
|-----------------|---------------|
| `domain_agent_clarification` | Rephrase question + quick replies |
| `escalation` | Apologize, suggest rephrasing |
| `aop_task_menu` | Present numbered task list |
| `aop_task_result` | Present result + remaining tasks |

---

## LLM Client (`app/llm_client.py`)

Unified client for Azure OpenAI and OpenAI APIs.

### Configuration Priority

1. Azure OpenAI: `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY`
2. OpenAI: `OPENAI_API_KEY` + optional `OPENAI_BASE_URL`

### `chat_json()`

```python
def chat_json(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 1.0,
    timeout: float = None
) -> Dict[str, Any]
```

- Forces `response_format={"type": "json_object"}` for structured output
- Retries once without temperature parameter if the model rejects it (o-series models)
- Returns parsed dict or `{"raw": msg}` on JSON decode failure
- Timeout configurable per-call via `LLM_TIMEOUT_SECONDS` env var (default 30s)
