# Agents — Types, Generation, and ReAct Reasoning

This document covers how agents are defined, generated, and executed in the Agent Factory.

---

## Agent Contract (`IAgent`)

Every agent implements the `IAgent` protocol (`app/runtime/interfaces.py`):

```python
class IAgent(Protocol):
    def load(self, spec: Dict[str, Any]) -> None:
        """Initialize from factory spec (paths, params, etc.)."""

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request. Must return JSON-safe dict."""

    def metadata(self) -> Dict[str, Any]:
        """Return lightweight info (id, type, ready flags)."""
```

This is a structural typing protocol — agents don't need to inherit from `IAgent`, they just need to implement the three methods.

---

## Agent Types

### 1. Domain Agent (Primary)

**Generator:** `app/shared/domain_agent.py`
**Engine:** `DomainAgentEngine` (ReAct reasoning loop)
**Use Case:** Full-featured specialist combining RAG + tools + multi-turn reasoning

This is the primary agent type. It handles knowledge retrieval, tool execution, policy enforcement, and multi-turn conversation — all autonomously via the ReAct loop.

**Inputs:**
```python
{
    "domain": "refunds",                    # Agent's domain
    "goal": "Help with refund requests",    # Natural language goal
    "knowledge_sources": [                  # Documents for RAG
        "data/BankFAQs.csv",
        "data/refunds_policy.yaml"
    ],
    "available_tools": [                    # Tools from registry
        "lookup_payment",
        "verify_identity",
        "initiate_refund"
    ],
    "policies": [                           # Natural language constraints
        "Follow refund eligibility rules",
        "Verify identity before processing"
    ],
    "max_steps": 8,                         # ReAct iteration limit
    "model": "gpt-5-mini"                   # LLM model
}
```

**Generated Files:**
```
generated/<agent_id>/
├── agent.py      # IAgent wrapper (loads engine)
├── config.json   # Metadata, tools, policies
└── corpus.json   # Serialized knowledge base
```

### 2. Tool Operator

**Generator:** `app/shared/tool_operator.py`
**Engine:** Direct stub execution
**Use Case:** Thin wrapper around a single tool

Returns predefined stub responses. Used as leaf agents in hierarchical delegation.

### Legacy Agent Types (Not Actively Used)

The following agent types exist in the codebase but are **not used by any active agent**. All three agents in the current factory spec are `domain_agent`, which unified and replaced both of these earlier types.

#### Knowledge RAG Agent (Legacy)

**Generator:** `app/shared/rag.py`
**Engine:** `RAGFiniteStateMachine` (`app/runtime/rag_fsm.py`)

A standalone FAQ retrieval agent with a state machine (ANALYZE → CLARIFY → RETRIEVE → RESPOND/DELEGATE) and solvability scoring. Superseded by `domain_agent`, which incorporates the same RAG retrieval capabilities within its ReAct loop alongside tool execution and policy enforcement.

#### Workflow Runner (Legacy)

**Generator:** `app/shared/workflow.py`
**Engine:** `GenericWorkflowEngine` (`app/runtime/workflow_engine.py`)

An FSM-based agent that executes step-by-step workflows defined in `workflow_spec.json`. Superseded by `domain_agent`, which handles multi-step workflows dynamically through ReAct reasoning rather than rigid state machine transitions.

---

## Agent Generation Pipeline

```
.factory/factory_spec.json
    │
    │  For each agent entry:
    ▼
┌─────────────────────────────────────────────┐
│ Blueprint Discovery                          │
│ factory/blueprints/<blueprint>/blueprint.yaml│
│ → entrypoint: app.shared.domain_agent.build_agent │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ build_agent(agent_id, inputs, gen_dir)      │
│                                              │
│ 1. Load knowledge sources → CorpusItem[]    │
│ 2. Serialize corpus → corpus.json           │
│ 3. Write config → config.json               │
│ 4. Generate agent code → agent.py           │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│ Agent Registration                           │
│ AgentRegistry.register(agent_id, agent)     │
│                                              │
│ agent.load(spec) → Initialize engine        │
│ agent.metadata()  → Expose to /health       │
│ agent.handle(req) → Process queries         │
└─────────────────────────────────────────────┘
```

### Generated `agent.py` Structure

```python
class Agent(IAgent):
    def load(self, spec):
        # 1. Load config.json (agent metadata)
        # 2. Load corpus.json → build TF-IDF index
        # 3. Register all stub tools from STUB_TOOLS
        # 4. Load LLM client (chat_json)
        # 5. Load conversation memory
        # 6. Load embeddings (optional, for dense retrieval)
        # 7. Pre-compute dense vectors at startup
        # 8. Create DomainAgentEngine with all components

    def handle(self, request):
        # Extract query + thread_id from request
        # Delegate to engine.handle()

    def metadata(self):
        # Return agent info for /health endpoint
```

---

## ReAct Reasoning Engine

The `DomainAgentEngine` (`app/runtime/domain_agent_engine.py`) implements the ReAct (Reason + Act) pattern.

### Loop Structure

```
handle(query, thread_id, context)
    │
    ├─ Get or create ThreadState for thread_id
    ├─ If pending_question and user answered → absorb into context
    │
    └─ For step in 1..max_steps:
         │
         ├─ BUILD PROMPT
         │   System: domain context, tools, policies, cached policy
         │   User: query, conversation history, previous steps
         │
         ├─ CALL LLM
         │   → JSON: {"thought": "...", "action": "...", "action_input": {...}}
         │
         ├─ EXECUTE ACTION
         │   ├─ retrieve_knowledge → query TF-IDF index
         │   ├─ call_tool → execute ITool with slot merging
         │   ├─ respond → return answer (TERMINAL)
         │   ├─ ask_user → return question (TERMINAL)
         │   └─ escalate → return escalation (TERMINAL)
         │
         ├─ RECORD STEP
         │   ReActStep(step_number, thought, action, action_input, observation)
         │
         └─ CACHE POLICY (if first retrieval)
             Store full policy in ThreadState.cached_policy_content
```

### System Prompt Structure

The system prompt given to the LLM includes:

```
You are a {domain} domain specialist.
Goal: {goal}

Available actions:
  1. retrieve_knowledge({"query": "search terms"})
  2. call_tool({"tool": "tool_name", "args": {...}})
  3. respond({"answer": "your final answer"})
  4. ask_user({"question": "what you need"})
  5. escalate({"reason": "why this needs human attention"})

Available tools:
  - lookup_payment: Stub implementation of 'lookup_payment'
  - verify_identity: Stub implementation of 'verify_identity'
  - initiate_refund: Stub implementation of 'initiate_refund'
  ...

Policy guidance:
  - Follow refund eligibility rules...
  - Verify identity before processing...

--- RETRIEVED POLICY (you MUST follow ONLY these steps) ---
{cached_policy_content}
--- END OF POLICY ---

Guidelines:
  - ONLY perform actions EXPLICITLY listed in the policy
  - NEVER invent security checks beyond what the policy states
  - Use call_tool to look up data BEFORE asking the user
  - Do NOT ask for information retrievable via tools
  ...
```

### Thread State

Each conversation thread maintains persistent state:

```python
@dataclass
class ThreadState:
    thread_id: str
    step_history: List[ReActStep]        # Complete reasoning history
    accumulated_slots: Dict[str, Any]    # Data collected across turns
    pending_question: Optional[str]      # Awaiting user response
    turn_count: int                      # Turn counter
    original_query: Optional[str]        # First query in thread
    cached_policy_content: Optional[str] # Full policy from first retrieval
```

### Multi-Turn Example

```
Turn 1: "I want a refund for order TXN-12345"
  Step 1: retrieve_knowledge("refund policy eligibility")
    → Retrieves refunds_policy.yaml content
    → Policy cached in ThreadState
  Step 2: call_tool("lookup_payment", {"transaction_id": "TXN-12345"})
    → {payment_found: true, amount: 100.00, age_days: 5}
  Step 3: call_tool("lookup_customer", {})
    → {kyc_status: "verified", account_status: "active"}
  Step 4: ask_user("What is the reason for your refund request?")
    → pending_question set

Turn 2: "I was charged twice"
  Step 1: (system prompt includes cached policy)
    THINK: "User provided reason. All eligibility checks passed. Amount ≤€5000 → auto-approve."
  Step 2: call_tool("initiate_refund", {reason: "double charge", ...})
    → {refund_id: "REF-001", status: "success"}
  Step 3: respond("Your refund REF-001 has been initiated...")
```

### Response Fields

The engine produces a rich response dict:

```python
{
    "answer": "Your refund has been initiated...",
    "score": 0.85,
    "agent_id": "refunds_agent",
    "needs_input": False,

    # Reasoning trace (for ReActTracePanel)
    "react_trace": [
        {
            "step": 1,
            "thought": "Need to look up the refund policy",
            "action": "retrieve_knowledge",
            "action_input": {"query": "refund policy"},
            "observation": "Retrieved from refunds_policy.yaml: ..."
        },
        ...
    ],

    # Knowledge sources (for SourcesPanel)
    "knowledge_sources": [
        {
            "query": "refund policy eligibility",
            "passages": ["..."],
            "sources": ["refunds_policy.yaml"],
            "from_prior_turn": False
        }
    ],

    # Tool results (for ReActTracePanel)
    "tool_results": [
        {
            "step": 2,
            "tool": "lookup_payment",
            "args": {"transaction_id": "TXN-12345"},
            "result": "{payment_found: true, ...}"
        }
    ],

    # Policy grounding (for SourcesPanel)
    "policy_sources": {
        "policies": ["refunds_policy.yaml"],
        "active_entries": ["Step 2: Look up payment details", "Step 6: Determine refund type"]
    }
}
```

---

## Factory Specification

Agents are defined in `.factory/factory_spec.json`:

```json
{
  "version": "1.0",
  "vertical": "fintech",
  "agents": [
    {
      "id": "agent_refunds",
      "type": "autogen",
      "blueprint": "domain_agent",
      "status": "ready",
      "inputs": {
        "domain": "refunds",
        "goal": "Help customers with refund requests...",
        "knowledge_sources": ["data/BankFAQs.csv", "data/refunds_policy.yaml"],
        "available_tools": ["lookup_payment", "verify_identity", "initiate_refund"],
        "policies": ["Follow refund eligibility rules..."]
      },
      "blueprint_meta": {
        "agent_kind": "domain_agent",
        "aop_eligible": true,
        "capabilities": ["refunds", "multi_turn", "tool_use", "knowledge_retrieval"]
      }
    }
  ]
}
```

### Current Agent Fleet

| Agent | Domain | Knowledge | Tools | AOP Eligible |
|-------|--------|-----------|-------|-------------|
| `agent_refunds` | Refunds | BankFAQs.csv, refunds_policy.yaml | lookup_payment, verify_identity, initiate_refund, create_ticket, handoff_to_human, lookup_customer | Yes |
| `agent_complaints` | Complaints | BankFAQs.csv, complaints_policy.yaml | create_ticket, handoff_to_human, lookup_customer | Yes |
| `agent_faq` | Customer FAQ | BankFAQs.csv, refunds_policy.yaml, complaints_policy.yaml | create_ticket, handoff_to_human | Yes |

---

## Blueprint Schema

Agent blueprints are validated against `app/shared/schemas/agent_blueprint.schema.json`:

```json
{
  "id": "string (required)",
  "agent_kind": "knowledge_rag | workflow_runner | tool_operator | domain_agent",
  "description": "string (required)",
  "capabilities": ["array of strings"],
  "inputs": {
    // Varies by agent_kind:
    // domain_agent: domain, goal, knowledge_sources, available_tools, policies
    // knowledge_rag: docs
    // workflow_runner: workflow_spec
    // tool_operator: tool
  }
}
```

---

## Solvability Estimation (PMPA Pattern)

### AOP Solvability (Neural + TF-IDF)

The AOP coordinator supports two pluggable estimator backends for scoring (subtask, agent) pairs:

- **Neural (default)** — `NeuralSolvabilityEstimator` uses all-MiniLM-L6-v2 sentence embeddings + a trained 3-layer MLP (`768→256→64→1`). Scoring: `α × neural_sim + β × historical_perf` (α=0.6, β=0.4). Better at handling lexical gaps and semantic paraphrases.
- **TF-IDF (fallback)** — `SolvabilityEstimator` uses TF-IDF cosine similarity. Faster, fully deterministic, no GPU required.

The neural estimator is used by default when `torch` and `sentence-transformers` are installed; otherwise the system falls back to TF-IDF automatically. Estimators can be switched at runtime via the API or the frontend EstimatorTogglePanel. See [Neural Solvability docs](neural-solvability.md) for the training pipeline, evaluation, and architecture details.

### RAG-Level Solvability (PMPA)

The RAG FSM implements solvability estimation using multiple signals:

```python
@dataclass
class SolvabilitySignals:
    tfidf_score: float       # Best TF-IDF cosine similarity
    coverage_ratio: float    # Fraction of query tokens in corpus
    top_k_avg: float         # Average score of top-k hits
    confidence: float        # Combined solvability (0.0 – 1.0)
    should_delegate: bool    # True if below delegation threshold
    reasoning: str           # Explanation
```

**Decision thresholds:**
| Threshold | Value | Action |
|-----------|-------|--------|
| `solvability_threshold` | 0.25 | Below → delegate to specialist |
| `clarification_threshold` | 0.15 | Below + short query → ask for clarification |
| `relevance_gate` | 0.12 | Below → hit not considered relevant |
| `max_clarifications` | 2 | Max clarification rounds before fallback |
