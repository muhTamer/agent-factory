# Architecture

This document describes the system architecture, core design decisions, and request lifecycle of the Agent Factory orchestration platform.

---

## Design Principles

1. **Intent-Aware Semantics (B3.5)** — Guardrails and routing operate on semantic intent (INFORMATIONAL vs ACTIONABLE), not raw text pattern matching.
2. **ReAct Reasoning** — Domain agents reason step-by-step with explicit Thought → Action → Observation cycles, producing a fully traceable decision path.
3. **Multi-Level Orchestration** — Single-intent queries route directly to one agent; multi-intent queries use AOP hierarchical delegation with subtask decomposition.
4. **Governance-First** — IEEE compliance (P3394, 2894-2024, 3152-2024), multi-level explainability, and audit trails are woven into every response.
5. **Slot Accumulation** — Multi-turn workflows carry forward collected information without re-asking, enabling natural conversation flow.
6. **Policy Grounding** — Agents follow documented YAML policy steps exactly. The full policy is cached and injected into the system prompt to prevent hallucination.
7. **Pluggable Backends** — Memory, guardrails, tools, and routers are defined via Protocols, allowing swapping implementations without code changes.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  Next.js Chat UI · Explainability Sidebar · Debug Panels    │
├─────────────────────────────────────────────────────────────┤
│                    API Layer (FastAPI)                       │
│  POST /chat · GET /health · GET /version                    │
├─────────────────────────────────────────────────────────────┤
│                   Orchestration Layer                        │
│  RuntimeSpine · LLMRouter · AOP Coordinator                 │
├─────────────────────────────────────────────────────────────┤
│                    Agent Layer                              │
│  DomainAgentEngine (ReAct) · RAGFiniteStateMachine          │
│  WorkflowEngine (FSM) · ToolOperator                        │
├─────────────────────────────────────────────────────────────┤
│                   Service Layer                             │
│  RAG Index · Tool Registry · LLM Client · Memory            │
│  Voice Renderer · Embeddings · MCP Manager                  │
├─────────────────────────────────────────────────────────────┤
│                  Governance Layer                            │
│  Policy Guardrails · IEEE Compliance · Explainability       │
│  Audit Writer · Message Envelope (UMF)                      │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                              │
│  BankFAQs.csv · refunds_policy.yaml · complaints_policy.yaml│
│  Corpus Index · Thread State · Conversation Memory          │
└─────────────────────────────────────────────────────────────┘
```

---

## Request Lifecycle

A complete request through the system follows this path:

### 1. Ingress

The frontend sends `POST /chat` with:
```json
{
  "query": "I want a refund for order TXN-12345",
  "thread_id": "abc-123",
  "context": {}
}
```

### 2. RuntimeSpine Pipeline

The spine (`app/runtime/spine.py`) executes the **B3.5 invariant pipeline**:

```
1. ROUTE      → LLMRouter classifies intent and ranks agents
2. INFER      → Determine INFORMATIONAL / ACTIONABLE / MIXED intent
3. GUARD_PRE  → PII redaction, intent blocking, query length check
4. EXECUTE    → Dispatch to agent or AOP coordinator
5. SELECT     → Choose best response (fanout strategy)
6. RESPOND    → Voice renderer generates customer-facing text
7. GUARD_POST → Hallucination detection, tone control, blocked phrases
8. RETURN     → Governance enrichment, audit logging, response delivery
```

### 3. Routing Decision

The **LLMRouter** (`app/runtime/router.py`) classifies the query:

| Intent | Preferred Agents | Example |
|--------|-----------------|---------|
| INFORMATIONAL | FAQ/knowledge agents | "What is your refund policy?" |
| ACTIONABLE | Domain specialists with tools | "Process my refund" |
| MIXED | Both scored, highest wins | "What's the policy and process my refund" |

Output: A `RoutePlan` with ranked candidates and confidence scores.

### 4. Orchestration Pattern

The spine classifies the **orchestration pattern**:

- **Direct** — Single-intent query → route to top-ranked agent
- **Hierarchical Delegation** — Multi-intent query → AOP Coordinator decomposes into subtasks

### 5a. Direct Execution (Single Intent)

```
LLMRouter → selects refunds_agent (score: 0.92)
    ↓
DomainAgentEngine.handle(query, thread_id)
    ↓
ReAct Loop:
  Step 1: THINK → "Need to look up the refund policy"
          ACT   → retrieve_knowledge({"query": "refund policy"})
          OBS   → "Retrieved from refunds_policy.yaml: ..."

  Step 2: THINK → "User provided TXN-12345, look it up"
          ACT   → call_tool({"tool": "lookup_payment", "args": {"transaction_id": "TXN-12345"}})
          OBS   → "Tool returned: {payment_found: true, amount: 100.00}"

  Step 3: THINK → "Payment found, eligible. Initiate refund."
          ACT   → call_tool({"tool": "initiate_refund", "args": {...}})
          OBS   → "Tool returned: {refund_id: REF-001, status: success}"

  Step 4: THINK → "Refund processed successfully"
          ACT   → respond({"answer": "Your refund REF-001 has been initiated..."})
```

### 5b. AOP Execution (Multi-Intent)

```
AOPCoordinator.orchestrate(query, context)
    ↓
Step 1: DECOMPOSE
  → Subtask 1: "INFORMATIONAL: What is the refund policy?"
  → Subtask 2: "ACTION: Process refund for order TXN-12345"

Step 2: SCORE AGENTS
  → Subtask 1 → customer_faqs_agent (solvability: 0.85)
  → Subtask 2 → refunds_agent (solvability: 0.92)

Step 3: CHECK COMPLETENESS
  → Both subtasks covered → coverage 100%

Step 4: EXECUTE (sequential, user-driven)
  → Present task menu: "I found 2 things to help with:
     1. Explain the refund policy
     2. Process your refund
     Which would you like to start with?"

Step 5: FEEDBACK
  → Record agent performance in PerformanceStore
```

### 6. Voice Rendering

The **VoiceAgent** (`app/runtime/voice.py`) transforms structured agent output into customer-facing messages:

- Sanitizes internal keys (workflow state, FSM slots, policy config)
- Generates short, friendly, WhatsApp-like messages
- Produces quick-reply suggestions for the user
- Never exposes internal file names, tool names, or jargon

### 7. Governance Enrichment

Before returning the response, the spine enriches it with:

- **IEEE P3394 (UMF)** — Message envelope with sender, receiver, timestamp, provenance, agent chain
- **IEEE 2894-2024** — Multi-level explanations (Summary/Detailed/Full)
- **IEEE 3152-2024** — AI disclosure, agent identity, audit trail, escalation support
- **Compliance Report** — Per-standard compliance rate with evidence and gaps

### 8. Response

The final response includes:
```json
{
  "text": "Your refund REF-001 has been initiated...",
  "messages": ["Your refund..."],
  "quick_replies": ["Track refund", "Ask something else"],
  "agent_id": "refunds_agent",
  "score": 0.92,
  "orchestration_pattern": "direct",
  "react_trace": [...],
  "knowledge_sources": [...],
  "tool_results": [...],
  "policy_sources": {...},
  "governance": {
    "compliance_report": {...},
    "explanations": {...},
    "audit_envelope": {...}
  }
}
```

---

## Thread State Management

Each conversation thread maintains:

| State | Storage | Scope |
|-------|---------|-------|
| `ThreadState` | DomainAgentEngine (in-memory dict) | Per-agent, per-thread |
| `ConversationMemory` | DictMemoryBackend (swappable) | Cross-agent, per-thread |
| `THREAD_CTX` | RuntimeSpine (in-memory dict) | Orchestration-level, per-thread |
| `AOPPlan` | Serialized in THREAD_CTX | AOP task tracking, per-thread |

### ThreadState (Agent-Level)

```python
@dataclass
class ThreadState:
    thread_id: str
    step_history: List[ReActStep]        # Full reasoning history
    accumulated_slots: Dict[str, Any]    # Collected data (order_id, email, etc.)
    pending_question: Optional[str]      # Awaiting user response
    turn_count: int                      # Conversation turn counter
    original_query: Optional[str]        # First query in this thread
    cached_policy_content: Optional[str] # Full policy text (cached from first retrieval)
```

### Multi-Turn Flow

```
Turn 1: User asks "I want a refund"
  → Agent retrieves policy, caches it
  → Agent asks: "What is your transaction ID?"
  → State: pending_question set, cached_policy populated

Turn 2: User says "TXN-12345"
  → Agent sees cached policy in system prompt
  → Agent calls lookup_payment(TXN-12345)
  → Agent calls verify_identity()
  → Agent asks: "What is the reason for your refund?"
  → State: accumulated_slots = {transaction_id, payment_details, kyc_status}

Turn 3: User says "Charged twice"
  → Agent calls initiate_refund()
  → Agent responds with refund confirmation
  → State: accumulated_slots += {refund_id, refund_status}
```

---

## Agent Generation Pipeline

```
Factory Spec (.factory/factory_spec.json)
    ↓
Blueprint Discovery (factory/blueprints/)
    ↓
Agent Generator (app/shared/domain_agent.py)
    ↓
Generated Agent Package:
  ├── agent.py      → IAgent implementation (loads engine at init)
  ├── config.json   → Agent metadata, tool list, policies
  └── corpus.json   → Serialized knowledge base (CorpusItem[])
    ↓
Agent Registration (AgentRegistry)
    ↓
Available via RuntimeSpine for routing
```

### Agent Types

| Type | Generator | Engine | Status |
|------|-----------|--------|--------|
| `domain_agent` | `domain_agent.py` | DomainAgentEngine (ReAct) | **Active** — primary type for all agents |
| `tool_operator` | `tool_operator.py` | Direct stub execution | **Active** — single-tool wrapper |
| `knowledge_rag` | `rag.py` | RAGFiniteStateMachine (PMPA) | Legacy — superseded by `domain_agent` |
| `workflow_runner` | `workflow.py` | GenericWorkflowEngine (FSM) | Legacy — superseded by `domain_agent` |

---

## Key Data Flows

### Knowledge Retrieval (RAG)

```
Input Files (CSV/YAML/MD)
    ↓ load_corpus()
CorpusItem[] (text, source, kind, meta)
    ↓ build_index()
Index (items, vocab, TF-IDF vecs, IDF)
    ↓ query_index(query, top_k)
[(score, CorpusItem), ...]
    ↓ Injected into ReAct prompt
LLM synthesizes answer from passages
```

### Tool Execution

```
ReAct Step: call_tool({"tool": "lookup_payment", "args": {...}})
    ↓
DomainAgentEngine._action_call_tool()
    ↓ Merge accumulated_slots + explicit args
ITool.execute(slots, context)
    ↓
StubTool / HTTPTool / SQLTool / MCPTool
    ↓ Returns slot updates
accumulated_slots.update(result)
```

### MCP Tool Call (Detailed)

When the ReAct engine calls an MCP-backed tool:

```
ReAct Engine
  │  action: call_tool, tool: "demo.lookup_customer"
  │  action_input: {"customer_id": "C001"}
  ▼
ToolRegistry.get("demo.lookup_customer")  →  MCPTool instance
  │
  ▼
MCPTool._extract_arguments(slots)
  │  Filter slots to schema-defined keys only
  ▼
MCPManager.call_tool_sync("demo", "lookup_customer", {"customer_id": "C001"})
  │
  ├─ asyncio.run_coroutine_threadsafe() → background event loop
  │
  ▼  (daemon thread)
ClientSession.call_tool("lookup_customer", {"customer_id": "C001"})
  │  JSON-RPC over stdio/HTTP to MCP server process
  ▼
MCP Server → lookup_customer("C001") → JSON response
  │
  ▼
MCPManager._parse_result()
  │  1. Try structuredContent (dict)
  │  2. Try text content (JSON string → dict)
  │  3. Fallback: {"status": "completed"}
  ▼
{"customer_id": "C001", "account_status": "active", "kyc_status": "verified"}
  │
  ▼
accumulated_slots.update(result)
```

### Guardrail Pipeline

```
PRE-GUARDRAILS (before agent execution):
  1. Query length check (max 5000 chars)
  2. Intent blocking (policy-based)
  3. PII redaction (email, phone, credit card)

POST-GUARDRAILS (before response delivery):
  1. Blocked phrase enforcement (internal jargon)
  2. Hallucination detection (refund claims without transaction context)
  3. Tone control (strip urgency questions, async promises, file references)
```

---

## Component Dependencies

```
RuntimeSpine
  ├── LLMRouter
  │     └── LLM Client (chat_json)
  ├── PolicyGuardrails
  │     └── PolicyPack
  ├── AOPCoordinator
  │     ├── SolvabilityEstimator
  │     ├── CompletenessDetector
  │     └── PerformanceStore
  ├── AgentRegistry
  │     └── IAgent implementations
  │           └── DomainAgentEngine
  │                 ├── RAG Index (TF-IDF + dense)
  │                 ├── Tool Registry (ITool[])
  │                 │     ├── StubTool / HttpTool / SqlTool
  │                 │     └── MCPTool → MCPManager (singleton)
  │                 │           └── MCP Server(s) (stdio / HTTP)
  │                 ├── LLM Client
  │                 └── ConversationMemory
  ├── VoiceAgent
  │     └── LLM Client
  ├── ExplainabilityEngine
  │     └── Trace
  ├── IEEEComplianceChecker
  └── AuditWriter (JSONL)
```

---

## MCP Integration Architecture

The Model Context Protocol (MCP) adapter allows domain agents to call tools hosted in external processes or remote servers. It is the fourth adapter type alongside Stub, HTTP, and SQL.

### Design Decisions

1. **Singleton MCPManager** — One manager instance shared across all agents. Prevents duplicate server subprocesses when multiple agents reference the same MCP server.

2. **Background daemon thread** — A dedicated thread runs an asyncio event loop to bridge the async MCP SDK to the synchronous `ITool.execute()` contract. The ReAct engine and tool interface remain unchanged.

3. **Optional dependency** — The `mcp` package import is guarded with `try/except`. If not installed, MCP config is silently skipped and the system operates normally with stub/HTTP/SQL tools.

4. **Zero engine changes** — `MCPTool.describe()` appends parameter info (from JSON Schema) to the description string. The existing prompt builder picks it up without modification.

### Component Diagram

```
ToolRegistry
  │
  ├── StubTool(s)    ← stub_tools.py (built-in demos)
  ├── HttpTool(s)    ← tools_config.json "type": "http"
  ├── SqlTool(s)     ← tools_config.json "type": "sql"
  │
  └── MCPTool(s)     ← tools_config.json "mcp_servers"
        │
        ▼
    MCPManager (singleton)
    ┌──────────────────────────────────────┐
    │ Background asyncio event loop         │
    │                                       │
    │  MCPServerConnection("demo")          │
    │    ├── ClientSession                  │
    │    ├── tools: {echo, add, lookup_...} │
    │    └── _lock: asyncio.Lock            │
    │                                       │
    │  MCPServerConnection("remote_tools")  │
    │    ├── ClientSession                  │
    │    ├── tools: {...}                   │
    │    └── _lock: asyncio.Lock            │
    └──────────────────────────────────────┘
```

### Lifecycle

| Phase | What happens |
|-------|-------------|
| **Startup** | `ToolRegistry.load_mcp_servers()` → `MCPManager.get_instance()` → `connect_servers()` → `list_tools()` → register `MCPTool` instances |
| **Runtime** | `MCPTool.execute()` → `MCPManager.call_tool_sync()` → async `session.call_tool()` → parse result → return slot updates |
| **Shutdown** | `ToolRegistry.shutdown()` → `MCPManager.shutdown()` → close sessions → close transports → stop event loop |

### Configuration

MCP servers are declared in `.factory/tools_config.json` under `mcp_servers`:

```json
{
  "mcp_servers": [
    {
      "enabled": true,
      "id": "demo",
      "transport": "stdio",
      "command": "python",
      "args": ["tests/fixtures/mock_mcp_server.py"],
      "timeout": 30,
      "tool_prefix": true
    }
  ]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Server identifier; used as tool name prefix |
| `enabled` | No | Connect at startup (default: `true`) |
| `transport` | Yes | `"stdio"` (subprocess) or `"streamable_http"` (remote) |
| `command` / `args` | stdio | Process to launch |
| `url` / `headers` | http | Remote endpoint; `${ENV_VAR}` expansion in headers |
| `timeout` | No | Connection and call timeout in seconds (default: 30) |
| `tool_prefix` | No | Prefix tool names with `{id}.` (default: `true`) |

### Test Coverage

| Test file | What it covers |
|-----------|---------------|
| `tests/test_mcp_integration.py` | 31 tests: MCPTool adapter, MCPManager lifecycle, mock server integration, error handling, argument extraction, result parsing |
| `tests/fixtures/mock_mcp_server.py` | FastMCP test server with 12 tools covering happy paths, error scenarios, and edge cases |

---

## Security Considerations

- **PII Redaction** — Emails, phone numbers, and credit card numbers are redacted in pre-guardrails before reaching agents
- **Internal Data Sanitization** — Voice renderer strips FSM state, slot names, policy file paths, and tool registry details before LLM generates customer text
- **No Credential Exposure** — HTTP tool adapter expands `${ENV_VAR}` tokens from environment, never hardcoded
- **Hallucination Prevention** — Post-guardrails detect false action claims (e.g., "refund processed") when no transaction context exists
- **Policy Grounding** — Agents are instructed to follow only documented policy steps; cached policy is injected verbatim to prevent invention of procedures
- **MCP Argument Filtering** — MCPTool extracts only schema-declared keys from slots before sending to the server, preventing unrelated slot data from leaking to external processes
- **MCP Environment Expansion** — `${ENV_VAR}` tokens in MCP server headers and env are expanded from `os.environ`, never hardcoded in configuration
