# Tools, RAG, and Embeddings

This document covers the tool system, retrieval-augmented generation (RAG), and embedding infrastructure.

---

## Tool System

### ITool Interface (`app/runtime/tools/interface.py`)

All tools implement the abstract `ITool` contract:

```python
class ITool(ABC):
    def execute(self, slots: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool and return slot updates."""

    def describe(self) -> Dict[str, Any]:
        """Return human/machine-readable tool metadata."""

    def __call__(self, slots, context) -> Dict[str, Any]:
        """Make instances directly callable."""
```

**Parameters:**
- `slots` — Current accumulated state (merged with explicit tool args by the engine)
- `context` — Runtime context (thread_id, etc.)
- **Return** — Dict of key/value updates merged back into the agent's accumulated slots

### Tool Registry (`app/runtime/tools/registry.py`)

Central registry mapping tool names to `ITool` implementations:

```python
class ToolRegistry:
    def register(name: str, tool: ITool) -> None
    def get(name: str) -> Optional[ITool]
    def all_names() -> List[str]
    def load_config(config: List[Dict]) -> None   # Config-driven setup
    def as_callable_dict() -> Dict[str, Callable]  # Engine-ready
```

### Tool Adapters

#### Stub Adapter (`app/runtime/tools/adapters/stub.py`)

Wraps a plain Python callable as an `ITool`:

```python
class StubTool(ITool):
    def __init__(self, name: str, fn: Callable[[Dict, Dict], Dict])
    def execute(self, slots, context) -> Dict
    def describe(self) -> Dict

    @classmethod
    def from_response(cls, name, response_dict) -> StubTool
        # Create a tool that always returns a fixed dict
```

#### HTTP Adapter (`app/runtime/tools/adapters/http.py`)

Calls external HTTP APIs:

```python
class HTTPTool(ITool):
    def __init__(self, name, url, method, headers, timeout, slot_map)
    def execute(self, slots, context) -> Dict
```

- **POST/PUT/PATCH** — Sends slots as JSON body
- **GET** — Sends slots as query parameters
- **Environment expansion** — `${ENV_VAR}` tokens in headers expanded from `os.environ`
- **Slot mapping** — Response keys renamed via `slot_map` before merge

#### SQL Adapter (`app/runtime/tools/adapters/sql.py`)

Executes parameterized SQL queries:

```python
class SQLTool(ITool):
    def __init__(self, name, dsn, query, slot_map)
    def execute(self, slots, context) -> Dict
```

- **SQLite** — `sqlite:///path.db` (stdlib, no extra deps)
- **PostgreSQL/MySQL** — Requires SQLAlchemy + driver
- **Named params** — `:param_name` substituted from slots
- **First row returned** — Columns renamed via `slot_map`

#### MCP Adapter (`app/runtime/tools/adapters/mcp.py`)

Connects to any [Model Context Protocol](https://modelcontextprotocol.io/) server and exposes each of its tools as a standard `ITool`:

```python
class MCPTool(ITool):
    def __init__(self, name, mcp_tool_name, server_id, schema, description, manager, timeout)
    def execute(self, slots, context) -> Dict
    def describe(self) -> Dict
```

- **Auto-discovery** — Tools are discovered at startup via `list_tools()`; no manual registration needed
- **Schema extraction** — JSON Schema from the MCP server is used to filter relevant slot keys and generate parameter descriptions for the LLM prompt
- **Sync bridge** — The async MCP SDK is bridged to the synchronous `ITool.execute()` contract via `MCPManager` (see below)
- **Prefixed names** — When `tool_prefix: true`, tools are registered as `{server_id}.{tool_name}` to avoid collisions (e.g., `demo.lookup_customer`)

### MCPManager (`app/runtime/tools/mcp_manager.py`)

Singleton that manages MCP server connections, tool discovery, and lifecycle:

```python
class MCPManager:
    @classmethod
    def get_instance(cls) -> MCPManager        # Thread-safe singleton
    def connect_servers(configs) -> Dict[str, List[str]]  # Connect + discover
    def get_tools() -> Dict[str, MCPTool]      # All discovered tools
    def call_tool_sync(server_id, tool_name, arguments, timeout) -> Dict
    def shutdown() -> None                      # Graceful cleanup
```

**Architecture:**

```
Main thread (sync)          Background daemon thread (async)
─────────────────           ─────────────────────────────────
ITool.execute()
  │
  ├─► MCPManager.call_tool_sync()
  │       │
  │       ├─► asyncio.run_coroutine_threadsafe()  ─►  event loop
  │       │                                             │
  │       │                                        session.call_tool()
  │       │                                             │
  │       ◄─── future.result(timeout) ◄────────────────┘
  │
  ▼
  slot updates merged into ReAct state
```

**Transport support:**

| Transport | Config keys | Use case |
|-----------|-------------|----------|
| `stdio` | `command`, `args`, `env` | Local process (subprocess) |
| `streamable_http` | `url`, `headers` | Remote MCP server |

**Result parsing priority:**
1. `structuredContent` — Used directly if present (MCP 1.x)
2. Text content — Parsed as JSON; if dict, merged as slot updates
3. Fallback — `{"status": "completed"}` if no parseable content

**Optional dependency** — The `mcp` package import is guarded. If not installed, MCP configuration is silently skipped and the system falls back to stub/HTTP/SQL tools.

---

### Stub Tools (`app/runtime/tools/stub_tools.py`)

Pre-defined demo implementations that return happy-path results for any input:

| Tool | Returns |
|------|---------|
| `verify_identity` | `{kyc_status: "verified", identity_verified: true}` |
| `lookup_payment` | `{payment_found: true, settlement_status: "settled", amount: <from slots>, age_days: 5}` |
| `initiate_refund` | `{refund_id: "DEMO-REF-001", refund_status: "success", refund_initiated: true}` |
| `create_ticket` | `{ticket_id: "DEMO-TKT-001", ticket_status: "created"}` |
| `handoff_to_human` | `{handed_off: true, handoff_agent: "human_ops_team"}` |
| `lookup_customer` | `{account_status: "active", kyc_status: "verified", customer_found: true}` |

All stubs return success regardless of input, allowing end-to-end workflow demos without real backend integrations.

### Tool Aliases

The factory spec may use abstract API-style names. `TOOL_ALIASES` maps them to concrete stub names:

```python
TOOL_ALIASES = {
    "PaymentsAPI":              "lookup_payment",
    "IdentityVerificationAPI":  "verify_identity",
    "TicketingSystemAPI":       "create_ticket",
    "EscalationWorkflowTool":   "create_ticket",
    "CRM":                      "lookup_customer",
    "AuditLogger":              "create_ticket",
    "ConversationLogger":       "create_ticket",
    "RAG_Retriever":            "",  # Handled internally by the engine
}
```

### Tool Registration in Agents

Generated agents register **all** stub tools unconditionally:

```python
from app.runtime.tools.registry import ToolRegistry
from app.runtime.tools.adapters.stub import StubTool
from app.runtime.tools.stub_tools import STUB_TOOLS

registry = ToolRegistry()
for tool_name, stub_fn in STUB_TOOLS.items():
    registry.register(tool_name, StubTool(tool_name, stub_fn))
tools = {name: registry.get(name) for name in registry.all_names()}
```

This ensures every tool referenced in policy workflows is available, preventing "tool not found" errors during the ReAct loop.

### How Tools Are Called

In the ReAct loop, when the LLM chooses `call_tool`:

```python
# LLM output:
{"thought": "Look up the payment", "action": "call_tool",
 "action_input": {"tool": "lookup_payment", "args": {"transaction_id": "TXN-12345"}}}

# Engine execution:
slots = {**state.accumulated_slots, **tool_args}  # Merge context
result = tool.execute(slots, {"thread_id": state.thread_id})
state.accumulated_slots.update(result)  # Absorb results
```

---

## RAG System (`app/shared/rag.py`)

Lightweight, dependency-free retrieval-augmented generation using TF-IDF.

### Corpus Loading

```python
def load_corpus(paths: List[str]) -> List[CorpusItem]
```

**Supported file types:**

| Extension | Behavior |
|-----------|----------|
| `.csv` | Detects Q/A columns (case-insensitive); falls back to text chunking |
| `.yaml`, `.yml` | Indexed as readable text chunks |
| `.json` | Indexed as text chunks |
| `.md` | Split by headings, re-chunked (max 1200 chars) |
| `.txt` | Split by headings, re-chunked (max 1200 chars) |
| Other | Text snapshot (max 12000 chars) |

### CorpusItem

```python
@dataclass
class CorpusItem:
    text: str              # Document chunk content
    source: str            # Filename (e.g., "BankFAQs.csv")
    kind: str              # "csv_qa" | "md" | "txt" | "yaml" | "other"
    meta: Dict[str, Any]   # Extra info (e.g., {"q": "...", "a": "..."} for FAQ pairs)
```

### Index Building

```python
def build_index(corpus_items: List[CorpusItem]) -> Index
```

**Algorithm:**
1. Tokenize each document (lowercase alphanumeric tokens)
2. Compute document frequency (DF) per token
3. Compute IDF: `log((N+1) / (DF+1)) + 1.0`
4. TF-IDF vectorization: `TF = 1 + log(count)`, weight = `TF * IDF`
5. L2 normalize each document vector

**Index structure:**
```python
@dataclass
class Index:
    items: List[CorpusItem]              # All corpus items
    vocab: Dict[str, int]                # Token → document frequency
    vecs: List[Dict[str, float]]         # TF-IDF sparse vectors per item
    idf: Dict[str, float]               # Inverse document frequency
```

### Retrieval

```python
def query_index(index: Index, query: str, top_k: int = 5) -> List[Tuple[float, CorpusItem]]
```

1. Tokenize query using same tokenizer
2. Compute query TF-IDF vector
3. Cosine similarity against all document vectors
4. Return top-k hits sorted by score (descending)

### Retrieval in the Domain Agent Engine

The engine's `_action_retrieve` method adds source-aware expansion:

- **Small sources (≤ 50 chunks):** Expand matched chunk to include ALL chunks from same source file. This ensures the agent sees the complete policy, not just one fragment.
- **Large sources (> 50 chunks):** Return only matched chunks to prevent context overload.

### Hybrid Retrieval (Optional)

When dense embeddings are available, the engine uses hybrid fusion:

```python
fused_score = sparse_weight * tfidf_score + dense_weight * dense_cosine_score
```

Default weights: `sparse_weight = 0.4`, `dense_weight = 0.6`.

Dense scores are computed as dot products against pre-computed embedding vectors.

---

## Embeddings (`app/runtime/embeddings.py`)

Thin wrapper around Azure OpenAI embeddings for dense retrieval.

### Configuration

| Variable | Purpose |
|----------|---------|
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | Dedicated embedding endpoint (optional) |
| `AZURE_OPENAI_EMBEDDING_KEY` | Embedding API key |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Model deployment (default: `text-embedding-3-small`) |

Falls back to the main LLM endpoint if no dedicated embedding endpoint is configured.

### Factory Function

```python
def get_embed_fn(
    model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> Callable[[List[str]], List[List[float]]]
```

Returns a callable that:
1. Batches input texts (100 per batch) to respect Azure token limits
2. Calls the embeddings API
3. L2-normalizes output vectors
4. Returns `List[List[float]]` (one vector per input text)

### Pre-Computation at Startup

When embedding is available, domain agents pre-compute dense vectors for their entire corpus at load time:

```python
if embed_fn and corpus_items:
    texts = [item.text for item in corpus_items]
    dense_vecs = embed_fn(texts)  # Pre-compute once
```

These vectors are stored in `DomainAgentEngine.dense_vecs` and used for hybrid retrieval on every query.

---

## Configuration-Driven Tool Setup

Tools can be configured via `tools_config.json`:

```json
{
  "tools": [
    {
      "name": "initiate_refund",
      "type": "http",
      "url": "https://erp.example.com/api/refunds",
      "method": "POST",
      "headers": {
        "Authorization": "Bearer ${REFUND_API_KEY}"
      },
      "timeout": 15,
      "slot_map": {
        "refundId": "refund_id",
        "status": "refund_status"
      }
    },
    {
      "name": "lookup_order",
      "type": "sql",
      "dsn": "sqlite:///data/orders.db",
      "query": "SELECT * FROM orders WHERE order_id = :order_id",
      "slot_map": {
        "total_amount": "order_amount"
      }
    }
  ]
}
```

### MCP Server Configuration

MCP servers are declared in the `mcp_servers` array. Each entry describes one server; tools are auto-discovered at startup:

```json
{
  "mcp_servers": [
    {
      "enabled": true,
      "id": "demo",
      "transport": "stdio",
      "command": "python",
      "args": ["tests/fixtures/configurable_mcp_server.py", "tests/fixtures/mcp_tools_config.json"],
      "env": {},
      "timeout": 30,
      "tool_prefix": true
    },
    {
      "enabled": false,
      "id": "remote_tools",
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN}"
      },
      "timeout": 15,
      "tool_prefix": true
    }
  ]
}
```

**MCP server config fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique server identifier; used as tool name prefix |
| `enabled` | No | `true` (default) to connect at startup |
| `transport` | Yes | `"stdio"` (local process) or `"streamable_http"` (remote) |
| `command` | stdio | Executable to launch (e.g., `"python"`) |
| `args` | stdio | Command-line arguments (e.g., `["server.py"]`) |
| `url` | http | Remote MCP endpoint URL |
| `headers` | http | Static headers; `${ENV_VAR}` tokens expanded from `os.environ` |
| `env` | No | Environment variables passed to the subprocess |
| `timeout` | No | Connection/call timeout in seconds (default: 30) |
| `tool_prefix` | No | Prefix tool names with `{id}.` to avoid collisions (default: `true`) |

### Configurable MCP Server (`tests/fixtures/configurable_mcp_server.py`)

A config-driven MCP server that reads tool definitions from a JSON file. Users define tools, parameters, responses, and conditional scenarios — no Python code required.

**Config file format** (`tests/fixtures/mcp_tools_config.json`):

```json
{
  "server_name": "demo-server",
  "tools": [
    {
      "name": "lookup_payment",
      "description": "Look up a payment by transaction ID.",
      "parameters": {
        "transaction_id": { "type": "string", "required": true }
      },
      "response": {
        "transaction_id": "{{transaction_id}}",
        "payment_found": true,
        "settlement_status": "settled",
        "original_transaction_amount": 2000
      },
      "scenarios": [
        {
          "_comment": "Payment not found",
          "when": { "transaction_id": { "starts_with": "FAIL" } },
          "response": { "payment_found": false, "message": "Not found." }
        }
      ]
    }
  ]
}
```

**Template syntax:**
- `{{param_name}}` — Replaced with the input parameter value
- `{{param_name:default}}` — Replaced with value, or `default` if not provided

**Scenario condition operators:**

| Operator | Example | Description |
|----------|---------|-------------|
| `starts_with` | `"starts_with": "FAIL"` | String prefix match (case-insensitive) |
| `equals` | `"equals": "high"` | Exact match |
| `contains` | `"contains": "fraud"` | Substring match (case-insensitive) |
| `greater_than` | `"greater_than": 5000` | Numeric comparison |
| `less_than` | `"less_than": 100` | Numeric comparison |
| `in` | `"in": ["fraud", "regulatory"]` | Membership check |

Scenarios are evaluated in order; the first match wins. The matched scenario's response is merged on top of the default response.

**Hot-reload:** The server re-reads `mcp_tools_config.json` on every tool call, so edits to response values and scenarios take effect immediately without restarting the runtime. Tool names, parameters, and descriptions are registered once at startup (a restart is needed to add/remove tools or change signatures).

### MCP Tool Config API (Concierge)

The concierge API exposes CRUD endpoints for editing `mcp_tools_config.json` from the frontend:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/concierge/mcp-tools` | GET | Read the full tool config |
| `/concierge/mcp-tools` | PUT | Replace the full tool config (tools list + server_name) |
| `/concierge/mcp-tools/{tool_name}` | PUT | Update a single tool definition |
| `/concierge/mcp-tools/{tool_name}` | DELETE | Delete a tool by name |

The frontend exposes these via a **Configure Tools** slide-over panel in the Runtime step, allowing users to edit tool responses and scenarios while testing in the chat.

### Startup Sequence

When `tools_config.json` exists in `.factory/`, the service loads tools in this order:

1. **Stub tools** — All 6 built-in stubs registered as baseline
2. **Config overrides** — `tools` array entries replace stubs with HTTP/SQL implementations
3. **MCP tools** — `mcp_servers` array entries connect to MCP servers and register discovered tools

MCP tools are additive — they do not replace stub tools unless they share the same name (unlikely with `tool_prefix: true`).
