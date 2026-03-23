# Agent Factory — Autonomous Multi-Agent Orchestration System

An autonomous agent factory that transforms domain specifications into executable, policy-driven customer service agents. The system combines **ReAct reasoning**, **retrieval-augmented generation (RAG)**, **hierarchical task planning (AOP)**, and **IEEE-compliant governance** to deliver transparent, auditable AI decision-making.

Built as a thesis project exploring four research questions around agent generation, explainability, governance trade-offs, and autonomous orchestration.

---

## Table of Contents

- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Overview

The Agent Factory takes a factory specification (`.factory/factory_spec.json`) and generates a fleet of domain-specialist agents, each equipped with:

- **Knowledge retrieval** — TF-IDF + optional dense embeddings over CSV, YAML, and Markdown corpora
- **Tool execution** — Pluggable tool adapters (HTTP, SQL, stub, MCP) via the `ITool` interface
- **ReAct reasoning** — Step-by-step Observe → Think → Act loop with full trace logging
- **Policy enforcement** — Natural language constraints injected into the LLM prompt, grounded in YAML policy documents
- **Multi-turn memory** — Thread-scoped conversation state with slot accumulation across turns

A **RuntimeSpine** orchestrates the full request pipeline: intent-aware routing → guardrails → agent execution → voice rendering → governance enrichment → audit logging.

For complex multi-intent queries, an **AOP Coordinator** decomposes the request into subtasks, scores agent suitability, checks plan completeness, and executes tasks sequentially with user-driven selection.

---

## Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                       │
│  Chat UI · Explainability Sidebar · Governance Panels            │
└──────────────────────────┬───────────────────────────────────────┘
                           │ POST /chat
┌──────────────────────────▼───────────────────────────────────────┐
│                      RuntimeSpine (FastAPI)                       │
│                                                                   │
│  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Router  │→ │ Guardrails│→ │ Execute  │→ │ Voice Renderer  │  │
│  │ (LLM)   │  │ (pre/post)│  │          │  │ (LLM)           │  │
│  └─────────┘  └───────────┘  └────┬─────┘  └─────────────────┘  │
│                                    │                              │
│              ┌─────────────────────┼──────────────────┐          │
│              │                     │                   │          │
│         Direct Route          AOP Coordinator     Governance      │
│              │              (multi-intent)        Enrichment      │
│              ▼                     │              (IEEE 2894,     │
│     ┌────────────────┐            ▼               3152, P3394)   │
│     │ Domain Agent   │    ┌──────────────┐                       │
│     │ Engine (ReAct) │    │ Decompose →  │                       │
│     │                │    │ Score →       │                       │
│     │ Observe→Think  │    │ Complete →   │                       │
│     │    →Act loop   │    │ Execute →    │                       │
│     └────────────────┘    │ Feedback     │                       │
│              │            └──────────────┘                       │
│    ┌─────────┴──────────┐                                        │
│    │                     │                                        │
│  RAG Index           Tool Registry                               │
│  (TF-IDF +           (Stub/HTTP/SQL/MCP)                         │
│   Dense)                  │                                      │
│                      MCP Manager                                 │
│                      (stdio / HTTP servers)                      │
└──────────────────────────────────────────────────────────────────┘
```

> For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md) and the [docs/](docs/) directory.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm (for the frontend)
- An Azure OpenAI or OpenAI API key

### 1. Clone and set up Python environment

```powershell
git clone https://github.com/muhTamer/agent-factory.git
cd agent-factory

py -3.11 -m venv .venv
.\.venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

```powershell
copy .env.example .env
# Edit .env with your API credentials
```

Required variables (Azure OpenAI):
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-01
```

Or for OpenAI directly:
```env
OPENAI_API_KEY=your-key
```

Optional (for dense retrieval):
```env
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://your-embedding-resource.cognitiveservices.azure.com/
AZURE_OPENAI_EMBEDDING_KEY=your-embedding-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 3. Install pre-commit hooks

```powershell
pre-commit install
pre-commit run --all-files
```

### 4. Start the backend

```powershell
.\.venv\Scripts\Activate
uvicorn app.main:app --reload --port 8080
```

The API is available at `http://127.0.0.1:8080`. Check `GET /health` for agent status.

### 5. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000/chat` to use the chat interface.

### 6. Run tests

```powershell
# All unit tests (no LLM calls)
pytest -q

# Integration tests (requires LLM API)
pytest -m integration -v

# E2E scenario tests
pytest tests/test_e2e_scenarios.py -v
```

---

## Project Structure

```
agent-factory/
├── app/                              # Backend application
│   ├── main.py                       # FastAPI entry point (/health, /chat)
│   ├── llm_client.py                 # Unified Azure/OpenAI LLM client
│   ├── infer_capabilities.py         # Document classification & agent proposal
│   │
│   ├── runtime/                      # Core runtime components
│   │   ├── spine.py                  # Request orchestration backbone (B3.5)
│   │   ├── router.py                 # Intent-aware LLM routing
│   │   ├── routing.py               # Router protocol & DefaultRouter
│   │   ├── domain_agent_engine.py    # ReAct reasoning engine
│   │   ├── rag_fsm.py               # RAG finite state machine (PMPA)
│   │   ├── policy_guardrails.py      # Safety guardrails (pre/post)
│   │   ├── governance_guardrails.py  # Governance-level-aware guardrails
│   │   ├── memory.py                # Thread-scoped conversation memory
│   │   ├── voice.py                 # Customer-facing text rendering
│   │   ├── embeddings.py            # Azure OpenAI embedding wrapper
│   │   ├── interfaces.py            # IAgent protocol contract
│   │   └── tools/                   # Tool system
│   │       ├── interface.py         # ITool abstract base
│   │       ├── registry.py          # ToolRegistry (name → ITool)
│   │       ├── stub_tools.py        # Demo tool implementations
│   │       ├── mcp_manager.py      # MCP server lifecycle & sync bridge
│   │       └── adapters/            # HTTP, SQL, Stub, MCP adapters
│   │
│   ├── orchestration/               # Multi-agent orchestration
│   │   ├── aop_coordinator.py       # Action-Oriented Planning (5-step)
│   │   ├── neural_solvability_estimator.py  # Neural MLP estimator
│   │   ├── scorer.py               # LLM-based response scorer
│   │   ├── training_data_generator.py  # Training data pipeline
│   │   └── reward_model_trainer.py  # MLP training with validation
│   │
│   ├── governance/                  # IEEE standards compliance
│   │   ├── ieee_compliance.py       # P3394, 2894-2024, 3152-2024 checker
│   │   └── explainability.py        # Multi-level explanation engine
│   │
│   ├── shared/                      # Shared utilities & generators
│   │   ├── rag.py                   # Corpus loading, TF-IDF indexing
│   │   ├── domain_agent.py          # Domain agent code generator
│   │   ├── workflow.py              # Workflow agent generator
│   │   ├── tool_operator.py         # Tool operator generator
│   │   └── schemas/                 # JSON Schema validation
│   │
│   ├── concierge/                   # Blueprint creation
│   │   └── blueprint_creator.py     # LLM-driven blueprint planning
│   │
│   └── deploy/                      # Deployment & spec building
│       └── spec_builder.py          # Blueprint discovery & input resolution
│
├── frontend/                         # Next.js chat UI
│   └── src/
│       ├── app/                     # Pages (/, /chat)
│       ├── components/
│       │   ├── chat/                # Chat UI components
│       │   ├── setup/               # Onboarding wizard & tool config panel
│       │   └── debug/               # Explainability panels + estimator toggle
│       ├── hooks/                   # useChat, useHealth, useAutoScroll
│       ├── store/                   # Zustand state management
│       ├── lib/                     # API client, constants, classify
│       └── types/                   # TypeScript type definitions
│
├── evaluation/                       # Research question evaluation harnesses
│   ├── harness.py                   # RQ1 harness — routing accuracy & orchestration
│   ├── rq2_harness.py               # RQ2 harness — explainability & IEEE compliance
│   ├── rq2_judge.py                 # RQ2 LLM-as-judge (faithfulness, completeness, clarity)
│   ├── run_governance_comparison.py # RQ3 harness — governance trade-off evaluation
│   ├── governance_metrics.py        # RQ3 metrics aggregation
│   ├── solvability_comparison.py    # TF-IDF vs Neural estimator comparison
│   ├── scenarios/                   # Ground truth & governance scenario definitions
│   ├── results/                     # Evaluation output (rq1/, rq2/, rq3/)
│   └── rq4/                         # RQ4 harness — multi-turn conversation evaluation
│
├── scripts/                          # Utility scripts
│   ├── _bootstrap.py                # Shared agent registry loading
│   ├── generate_training_data.py    # Generate reward model training data
│   ├── train_reward_model.py        # Train the MLP reward model
│   └── run_comparison.py            # Run TF-IDF vs Neural comparison
│
├── models/                           # Trained model artifacts
│   ├── reward_mlp.pt               # MLP weights
│   └── training_metadata.json      # Training metadata
│
├── data/                             # Knowledge & policy documents
│   ├── BankFAQs.csv                 # FAQ question/answer pairs
│   ├── refunds_policy.yaml          # Refund workflow & eligibility rules
│   └── complaints_policy.yaml       # Complaint handling procedures
│
├── generated/                        # Auto-generated agent packages (gitignored)
│   ├── refunds_agent/               # {agent.py, config.json, corpus.json}
│   ├── complaints_agent/
│   └── customer_faqs_agent/
│
├── .factory/                         # Factory configuration
│   ├── factory_spec.json            # Agent & tool definitions
│   ├── tools_config.json            # Tool adapter config (stub/HTTP/SQL/MCP)
│   └── audit/                       # Runtime trace logs
│
├── factory/blueprints/               # Agent blueprint templates
│   └── domain_agent/blueprint.yaml
│
├── tests/                            # Test suite
│   ├── test_e2e_scenarios.py        # End-to-end with real LLM
│   ├── test_domain_agent_engine.py  # ReAct engine unit tests
│   ├── test_domain_agent_workflows.py
│   ├── test_multi_intent_workflows.py
│   ├── test_complaint_pipeline.py   # Complaint flow scenario tests
│   ├── fixtures/
│   │   ├── configurable_mcp_server.py  # Config-driven MCP server (hot-reload)
│   │   └── mcp_tools_config.json       # Tool definitions & scenarios
│   └── ...
│
├── docs/                             # Detailed documentation
│   ├── backend.md                   # Backend components reference
│   ├── frontend.md                  # Frontend components reference
│   ├── agents.md                    # Agent types & generation
│   ├── tools-and-rag.md             # Tools, RAG, embeddings
│   ├── governance.md                # IEEE standards & guardrails
│   └── neural-solvability.md       # Neural solvability estimator
│
├── ARCHITECTURE.md                   # System architecture & data flow
├── .env.example                      # Environment variable template
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Test configuration
└── .pre-commit-config.yaml           # Linting (Black + Ruff)
```

---

## Configuration

### Factory Specification (`.factory/factory_spec.json`)

The factory spec defines the agent fleet. Each agent entry specifies:

```json
{
  "id": "agent_refunds",
  "type": "autogen",
  "blueprint": "domain_agent",
  "inputs": {
    "domain": "refunds",
    "goal": "Help customers with refund requests",
    "knowledge_sources": ["data/BankFAQs.csv", "data/refunds_policy.yaml"],
    "available_tools": ["lookup_payment", "verify_identity", "initiate_refund"],
    "policies": ["Follow refund eligibility rules", "Verify identity before processing"]
  }
}
```

### Policy Documents (`data/`)

YAML policy files define agent behavior rules. The refund policy includes:
- Eligibility criteria (KYC status, account status, transaction age)
- Amount thresholds (auto-approve ≤ €5,000, manager approval > €5,000)
- Step-by-step agent workflow (collect reference → lookup → verify → process)

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Yes* | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Yes* | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Yes* | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | No | API version (default: `2024-02-01`) |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (alternative to Azure) |
| `AZURE_OPENAI_EMBEDDING_ENDPOINT` | No | Separate embedding endpoint |
| `AZURE_OPENAI_EMBEDDING_KEY` | No | Embedding API key |
| `LLM_TIMEOUT_SECONDS` | No | LLM call timeout (default: 30) |

*Either Azure or OpenAI credentials required.

---

## Running the System

### Backend Only

```powershell
uvicorn app.main:app --reload --port 8080
```

**Endpoints:**
- `GET /health` — Agent status, loaded agents, capabilities
- `GET /version` — API version
- `POST /chat` — Send a query; body: `{"query": "...", "thread_id": "..."}`

### Full Stack (Backend + Frontend)

Terminal 1 (Backend):
```powershell
.\.venv\Scripts\Activate
uvicorn app.main:app --reload --port 8080
```

Terminal 2 (Frontend):
```powershell
cd frontend
npm run dev
```

Navigate to `http://localhost:3000/chat`.

### Example Chat Flow

1. **User:** "I want a refund for order TXN-12345"
2. **System:** Routes to refund agent → retrieves policy → looks up payment → verifies identity → initiates refund
3. **Explainability sidebar** shows: Router plan, ReAct trace, knowledge sources, policy grounding, governance compliance

---

## Testing

```powershell
# Run all non-integration tests
pytest -q

# Run integration tests (requires LLM API key)
pytest -m integration -v

# Run specific test file
pytest tests/test_domain_agent_engine.py -v

# Run E2E scenarios
pytest tests/test_e2e_scenarios.py -v
```

### Test Categories

| Test File | What It Tests |
|-----------|---------------|
| `test_domain_agent_engine.py` | ReAct loop, tool calls, retrieval, escalation |
| `test_domain_agent_workflows.py` | Multi-turn refund/complaint workflows |
| `test_e2e_scenarios.py` | End-to-end with real LLM (integration) |
| `test_multi_intent_workflows.py` | AOP decomposition and execution |
| `test_aop_coordinator.py` | AOP coordinator logic |
| `test_spine_orchestration.py` | RuntimeSpine request pipeline |
| `test_ieee_compliance.py` | IEEE standards compliance checking |
| `test_faq_rag_agent.py` | FAQ retrieval and solvability |
| `test_tool_operators.py` | Tool operator agent loading |
| `test_mcp_integration.py` | MCP adapter, manager, and server integration |
| `test_complaint_pipeline.py` | Complaint flow: triage, compensation, escalation scenarios |

---

## Documentation

Detailed documentation is available in the [docs/](docs/) directory:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System design, data flow diagrams, request lifecycle
- **[docs/backend.md](docs/backend.md)** — Backend components: spine, router, engine, memory, voice
- **[docs/frontend.md](docs/frontend.md)** — Frontend components: chat UI, debug panels, state management
- **[docs/agents.md](docs/agents.md)** — Agent types, generation pipeline, ReAct reasoning
- **[docs/tools-and-rag.md](docs/tools-and-rag.md)** — Tool system, RAG indexing, embeddings
- **[docs/governance.md](docs/governance.md)** — IEEE standards, explainability levels, guardrails
- **[docs/neural-solvability.md](docs/neural-solvability.md)** — Neural solvability estimator: architecture, training, evaluation
