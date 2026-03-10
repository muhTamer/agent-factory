# Frontend Components Reference

The frontend is a **Next.js 16** application with **React 19**, **Zustand** for state management, and **Tailwind CSS** for styling. It provides a chat interface with a rich explainability sidebar for inspecting agent decisions.

---

## Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Next.js | 16.1.6 | React framework (App Router) |
| React | 19.2.3 | UI library |
| Zustand | 5.0.11 | Lightweight state management |
| Tailwind CSS | 4 | Utility-first CSS |
| Radix UI | 1.4.3 | Headless accessible components |
| Lucide React | 0.575.0 | Icon library |

---

## Application Structure

```
frontend/src/
├── app/                     # Next.js App Router pages
│   ├── layout.tsx           # Root layout (Inter font, metadata)
│   ├── page.tsx             # / → SetupWizard
│   └── chat/page.tsx        # /chat → ChatContainer
│
├── components/
│   ├── chat/                # Chat UI components
│   │   ├── ChatContainer.tsx        # Main 3-column layout
│   │   ├── MessageList.tsx          # Message feed with auto-scroll
│   │   ├── MessageBubble.tsx        # Rich message rendering
│   │   ├── ChatInput.tsx            # Text input with send button
│   │   ├── QuickReplies.tsx         # Suggestion pills
│   │   ├── ConversationSidebar.tsx  # Thread history (left panel)
│   │   ├── ExplainabilityPanel.tsx  # Debug panels (right panel)
│   │   ├── TypingIndicator.tsx      # Loading animation
│   │   └── AgentAvatar.tsx          # Agent icon badges
│   │
│   └── debug/               # Explainability/debug panels
│       ├── CollapsibleSection.tsx   # Shared collapsible wrapper
│       ├── RouterPlanPanel.tsx      # Router decision visualization
│       ├── SolvabilityPanel.tsx     # AOP subtask scores
│       ├── PolicyCheckPanel.tsx     # Guardrail pass/block status
│       ├── SourcesPanel.tsx         # Knowledge sources & policy grounding
│       ├── ReActTracePanel.tsx      # Agent reasoning chain timeline
│       ├── GovernancePanel.tsx      # IEEE compliance & explainability
│       ├── DebugSidebar.tsx         # Alternative debug layout
│       └── RawJsonViewer.tsx        # Raw JSON inspector
│
├── hooks/                   # Custom React hooks
│   ├── useChat.ts           # Chat send/receive logic
│   ├── useHealth.ts         # Backend health polling
│   └── useAutoScroll.ts     # Auto-scroll on new messages
│
├── store/                   # Zustand state stores
│   ├── chatStore.ts         # Transient chat state
│   ├── threadStore.ts       # Persistent thread/message storage
│   └── setupStore.ts        # Setup wizard state
│
├── lib/                     # Utilities and API client
│   ├── api.ts               # HTTP client (getHealth, postChat)
│   ├── constants.ts         # API_BASE, agent display config
│   ├── classify.ts          # Response type classification
│   └── utils.ts             # cn() for Tailwind class merging
│
└── types/                   # TypeScript definitions
    ├── chat.ts              # ChatMessage, ResponseKind, snapshots
    ├── api.ts               # Request/response contracts
    └── concierge.ts         # Setup wizard types
```

---

## Page Layout

### Desktop (3-column)

```
┌──────────────┬─────────────────────────┬──────────────────┐
│ Conversation │      Message List       │ Explainability   │
│   Sidebar    │                         │    Panel         │
│              │   ┌─────────────────┐   │                  │
│  Thread 1    │   │ User: "I want   │   │ Router Plan      │
│  Thread 2 ←  │   │  a refund"      │   │ Sources          │
│  Thread 3    │   │                 │   │ ReAct Trace      │
│              │   │ Agent: "What is │   │ Governance       │
│              │   │  your order #?" │   │                  │
│              │   └─────────────────┘   │                  │
│              │                         │                  │
│              │  [Quick Replies]        │                  │
│              │  [Chat Input      ▶]   │                  │
└──────────────┴─────────────────────────┴──────────────────┘
```

### Mobile

Sidebar and explainability panel are shown as overlays triggered by hamburger and info buttons.

---

## Chat Components

### ChatContainer

The root component for the `/chat` page. Manages the 3-column layout and coordinates between stores.

**State Sources:**
- `useChatStore` — messages, isLoading, agents, quickReplies, selectedMessageId
- `useThreadStore` — threads, activeThreadId
- `useChat()` — sendMessage function
- `useHealth()` — backend connection polling

### MessageBubble

Renders a single message with context-aware styling based on `responseKind`:

| Response Kind | Style | Content |
|--------------|-------|---------|
| `faq` | Green badge | Answer + citations |
| `clarify` | Yellow badge | Clarification question |
| `delegate` | Blue badge | Delegation notice |
| `guardrails_block` | Red badge | Blocked message + reason |
| `workflow_progress` | Progress bar | Slot summary |
| `aop_task_menu` | Numbered list | Task selection menu |
| `aop_task_result` | Result card | Subtask result + remaining |
| `error` | Red bubble | Error message |

### ConversationSidebar

Thread history management:
- Create new threads
- Switch between threads (loads messages from threadStore)
- Delete threads
- Clear all threads
- Sorted by most recent update

### ExplainabilityPanel

Right sidebar showing debug information for the selected message. Components rendered (in order):

1. **RouterPlanPanel** — Strategy, pattern, selected agent, candidate scores
2. **SolvabilityPanel** — AOP subtask solvability bars (if AOP)
3. **PolicyCheckPanel** — Guardrail pass/block status
4. **SourcesPanel** — Knowledge sources, policy grounding, citations
5. **ReActTracePanel** — Full reasoning chain timeline
6. **GovernancePanel** — IEEE compliance, explainability levels
7. **RawJsonViewer** — Raw API response (debug mode only)

---

## Debug Panels

### CollapsibleSection

Shared wrapper for all debug panels. Provides consistent collapsible behavior.

```tsx
interface Props {
  icon: ReactNode;        // Lucide icon
  title: string;          // Section header
  tooltip: string;        // Hover explanation
  badge?: ReactNode;      // Optional status indicator
  defaultOpen?: boolean;  // Start expanded (default: true)
  children: ReactNode;    // Panel content
}
```

### RouterPlanPanel

Displays the routing decision:

- **Strategy** — `single` (one agent) or `fanout` (parallel execution)
- **Pattern** — `Direct` or `Multi-agent` (hierarchical delegation)
- **Selected Agent** — Winner card with name and confidence percentage
- **Candidates Table** — All agents with scores, reasons, and bar charts
- **Latency** — Total response time

### SolvabilityPanel

For AOP (multi-intent) queries, shows per-subtask solvability:

- Subtask description
- Solvability score (0.000 – 1.000)
- Color-coded bar (green = success, red = failed)
- Coverage ratio and missing aspects

### PolicyCheckPanel

Shows guardrail result:
- **Passed** (green shield) — Response delivered normally
- **Blocked** (red shield) — Response blocked with reason

### SourcesPanel

Aggregates knowledge evidence from multiple sources:

**Policy Grounding:**
- Policy file name badge (e.g., `refunds_policy.yaml`)
- Active workflow steps this turn (extracted from agent reasoning)

**Knowledge Sources:**
- Retrieval query
- Full passage text (no truncation)
- Source file badge
- "Retrieved in an earlier turn" label for cached sources

**RAG Citations:**
- Question/text pairs with green checkmarks

### ReActTracePanel

Visualizes the agent's reasoning chain as a step-by-step timeline:

| Action Type | Color | Icon |
|-------------|-------|------|
| `retrieve_knowledge` | Blue | Search |
| `call_tool` | Amber | Wrench |
| `respond` | Green | MessageSquare |
| `ask_user` | Purple | HelpCircle |
| `escalate` | Red | AlertTriangle |

Each step shows:
- Step number and action type badge
- **Thought** — The agent's reasoning
- **Action Input** — Tool name, args, or query (formatted JSON)
- **Observation** — Action result

Last step expanded by default; others collapsed.

### GovernancePanel

Three sections showing IEEE standards compliance:

**1. Compliance Overview:**
- Overall compliance rate (percentage)
- Per-standard breakdown with clickable IEEE hyperlinks:
  - [IEEE P3394](https://standards.ieee.org/ieee/3394/11401/) — Universal Message Format
  - [IEEE 2894-2024](https://standards.ieee.org/ieee/2894/10538/) — Explainable AI Guide
  - [IEEE 3152-2024](https://standards.ieee.org/ieee/3152/10876/) — Transparent Agency

**2. Explainability Levels:**
- **Summary** (User-facing) — Plain language description
- **Detailed** (Auditor) — Decisions, scores, policy references
- **Full** (Developer) — Complete event log with timestamps

**3. Message Envelope (UMF):**
- AI-generated disclosure
- Sender/receiver identification
- Agent chain (delegation path)

---

## State Management

### `useChatStore` (Transient)

Holds current session state. Resets when the user switches threads.

```typescript
{
  messages: ChatMessage[]           // Active thread messages
  threadId: string | null           // Backend thread_id
  isLoading: boolean                // Typing indicator
  agents: Record<string, AgentMeta> // From /health endpoint
  backendConnected: boolean         // Connection status
  debugMode: boolean                // Show raw JSON viewer
  quickReplies: string[]            // Suggestion pills
  selectedMessageId: string | null  // For explainability panel
  error: string | null              // Error banner
}
```

### `useThreadStore` (Persistent)

Persisted to `localStorage` under key `"af-threads"`. Survives page refreshes.

```typescript
{
  threads: Thread[]                           // Thread metadata
  activeThreadId: string | null               // Currently viewed thread
  messagesMap: Record<string, ChatMessage[]>  // Per-thread message history
}
```

**Thread:**
```typescript
{
  id: string              // Local UUID
  title: string           // Derived from first message
  preview: string         // Last message snippet
  createdAt: number       // Timestamp
  updatedAt: number       // Last activity timestamp
  backendThreadId: string // Backend thread_id (from API)
}
```

---

## Custom Hooks

### `useChat()`

Handles the send/receive flow:

1. Get or create active thread ID
2. Add user message to `threadStore`
3. Set loading state in `chatStore`
4. `POST /chat` to backend API
5. Classify response → extract display text, metadata, snapshots
6. Add agent message to `threadStore`
7. Update `chatStore` UI state (only if thread is still active)
8. Update thread metadata (title, preview, backendThreadId)

### `useHealth(intervalMs = 10000)`

Polls `GET /health` every 10 seconds:
- Updates `chatStore.agents` with loaded agent metadata
- Updates `chatStore.backendConnected` status
- Clears interval on component unmount

### `useAutoScroll(deps)`

Auto-scrolls the message list to the bottom when dependencies change (new messages).

---

## Type Definitions

### Response Kinds

```typescript
type ResponseKind =
  | "faq"                // FAQ/RAG answer
  | "clarify"            // Needs clarification
  | "delegate"           // Route to specialist
  | "workflow_progress"  // Workflow in progress
  | "workflow_complete"  // Workflow done
  | "hierarchical"       // AOP multi-agent result
  | "aop_task_menu"      // AOP task planning
  | "aop_task_result"    // AOP task execution
  | "guardrails_block"   // Blocked by safety
  | "error"              // System error
```

### ChatMessage

```typescript
interface ChatMessage {
  id: string
  role: "user" | "agent" | "system"
  content: string
  timestamp: number
  agentId?: string
  agentName?: string
  responseKind?: ResponseKind
  raw?: Record<string, any>          // Full API response
  routerPlan?: RouterPlan            // Router decision
  voiceChat?: ChatVoice              // Rendered messages
  latencyMs?: number                 // Response time
  workflowState?: WorkflowSnapshot   // FSM state
  aopData?: AopSnapshot              // AOP results
  aopTaskMenu?: AopTaskMenuSnapshot  // Task menu
}
```

---

## API Client (`lib/api.ts`)

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8080"

async function getHealth(): Promise<HealthResponse>
async function postChat(body: ChatRequest): Promise<ChatResponse>
```

## Response Classification (`lib/classify.ts`)

```typescript
function classifyResponse(data: any): ResponseKind
function extractDisplayText(data: any): string
function extractWorkflowSnapshot(data: any): WorkflowSnapshot | undefined
function extractAopSnapshot(data: any): AopSnapshot | undefined
function extractAopTaskMenu(data: any): AopTaskMenuSnapshot | undefined
```

The classifier examines the API response shape to determine the appropriate `ResponseKind`, then extracts the display text preferring `chat.messages[]` > `text` > `answer` > `question` > `error`.
