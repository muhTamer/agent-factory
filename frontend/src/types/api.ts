// Types matching the FastAPI backend response contract

export interface ChatRequest {
  query: string;
  thread_id?: string | null;
  request_id?: string | null;
  context?: Record<string, unknown> | null;
}

export interface RouterCandidate {
  id: string;
  score: number;
  reason: string;
}

export interface RouterPlan {
  primary: string;
  strategy: "single" | "fanout";
  candidates: RouterCandidate[];
}

export interface ChatVoice {
  messages: string[];
  quick_replies: string[];
}

export interface AgentMeta {
  id: string;
  type: string;
  ready: boolean;
  description?: string;
  capabilities?: string[];
  workflow_id?: string;
  state?: string;
  terminal?: boolean;
  context?: { docs: number; policies: number; tools: number };
}

export interface HealthResponse {
  status: string;
  agents: Record<string, AgentMeta>;
  dry_run: boolean;
  request_id: string;
}

// --- Response shapes (discriminated by fields) ---

export interface ChatResponseBase {
  thread_id: string;
  request_id: string;
  router_plan?: RouterPlan;
  agent_id?: string;
  score?: number;
  chat?: ChatVoice;
  text?: string;
}

export interface FaqResponse extends ChatResponseBase {
  answer: string;
  citations?: Record<string, unknown>[];
  rag_state?: "RESPOND";
}

export interface RagClarifyResponse extends ChatResponseBase {
  action: "clarify";
  question: string;
  rag_state: "CLARIFY";
  thread_active: true;
  solvability?: Record<string, unknown>;
}

export interface RagDelegateResponse extends ChatResponseBase {
  action: "delegate";
  delegate: { suggested_type: string; suggested_id?: string; reason: string };
  rag_state?: "DELEGATE";
  solvability?: Record<string, unknown>;
}

export interface WorkflowHistoryEntry {
  state: string;
  timestamp: number;
  actions: string[];
}

export interface WorkflowInProgressResponse extends ChatResponseBase {
  workflow_id: string;
  current_state: string;
  terminal: false;
  slots: Record<string, unknown>;
  missing_slots?: string[];
  history?: WorkflowHistoryEntry[];
  action?: string;
  status?: "awaiting_info" | "missing_info" | "in_progress";
  mapper?: Record<string, unknown>;
}

export interface WorkflowCompletedResponse extends ChatResponseBase {
  workflow_id: string;
  current_state: string;
  terminal: true;
  slots: Record<string, unknown>;
  history?: WorkflowHistoryEntry[];
  status?: "completed";
}

export interface SubtaskResult {
  subtask: string;
  agent_id: string | null;
  success: boolean;
  solvability_score: number;
  latency_ms: number;
  result: Record<string, unknown> | null;
}

export interface AopResponse extends ChatResponseBase {
  orchestration_pattern: "hierarchical_delegation";
  subtask_results: SubtaskResult[];
  completeness: {
    complete: boolean;
    missing: string[];
    coverage_ratio: number;
    reasoning: string;
  };
  solvability: {
    assignments: Record<string, string>;
    assignment_scores: Record<string, number>;
  };
  total_latency_ms: number;
}

export interface GuardrailsBlockResponse {
  error: string;
  reason?: string;
  text?: string;
  request_id?: string;
  response?: { text: string };
}

export interface ErrorResponse {
  error: string;
  request_id?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ChatResponse = Record<string, any>;
