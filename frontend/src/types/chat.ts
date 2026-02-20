import type { ChatResponse, ChatVoice, RouterPlan } from "./api";

export type MessageRole = "user" | "agent" | "system";

export type ResponseKind =
  | "faq"
  | "clarify"
  | "delegate"
  | "workflow_progress"
  | "workflow_complete"
  | "hierarchical"
  | "guardrails_block"
  | "error";

export interface WorkflowSnapshot {
  workflowId: string;
  currentState: string;
  terminal: boolean;
  slots: Record<string, unknown>;
  missingSlots?: string[];
  allStates: string[];
}

export interface AopSnapshot {
  subtaskResults: Array<{
    subtask: string;
    agentId: string | null;
    success: boolean;
    solvabilityScore: number;
    latencyMs: number;
  }>;
  totalLatencyMs: number;
  completeness: {
    complete: boolean;
    missing: string[];
    coverageRatio: number;
  };
}

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  agentId?: string;
  agentName?: string;
  responseKind?: ResponseKind;
  raw?: ChatResponse;
  routerPlan?: RouterPlan;
  voiceChat?: ChatVoice;
  latencyMs?: number;
  workflowState?: WorkflowSnapshot;
  aopData?: AopSnapshot;
}
