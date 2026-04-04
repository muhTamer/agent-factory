import { create } from "zustand";
import type { ChatMessage, WorkflowSnapshot } from "@/types/chat";
import type { AgentMeta } from "@/types/api";

interface ChatState {
  messages: ChatMessage[];
  clearMessages: () => void;

  /** The backend-generated thread_id for multi-turn API calls */
  threadId: string | null;
  setThreadId: (id: string) => void;

  isLoading: boolean;
  setLoading: (v: boolean) => void;

  agents: Record<string, AgentMeta>;
  setAgents: (agents: Record<string, AgentMeta>) => void;

  backendConnected: boolean;
  setBackendConnected: (v: boolean) => void;

  activeWorkflow: WorkflowSnapshot | null;
  setActiveWorkflow: (w: WorkflowSnapshot | null) => void;

  debugMode: boolean;
  toggleDebugMode: () => void;

  quickReplies: string[];
  setQuickReplies: (qr: string[]) => void;

  selectedMessageId: string | null;
  setSelectedMessageId: (id: string | null) => void;

  error: string | null;
  setError: (e: string | null) => void;

  /** LLM usage stats returned from backend */
  usage: { session_llm_calls: number; session_llm_limit: number; session_remaining: number } | null;
  setUsage: (u: ChatState["usage"]) => void;

  /** Internal: tracks which threads have in-flight requests */
  _loadingThreads: Record<string, boolean>;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],

  clearMessages: () =>
    set({
      messages: [],
      threadId: null,
      activeWorkflow: null,
      quickReplies: [],
      selectedMessageId: null,
    }),

  threadId: null,
  setThreadId: (id) => set({ threadId: id }),

  isLoading: false,
  setLoading: (v) => set({ isLoading: v }),

  agents: {},
  setAgents: (agents) => set({ agents }),

  backendConnected: false,
  setBackendConnected: (v) => set({ backendConnected: v }),

  activeWorkflow: null,
  setActiveWorkflow: (w) => set({ activeWorkflow: w }),

  debugMode: false,
  toggleDebugMode: () => set((s) => ({ debugMode: !s.debugMode })),

  quickReplies: [],
  setQuickReplies: (qr) => set({ quickReplies: qr }),

  selectedMessageId: null,
  setSelectedMessageId: (id) => set({ selectedMessageId: id }),

  error: null,
  setError: (e) => set({ error: e }),

  usage: null,
  setUsage: (u) => set({ usage: u }),

  _loadingThreads: {},
}));
