import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { ChatMessage } from "@/types/chat";
import { saveChatHistory, getChatHistory } from "@/lib/concierge-api";

export interface Thread {
  id: string;
  title: string;
  preview: string;
  createdAt: number;
  updatedAt: number;
  /** The backend-generated thread_id (UUID) for multi-turn context */
  backendThreadId: string | null;
}

interface ThreadState {
  threads: Thread[];
  activeThreadId: string | null;
  messagesMap: Record<string, ChatMessage[]>;

  createThread: () => string;
  switchThread: (id: string) => void;
  deleteThread: (id: string) => void;
  clearAllThreads: () => void;
  setActiveThreadId: (id: string | null) => void;

  addMessage: (threadId: string, msg: ChatMessage) => void;
  getMessages: (threadId: string) => ChatMessage[];
  setMessages: (threadId: string, msgs: ChatMessage[]) => void;
  getThread: (threadId: string) => Thread | undefined;
  updateThreadMeta: (
    threadId: string,
    patch: Partial<Pick<Thread, "title" | "preview" | "backendThreadId">>
  ) => void;

  /** Load chat history from backend and merge into store */
  loadFromBackend: () => Promise<void>;
}

function makeId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export const useThreadStore = create<ThreadState>()(
  persist(
    (set, get) => ({
      threads: [],
      activeThreadId: null,
      messagesMap: {},

      createThread: () => {
        const id = makeId();
        const now = Date.now();
        const thread: Thread = {
          id,
          title: "New conversation",
          preview: "",
          createdAt: now,
          updatedAt: now,
          backendThreadId: null,
        };
        set((s) => ({
          threads: [thread, ...s.threads],
          activeThreadId: id,
          messagesMap: { ...s.messagesMap, [id]: [] },
        }));
        return id;
      },

      switchThread: (id) => {
        set({ activeThreadId: id });
      },

      deleteThread: (id) => {
        set((s) => {
          const threads = s.threads.filter((t) => t.id !== id);
          const messagesMap = { ...s.messagesMap };
          delete messagesMap[id];
          const activeThreadId =
            s.activeThreadId === id
              ? threads[0]?.id ?? null
              : s.activeThreadId;
          return { threads, messagesMap, activeThreadId };
        });
      },

      clearAllThreads: () => {
        set({ threads: [], activeThreadId: null, messagesMap: {} });
      },

      setActiveThreadId: (id) => set({ activeThreadId: id }),

      addMessage: (threadId, msg) => {
        set((s) => {
          const existing = s.messagesMap[threadId] ?? [];
          return {
            messagesMap: {
              ...s.messagesMap,
              [threadId]: [...existing, msg],
            },
          };
        });
      },

      getMessages: (threadId) => {
        return get().messagesMap[threadId] ?? [];
      },

      setMessages: (threadId, msgs) => {
        set((s) => ({
          messagesMap: { ...s.messagesMap, [threadId]: msgs },
        }));
      },

      getThread: (threadId) => {
        return get().threads.find((t) => t.id === threadId);
      },

      updateThreadMeta: (threadId, patch) => {
        set((s) => ({
          threads: s.threads.map((t) =>
            t.id === threadId
              ? { ...t, ...patch, updatedAt: Date.now() }
              : t
          ),
        }));
      },

      loadFromBackend: async () => {
        try {
          const data = await getChatHistory();
          if (data.threads && data.threads.length > 0) {
            set({
              threads: data.threads as Thread[],
              messagesMap: data.messagesMap as Record<string, ChatMessage[]>,
              activeThreadId: data.activeThreadId,
            });
          }
        } catch (err) {
          console.warn("[ThreadStore] Failed to load from backend:", err);
        }
      },
    }),
    {
      name: "af-threads",
      version: 3,
      migrate: (persisted) => persisted as any, // eslint-disable-line @typescript-eslint/no-explicit-any
      partialize: (state) => ({
        threads: state.threads,
        activeThreadId: state.activeThreadId,
        messagesMap: state.messagesMap,
      }),
    }
  )
);

// ── Debounced sync to backend ──────────────────────────────────────
// Saves chat history to the backend 2s after the last state change.
let _syncTimer: ReturnType<typeof setTimeout> | null = null;

function _scheduleSync() {
  if (_syncTimer) clearTimeout(_syncTimer);
  _syncTimer = setTimeout(() => {
    const { threads, activeThreadId, messagesMap } = useThreadStore.getState();
    if (threads.length === 0) return;
    saveChatHistory({ threads, activeThreadId, messagesMap }).catch((err) =>
      console.warn("[ThreadStore] Backend sync failed:", err)
    );
  }, 2000);
}

// Subscribe to store changes — fires on every state update
useThreadStore.subscribe((state, prevState) => {
  // Only sync when threads or messages actually changed
  if (
    state.threads !== prevState.threads ||
    state.messagesMap !== prevState.messagesMap
  ) {
    _scheduleSync();
  }
});
