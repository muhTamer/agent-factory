"use client";

import { useThreadStore } from "@/store/threadStore";
import { useChatStore } from "@/store/chatStore";
import { Plus, Trash2, MessageSquare, X } from "lucide-react";

interface Props {
  onClose?: () => void;
}

export function ConversationSidebar({ onClose }: Props) {
  const threads = useThreadStore((s) => s.threads);
  const activeThreadId = useThreadStore((s) => s.activeThreadId);

  const handleNewChat = () => {
    // Create thread directly in threadStore
    useThreadStore.getState().createThread();
    // Clear chatStore state for the new empty thread
    useChatStore.setState({
      messages: [],
      threadId: null,
      isLoading: false,
      activeWorkflow: null,
      quickReplies: [],
      selectedMessageId: null,
      error: null,
    });
    onClose?.();
  };

  const handleSwitch = (id: string) => {
    if (id === activeThreadId) {
      onClose?.();
      return;
    }
    // Switch active thread in threadStore
    useThreadStore.getState().switchThread(id);
    // Read messages and metadata after switch
    const msgs = useThreadStore.getState().getMessages(id);
    const thread = useThreadStore.getState().getThread(id);
    // Derive isLoading from per-thread tracking
    const threadIsLoading = !!useChatStore.getState()._loadingThreads[id];
    // Load into chatStore
    useChatStore.setState({
      messages: [...msgs],
      threadId: thread?.backendThreadId ?? null,
      isLoading: threadIsLoading,
      activeWorkflow: null,
      quickReplies: [],
      selectedMessageId: null,
      error: null,
    });
    onClose?.();
  };

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    useThreadStore.getState().deleteThread(id);
    // If we deleted the active thread, load the next one or clear
    const nextActiveId = useThreadStore.getState().activeThreadId;
    if (nextActiveId) {
      const msgs = useThreadStore.getState().getMessages(nextActiveId);
      const thread = useThreadStore.getState().getThread(nextActiveId);
      useChatStore.setState({
        messages: [...msgs],
        threadId: thread?.backendThreadId ?? null,
        isLoading: false,
        activeWorkflow: null,
        quickReplies: [],
        selectedMessageId: null,
        error: null,
      });
    } else {
      useChatStore.setState({
        messages: [],
        threadId: null,
        isLoading: false,
        activeWorkflow: null,
        quickReplies: [],
        selectedMessageId: null,
        error: null,
      });
    }
  };

  const formatTime = (ts: number) => {
    const d = new Date(ts);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) {
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) {
      return "Yesterday";
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  };

  const sorted = [...threads].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <div className="flex h-full w-full flex-col bg-slate-50">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-white px-4 py-3">
        <h2 className="text-base font-semibold text-slate-700">Conversations</h2>
        <div className="flex items-center gap-1">
          <button
            onClick={handleNewChat}
            title="New conversation"
            className="rounded p-1.5 text-slate-500 transition-colors hover:bg-blue-50 hover:text-blue-600"
          >
            <Plus size={18} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="rounded p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 lg:hidden"
            >
              <X size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto">
        {sorted.length === 0 ? (
          <div className="px-4 py-8 text-center">
            <MessageSquare
              size={32}
              className="mx-auto mb-2 text-slate-300"
            />
            <p className="text-sm text-slate-400">No conversations yet</p>
            <p className="mt-1 text-sm text-slate-400">
              Start a new chat to begin
            </p>
          </div>
        ) : (
          <div className="py-1">
            {sorted.map((thread) => {
              const isActive = thread.id === activeThreadId;
              return (
                <div
                  key={thread.id}
                  onClick={() => handleSwitch(thread.id)}
                  className={`group flex cursor-pointer items-start gap-3 px-4 py-3 transition-colors ${
                    isActive
                      ? "bg-blue-50 border-r-2 border-blue-500"
                      : "hover:bg-slate-100"
                  }`}
                >
                  <MessageSquare
                    size={16}
                    className={`mt-0.5 shrink-0 ${
                      isActive ? "text-blue-500" : "text-slate-400"
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-2">
                      <p
                        className={`truncate text-sm ${
                          isActive
                            ? "font-semibold text-blue-700"
                            : "font-medium text-slate-700"
                        }`}
                      >
                        {thread.title}
                      </p>
                      <span className="shrink-0 text-xs text-slate-400">
                        {formatTime(thread.updatedAt)}
                      </span>
                    </div>
                    {thread.preview && (
                      <p className="mt-0.5 truncate text-sm text-slate-400">
                        {thread.preview}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={(e) => handleDelete(e, thread.id)}
                    className="mt-0.5 shrink-0 rounded p-1 text-slate-300 opacity-0 transition-all hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
                    title="Delete conversation"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
