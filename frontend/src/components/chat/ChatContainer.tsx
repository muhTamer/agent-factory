"use client";

import { useState, useEffect } from "react";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { useChat } from "@/hooks/useChat";
import { useHealth } from "@/hooks/useHealth";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { QuickReplies } from "./QuickReplies";
import { ConversationSidebar } from "./ConversationSidebar";
import { ExplainabilityPanel } from "./ExplainabilityPanel";
import { UsageBadge } from "./UsageBadge";
import { UserMenu } from "../UserMenu";
import { useRouter } from "next/navigation";
import { useSetupStore } from "@/store/setupStore";
import {
  Bot,
  Loader2,
  Menu,
  Plus,
  Settings,
  Wifi,
  WifiOff,
} from "lucide-react";

export function ChatContainer() {
  const router = useRouter();
  useHealth();
  const { sendMessage } = useChat();

  const isLoading = useChatStore((s) => s.isLoading);
  const backendConnected = useChatStore((s) => s.backendConnected);
  const agents = useChatStore((s) => s.agents);
  const quickReplies = useChatStore((s) => s.quickReplies);
  const selectedMessageId = useChatStore((s) => s.selectedMessageId);

  const [historyOpen, setHistoryOpen] = useState(false);
  const [explainOpen, setExplainOpen] = useState(false);

  // On mount, hydrate chatStore from the active (or first) persisted thread
  useEffect(() => {
    const ts = useThreadStore.getState();
    const targetId = ts.activeThreadId ?? ts.threads[0]?.id;
    if (!targetId) return;
    // Ensure it's marked active
    if (ts.activeThreadId !== targetId) ts.switchThread(targetId);
    const msgs = ts.getMessages(targetId);
    const thread = ts.getThread(targetId);
    useChatStore.setState({
      messages: [...msgs],
      threadId: thread?.backendThreadId ?? null,
    });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const agentCount = Object.keys(agents).length;

  // When a message is selected on mobile, open explainability
  const handleMessageSelected = () => {
    setExplainOpen(true);
  };

  return (
    <div className="flex h-screen bg-white">
      {/* ── Left panel: Conversation History (desktop) ── */}
      <div className="hidden lg:block w-1/4 shrink-0 border-r">
        <ConversationSidebar />
      </div>

      {/* ── Center: Chat area ── */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <header className="flex items-center justify-between border-b px-2 py-2 sm:px-4 sm:py-3">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            {/* Burger menu — mobile only */}
            <button
              onClick={() => setHistoryOpen(true)}
              className="rounded p-1.5 text-slate-500 transition-colors hover:bg-slate-100 lg:hidden"
              title="Conversation history"
            >
              <Menu size={20} />
            </button>

            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500 text-white">
              <Bot size={18} />
            </div>
            <div>
              <h1 className="text-sm sm:text-base font-semibold text-slate-800 truncate">
                Customer Service
              </h1>
              <div className="flex items-center gap-1.5 text-xs sm:text-sm text-slate-400">
                {backendConnected ? (
                  <>
                    <Wifi size={10} className="text-green-500" />
                    <span>
                      {agentCount} agent{agentCount !== 1 ? "s" : ""} online
                    </span>
                  </>
                ) : (
                  <>
                    <WifiOff size={10} className="text-red-400" />
                    <span>Disconnected</span>
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <UsageBadge />
            {/* New chat button — mobile only (desktop has it in sidebar) */}
            <button
              onClick={() => {
                useThreadStore.getState().createThread();
                useChatStore.setState({
                  messages: [],
                  threadId: null,
                  isLoading: false,
                  activeWorkflow: null,
                  quickReplies: [],
                  selectedMessageId: null,
                  error: null,
                });
              }}
              title="New chat"
              className="rounded p-2 text-slate-400 transition-colors hover:bg-blue-50 hover:text-blue-600 lg:hidden"
            >
              <Plus size={18} />
            </button>

            {/* Back to runtime controls */}
            <button
              onClick={() => {
                useSetupStore.getState().setStep("runtime");
                router.push("/");
              }}
              title="Back to runtime"
              className="rounded p-2 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            >
              <Settings size={18} />
            </button>

            <UserMenu />
          </div>
        </header>

        {/* Loading overlay when agents aren't ready */}
        {!backendConnected ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 text-slate-500">
            <Loader2 size={28} className="animate-spin text-blue-500" />
            <p className="text-sm font-medium">Loading agents...</p>
            <p className="text-xs text-slate-400">
              Waiting for the runtime to finish initializing
            </p>
          </div>
        ) : (
          <>
            {/* Messages */}
            <MessageList onMessageClick={handleMessageSelected} />

            {/* Quick replies */}
            <QuickReplies replies={quickReplies} onSelect={sendMessage} />
          </>
        )}

        {/* Input */}
        <ChatInput
          onSend={sendMessage}
          disabled={isLoading || !backendConnected}
        />
      </div>

      {/* ── Right panel: Explainability (desktop) ── */}
      {selectedMessageId && (
        <div className="hidden lg:block w-1/4 shrink-0 border-l">
          <ExplainabilityPanel />
        </div>
      )}

      {/* ── Mobile overlay: Conversation History ── */}
      {historyOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-sm"
            onClick={() => setHistoryOpen(false)}
          />
          <div className="absolute left-0 top-0 h-full w-80 shadow-xl animate-slide-in-left">
            <ConversationSidebar onClose={() => setHistoryOpen(false)} />
          </div>
        </div>
      )}

      {/* ── Mobile overlay: Explainability ── */}
      {explainOpen && selectedMessageId && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div
            className="absolute inset-0 bg-black/30 backdrop-blur-sm"
            onClick={() => setExplainOpen(false)}
          />
          <div className="absolute right-0 top-0 h-full w-full max-w-md shadow-xl animate-slide-in-right">
            <ExplainabilityPanel onClose={() => setExplainOpen(false)} />
          </div>
        </div>
      )}
    </div>
  );
}
