"use client";

import { useState } from "react";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import { useChat } from "@/hooks/useChat";
import { useHealth } from "@/hooks/useHealth";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { QuickReplies } from "./QuickReplies";
import { ConversationSidebar } from "./ConversationSidebar";
import { ExplainabilityPanel } from "./ExplainabilityPanel";
import {
  Bot,
  Menu,
  Plus,
  Wifi,
  WifiOff,
} from "lucide-react";

export function ChatContainer() {
  useHealth();
  const { sendMessage } = useChat();

  const isLoading = useChatStore((s) => s.isLoading);
  const backendConnected = useChatStore((s) => s.backendConnected);
  const agents = useChatStore((s) => s.agents);
  const quickReplies = useChatStore((s) => s.quickReplies);
  const selectedMessageId = useChatStore((s) => s.selectedMessageId);

  const [historyOpen, setHistoryOpen] = useState(false);
  const [explainOpen, setExplainOpen] = useState(false);

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
        <header className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-3">
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
              <h1 className="text-base font-semibold text-slate-800">
                Customer Service
              </h1>
              <div className="flex items-center gap-1.5 text-sm text-slate-400">
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

          <div className="flex items-center gap-1">
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
          </div>
        </header>

        {/* Messages */}
        <MessageList onMessageClick={handleMessageSelected} />

        {/* Quick replies */}
        <QuickReplies replies={quickReplies} onSelect={sendMessage} />

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
