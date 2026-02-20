"use client";

import { useChatStore } from "@/store/chatStore";
import { useChat } from "@/hooks/useChat";
import { useHealth } from "@/hooks/useHealth";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { QuickReplies } from "./QuickReplies";
import { DebugSidebar } from "@/components/debug/DebugSidebar";
import {
  Bot,
  Bug,
  Trash2,
  Wifi,
  WifiOff,
} from "lucide-react";

export function ChatContainer() {
  useHealth();
  const { sendMessage } = useChat();

  const isLoading = useChatStore((s) => s.isLoading);
  const backendConnected = useChatStore((s) => s.backendConnected);
  const agents = useChatStore((s) => s.agents);
  const debugMode = useChatStore((s) => s.debugMode);
  const toggleDebugMode = useChatStore((s) => s.toggleDebugMode);
  const quickReplies = useChatStore((s) => s.quickReplies);
  const clearMessages = useChatStore((s) => s.clearMessages);

  const agentCount = Object.keys(agents).length;

  return (
    <div className="flex h-screen bg-white">
      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        {/* Header */}
        <header className="flex items-center justify-between border-b px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500 text-white">
              <Bot size={18} />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-slate-800">
                Customer Service
              </h1>
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                {backendConnected ? (
                  <>
                    <Wifi size={10} className="text-green-500" />
                    <span>{agentCount} agent{agentCount !== 1 ? "s" : ""} online</span>
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
            <button
              onClick={clearMessages}
              title="Clear chat"
              className="rounded p-2 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            >
              <Trash2 size={16} />
            </button>
            <button
              onClick={toggleDebugMode}
              title="Toggle debug panel"
              className={`rounded p-2 transition-colors ${
                debugMode
                  ? "bg-blue-50 text-blue-600"
                  : "text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              }`}
            >
              <Bug size={16} />
            </button>
          </div>
        </header>

        {/* Messages */}
        <MessageList />

        {/* Quick replies */}
        <QuickReplies replies={quickReplies} onSelect={sendMessage} />

        {/* Input */}
        <ChatInput onSend={sendMessage} disabled={isLoading || !backendConnected} />
      </div>

      {/* Debug sidebar */}
      <DebugSidebar />
    </div>
  );
}
