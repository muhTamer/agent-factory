"use client";

import { useChatStore } from "@/store/chatStore";
import { useAutoScroll } from "@/hooks/useAutoScroll";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

export function MessageList() {
  const messages = useChatStore((s) => s.messages);
  const isLoading = useChatStore((s) => s.isLoading);
  const endRef = useAutoScroll([messages.length, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      {messages.length === 0 && (
        <div className="flex h-full items-center justify-center">
          <div className="text-center max-w-sm">
            <h2 className="text-lg font-semibold text-slate-700 mb-2">
              How can I help you today?
            </h2>
            <p className="text-sm text-slate-400">
              Try asking about a refund policy, reporting an issue, or updating
              your account details.
            </p>
          </div>
        </div>
      )}

      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}

      {isLoading && <TypingIndicator />}

      <div ref={endRef} />
    </div>
  );
}
