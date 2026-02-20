"use client";

import { useCallback } from "react";
import { postChat } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import {
  classifyResponse,
  extractDisplayText,
  extractWorkflowSnapshot,
  extractAopSnapshot,
} from "@/lib/classify";
import { getAgentDisplay } from "@/lib/constants";
import type { ChatMessage } from "@/types/chat";

function makeId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function useChat() {
  const store = useChatStore();

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || store.isLoading) return;

      // Add user message immediately (optimistic)
      const userMsg: ChatMessage = {
        id: makeId(),
        role: "user",
        content: text.trim(),
        timestamp: Date.now(),
      };
      store.addMessage(userMsg);
      store.setLoading(true);
      store.setError(null);
      store.setQuickReplies([]);

      const t0 = Date.now();

      try {
        const data = await postChat({
          query: text.trim(),
          thread_id: store.threadId,
          context: { domain: "retail" },
        });

        const latencyMs = Date.now() - t0;

        // Store thread_id for multi-turn
        if (data.thread_id) {
          store.setThreadId(data.thread_id);
        }

        const kind = classifyResponse(data);
        const displayText = extractDisplayText(data);
        const workflowState = extractWorkflowSnapshot(data);
        const aopData = extractAopSnapshot(data);

        // Agent name resolution
        const agentId = data.agent_id || "";
        const agentMeta = store.agents[agentId];
        const display = getAgentDisplay(agentId, agentMeta?.type);

        const agentMsg: ChatMessage = {
          id: makeId(),
          role: "agent",
          content: displayText,
          timestamp: Date.now(),
          agentId,
          agentName: display.label,
          responseKind: kind,
          raw: data,
          routerPlan: data.router_plan,
          voiceChat: data.chat,
          latencyMs,
          workflowState,
          aopData,
        };
        store.addMessage(agentMsg);

        // Update workflow tracking
        if (workflowState) {
          store.setActiveWorkflow(workflowState.terminal ? null : workflowState);
        }

        // Quick replies
        if (data.chat?.quick_replies?.length) {
          store.setQuickReplies(data.chat.quick_replies);
        }

        // Auto-select for debug
        store.setSelectedMessageId(agentMsg.id);
      } catch (err) {
        const errorMsg: ChatMessage = {
          id: makeId(),
          role: "system",
          content:
            err instanceof Error
              ? err.message
              : "Failed to reach the runtime service.",
          timestamp: Date.now(),
          responseKind: "error",
        };
        store.addMessage(errorMsg);
        store.setError(errorMsg.content);
      } finally {
        store.setLoading(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [store.threadId, store.isLoading, store.agents]
  );

  return { sendMessage };
}
