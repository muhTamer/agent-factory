"use client";

import { useCallback } from "react";
import { postChat } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";
import { useThreadStore } from "@/store/threadStore";
import {
  classifyResponse,
  extractDisplayText,
  extractWorkflowSnapshot,
  extractAopSnapshot,
  extractAopTaskMenu,
  extractAopTaskResult,
} from "@/lib/classify";
import { getAgentDisplay } from "@/lib/constants";
import type { ChatMessage } from "@/types/chat";

function makeId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

export function useChat() {
  const sendMessage = useCallback(async (text: string) => {
    // Read latest state at call time (not render time)
    if (!text.trim() || useChatStore.getState().isLoading) return;

    // ── Capture which thread this request belongs to ──
    let threadLocalId = useThreadStore.getState().activeThreadId;
    if (!threadLocalId) {
      threadLocalId = useThreadStore.getState().createThread();
    }
    const capturedThread = threadLocalId;
    const capturedBackendId = useChatStore.getState().threadId;

    /** True while the user is still looking at this thread */
    const isActive = () =>
      useThreadStore.getState().activeThreadId === capturedThread;

    // ── User message ──
    const userMsg: ChatMessage = {
      id: makeId(),
      role: "user",
      content: text.trim(),
      timestamp: Date.now(),
    };

    // Always persist to threadStore under the correct thread
    useThreadStore.getState().addMessage(capturedThread, userMsg);

    // Mark this thread as loading (persists across thread switches)
    useChatStore.setState((prev) => ({
      _loadingThreads: { ...prev._loadingThreads, [capturedThread]: true },
    }));

    // Update UI only if user is still on this thread
    if (isActive()) {
      const currentMsgs = useThreadStore.getState().getMessages(capturedThread);
      useChatStore.setState({
        messages: [...currentMsgs],
        isLoading: true,
        error: null,
        quickReplies: [],
      });
    }

    // Thread title from first user message
    const allThreadMsgs = useThreadStore.getState().getMessages(capturedThread);
    if (allThreadMsgs.filter((m) => m.role === "user").length <= 1) {
      useThreadStore.getState().updateThreadMeta(capturedThread, {
        title: text.trim().slice(0, 60),
      });
    }

    const t0 = Date.now();

    try {
      const data = await postChat({
        query: text.trim(),
        thread_id: capturedBackendId,
        context: { domain: "retail" },
      });

      const latencyMs = Date.now() - t0;

      // ── Persist usage stats ──
      if (data.usage) {
        useChatStore.setState({ usage: data.usage });
      }

      // ── Persist backend thread_id (always, regardless of active thread) ──
      if (data.thread_id) {
        useThreadStore.getState().updateThreadMeta(capturedThread, {
          backendThreadId: data.thread_id,
        });
        if (isActive()) {
          useChatStore.setState({ threadId: data.thread_id });
        }
      }

      // ── Build agent message ──
      const kind = classifyResponse(data);
      const displayText = extractDisplayText(data);
      const workflowState = extractWorkflowSnapshot(data);
      const aopData = extractAopSnapshot(data);
      const aopTaskMenu = extractAopTaskMenu(data);
      const aopTaskResult = extractAopTaskResult(data);

      const agents = useChatStore.getState().agents;
      const agentId = data.agent_id || "";
      const agentMeta = agents[agentId];
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
        aopTaskMenu,
        aopTaskResult,
      };

      // Always persist to threadStore under the correct thread
      useThreadStore.getState().addMessage(capturedThread, agentMsg);
      useThreadStore.getState().updateThreadMeta(capturedThread, {
        preview: displayText.slice(0, 100),
      });

      // ── Update UI only if user is still on this thread ──
      if (isActive()) {
        const msgs = useThreadStore.getState().getMessages(capturedThread);
        useChatStore.setState({
          messages: [...msgs],
          isLoading: false,
          selectedMessageId: agentMsg.id,
          activeWorkflow: workflowState
            ? (workflowState.terminal ? null : workflowState)
            : null,
          quickReplies: data.chat?.quick_replies?.length
            ? data.chat.quick_replies
            : [],
        });
      }
    } catch (err) {
      // Surface rate-limit / usage-cap errors with a friendly message
      let errorText = "Failed to reach the runtime service.";
      if (err instanceof Error) {
        errorText = err.message;
      }
      // Try to parse structured error from 429 responses
      try {
        const parsed = JSON.parse(errorText);
        if (parsed?.detail?.message) errorText = parsed.detail.message;
      } catch {
        // not JSON — use as-is
      }

      const errorMsg: ChatMessage = {
        id: makeId(),
        role: "system",
        content: errorText,
        timestamp: Date.now(),
        responseKind: "error",
      };

      useThreadStore.getState().addMessage(capturedThread, errorMsg);

      if (isActive()) {
        const msgs = useThreadStore.getState().getMessages(capturedThread);
        useChatStore.setState({
          messages: [...msgs],
          error: errorMsg.content,
          isLoading: false,
        });
      }
    } finally {
      // Always unmark this thread from loading set
      useChatStore.setState((prev) => {
        const { [capturedThread]: _, ...rest } = prev._loadingThreads;
        return {
          _loadingThreads: rest,
          // If still on this thread, ensure isLoading reflects reality
          ...(isActive() ? { isLoading: false } : {}),
        };
      });
    }
  }, []);

  return { sendMessage };
}
