import type { ChatResponse } from "@/types/api";
import type { ResponseKind, WorkflowSnapshot, AopSnapshot } from "@/types/chat";

export function classifyResponse(data: ChatResponse): ResponseKind {
  if (data.error) {
    return data.reason ? "guardrails_block" : "error";
  }
  if (data.orchestration_pattern === "hierarchical_delegation") {
    return "hierarchical";
  }
  if (data.action === "clarify") return "clarify";
  if (data.action === "delegate") return "delegate";
  if (data.workflow_id || data.current_state) {
    return data.terminal === true ? "workflow_complete" : "workflow_progress";
  }
  if (data.answer !== undefined && data.answer !== null) return "faq";
  return "faq";
}

export function extractDisplayText(data: ChatResponse): string {
  // Prefer voice chat messages
  if (data.chat?.messages?.length) {
    return data.chat.messages.join("\n\n");
  }
  // Then text field
  if (data.text) return data.text;
  // Then answer
  if (data.answer) return data.answer;
  // Clarification question
  if (data.question) return data.question;
  // Error
  if (data.error) return data.error;
  // Delegation
  if (data.delegate?.reason) return `Routing to specialist: ${data.delegate.reason}`;
  return "No response.";
}

export function extractWorkflowSnapshot(
  data: ChatResponse
): WorkflowSnapshot | undefined {
  if (!data.workflow_id && !data.current_state) return undefined;

  const historyStates: string[] = [];
  if (Array.isArray(data.history)) {
    for (const h of data.history) {
      if (h.state && !historyStates.includes(h.state)) {
        historyStates.push(h.state);
      }
    }
  }
  if (data.current_state && !historyStates.includes(data.current_state)) {
    historyStates.push(data.current_state);
  }

  return {
    workflowId: data.workflow_id || "",
    currentState: data.current_state || "",
    terminal: !!data.terminal,
    slots: data.slots || {},
    missingSlots: data.missing_slots,
    allStates: historyStates,
  };
}

export function extractAopSnapshot(
  data: ChatResponse
): AopSnapshot | undefined {
  if (data.orchestration_pattern !== "hierarchical_delegation") return undefined;
  if (!Array.isArray(data.subtask_results)) return undefined;

  return {
    subtaskResults: data.subtask_results.map(
      (s: { subtask: string; agent_id: string | null; success: boolean; solvability_score: number; latency_ms: number }) => ({
        subtask: s.subtask,
        agentId: s.agent_id,
        success: s.success,
        solvabilityScore: s.solvability_score,
        latencyMs: s.latency_ms,
      })
    ),
    totalLatencyMs: data.total_latency_ms || 0,
    completeness: {
      complete: data.completeness?.complete ?? true,
      missing: data.completeness?.missing || [],
      coverageRatio: data.completeness?.coverage_ratio ?? 1,
    },
  };
}
