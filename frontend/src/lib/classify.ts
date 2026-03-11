import type { ChatResponse } from "@/types/api";
import type { ResponseKind, WorkflowSnapshot, AopSnapshot, AopTaskMenuSnapshot, AopTaskResultSnapshot } from "@/types/chat";

export function classifyResponse(data: ChatResponse): ResponseKind {
  if (data.error) {
    return data.reason ? "guardrails_block" : "error";
  }
  if (data.orchestration_pattern === "aop_task_menu") {
    return "aop_task_menu";
  }
  if (data.orchestration_pattern === "aop_task_result") {
    return "aop_task_result";
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

export function extractAopTaskMenu(
  data: ChatResponse
): AopTaskMenuSnapshot | undefined {
  if (data.orchestration_pattern !== "aop_task_menu") return undefined;
  if (!Array.isArray(data.task_menu)) return undefined;

  return {
    taskMenu: data.task_menu.map(
      (t: { index: number; subtask: string; agent_id: string | null; solvability_score: number }) => ({
        index: t.index,
        subtask: t.subtask,
        agentId: t.agent_id,
        solvabilityScore: t.solvability_score,
      })
    ),
    planQuery: data.plan_query || "",
  };
}

export function extractAopTaskResult(
  data: ChatResponse
): AopTaskResultSnapshot | undefined {
  if (data.orchestration_pattern !== "aop_task_result") return undefined;
  const ex = data.executed_subtask;
  if (!ex) return undefined;

  return {
    executedSubtask: {
      subtask: ex.subtask || "",
      agentId: ex.agent_id ?? null,
      success: !!ex.success,
      solvabilityScore: ex.solvability_score ?? 0,
      latencyMs: ex.latency_ms ?? 0,
    },
    remainingSubtasks: Array.isArray(data.remaining_subtasks)
      ? data.remaining_subtasks.map(
          (r: { index: number; subtask: string; agent_id: string | null }) => ({
            index: r.index,
            subtask: r.subtask,
            agentId: r.agent_id,
          })
        )
      : [],
    planQuery: data.plan_query || "",
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
