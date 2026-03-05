"use client";

import type { ChatMessage } from "@/types/chat";
import { AgentAvatar } from "./AgentAvatar";
import { WorkflowProgressBar } from "@/components/workflow/WorkflowProgressBar";
import { SlotSummary } from "@/components/workflow/SlotSummary";
import { HierarchicalResults } from "@/components/orchestration/HierarchicalResults";
import { GovernanceBadge } from "@/components/debug/GovernancePanel";
import { getAgentDisplay } from "@/lib/constants";
import {
  Shield,
  AlertTriangle,
  ArrowRight,
  HelpCircle,
  Route,
  Brain,
  Clock,
  Layers,
  ChevronRight,
  ChevronDown,
  Info,
  ListChecks,
} from "lucide-react";
import { useChatStore } from "@/store/chatStore";
import { useState } from "react";

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const debugMode = useChatStore((s) => s.debugMode);
  const setSelectedMessageId = useChatStore((s) => s.setSelectedMessageId);
  const [taskPlanOpen, setTaskPlanOpen] = useState(false);
  const [remainingOpen, setRemainingOpen] = useState(false);
  const [routerPlanOpen, setRouterPlanOpen] = useState(false);
  const [metadataOpen, setMetadataOpen] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="max-w-[85%] sm:max-w-[70%] rounded-xl rounded-br-sm bg-blue-500 px-4 py-2.5 text-white">
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  if (message.role === "system") {
    return (
      <div className="flex justify-center mb-4">
        <div className="max-w-[90%] sm:max-w-[80%] rounded-lg bg-red-50 border border-red-200 px-4 py-2.5">
          <p className="text-sm text-red-700">{message.content}</p>
        </div>
      </div>
    );
  }

  // Agent message
  const display = message.agentId
    ? getAgentDisplay(message.agentId)
    : { icon: "Bot", label: message.agentName || "Agent" };

  const kind = message.responseKind;
  const raw = message.raw;

  // Bubble color per kind
  let bubbleClass = "bg-slate-100";
  let KindIcon: React.ElementType | null = null;

  if (kind === "guardrails_block") {
    bubbleClass = "bg-red-50 border border-red-200";
    KindIcon = Shield;
  } else if (kind === "clarify") {
    bubbleClass = "bg-amber-50 border border-amber-200";
    KindIcon = HelpCircle;
  } else if (kind === "delegate") {
    bubbleClass = "bg-orange-50 border border-orange-200";
    KindIcon = ArrowRight;
  } else if (kind === "error") {
    bubbleClass = "bg-red-50 border border-red-200";
    KindIcon = AlertTriangle;
  } else if (kind === "workflow_complete") {
    bubbleClass = "bg-green-50 border border-green-200";
  }

  // Extract data for inline explainability
  const routerPlan = message.routerPlan;
  const solvability = raw?.solvability;
  const delegateInfo = raw?.delegate;
  const agentId = raw?.agent_id || message.agentId;
  const score = raw?.score;
  const mapper = raw?.mapper;
  const history = raw?.history;

  return (
    <div className="flex items-start gap-2 mb-4 group">
      <div
        className="max-w-[92%] sm:max-w-[80%]"
        onClick={(e) => {
          // Only select for debug sidebar if clicking the bubble itself,
          // not a nested interactive element (button, summary, a).
          if (debugMode && !(e.target as HTMLElement).closest("details, button, summary")) {
            setSelectedMessageId(message.id);
          }
        }}
      >
        <AgentAvatar iconName={display.icon} label={display.label} />

        <div
          className={`rounded-xl rounded-tl-sm px-4 py-2.5 ${bubbleClass} ${debugMode ? "cursor-pointer hover:ring-2 hover:ring-blue-300" : ""}`}
        >
          {KindIcon && (
            <div className="flex items-center gap-1.5 mb-1">
              <KindIcon size={14} className="text-slate-500" />
              <span className="text-xs font-medium text-slate-500 uppercase">
                {kind === "guardrails_block"
                  ? "Blocked"
                  : kind === "clarify"
                    ? "Needs Clarification"
                    : kind === "delegate"
                      ? "Delegated"
                      : "Error"}
              </span>
            </div>
          )}

          <p className="text-sm whitespace-pre-wrap text-slate-800">
            {message.content}
          </p>

          {/* Agent ID + Score bar (for FAQ/answer responses) */}
          {kind === "faq" && (agentId || score != null) && (
            <details className="mt-2">
              <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                <Layers size={12} />
                Agent &amp; Score
              </summary>
              <div className="mt-1 flex items-center gap-3 rounded-md bg-white/60 px-2.5 py-1.5 text-xs">
                {agentId && (
                  <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 font-medium text-blue-700">
                    <Layers size={10} />
                    {agentId}
                  </span>
                )}
                {score != null && (
                  <span className="flex items-center gap-1.5 text-slate-600">
                    Retrieval:
                    <span className="font-mono font-medium">
                      {Number(score).toFixed(3)}
                    </span>
                    <div className="h-1.5 w-16 rounded-full bg-slate-200">
                      <div
                        className={`h-1.5 rounded-full ${Number(score) >= 0.8 ? "bg-green-500" : Number(score) >= 0.5 ? "bg-amber-500" : "bg-red-400"}`}
                        style={{ width: `${Math.round(Number(score) * 100)}%` }}
                      />
                    </div>
                  </span>
                )}
              </div>
            </details>
          )}

          {/* Delegation details */}
          {kind === "delegate" && delegateInfo && (
            <details className="mt-2">
              <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                <ArrowRight size={12} />
                Delegation Details
              </summary>
              <div className="mt-1 rounded-md border border-orange-200 bg-orange-50 px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <ChevronRight size={12} className="text-orange-500" />
                  <span className="text-slate-600">Suggested handler:</span>
                  <span className="rounded bg-orange-200 px-1.5 py-0.5 font-medium text-orange-800">
                    {delegateInfo.suggested_type || "unknown"}
                  </span>
                </div>
                {delegateInfo.suggested_id && (
                  <p className="mt-1 text-slate-500">
                    Target:{" "}
                    <span className="font-medium text-slate-700">
                      {delegateInfo.suggested_id}
                    </span>
                  </p>
                )}
              </div>
            </details>
          )}

          {/* Solvability analysis (for clarify/delegate responses) */}
          {(kind === "clarify" || kind === "delegate") && solvability && (
            <details className="mt-2">
              <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                <Brain size={12} />
                Solvability Analysis
              </summary>
              <div className="mt-1.5 rounded-md bg-white/80 p-2 text-xs">
                <SolvabilityInline data={solvability} />
              </div>
            </details>
          )}

          {/* Citations */}
          {kind === "faq" && (raw?.citations?.length ?? 0) > 0 && (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700">
                Sources ({raw!.citations!.length})
              </summary>
              <ul className="mt-1 space-y-0.5 text-xs text-slate-500">
                {raw!.citations!.map(
                  (c: Record<string, unknown>, i: number) => (
                    <li key={i}>
                      {String(c.question || c.source || `Source ${i + 1}`)}
                    </li>
                  )
                )}
              </ul>
            </details>
          )}

          {/* Workflow progress */}
          {(kind === "workflow_progress" || kind === "workflow_complete") &&
            message.workflowState && (
              <details className="mt-2">
                <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                  <ListChecks size={12} />
                  Workflow Progress
                </summary>
                <div className="mt-1">
                  <WorkflowProgressBar snapshot={message.workflowState} />
                  <SlotSummary
                    slots={message.workflowState.slots}
                    missingSlots={message.workflowState.missingSlots}
                  />
                </div>
              </details>
            )}

          {/* Workflow mapper (expandable) */}
          {(kind === "workflow_progress" || kind === "workflow_complete") &&
            mapper &&
            Object.keys(mapper).length > 0 && (
              <details className="mt-2">
                <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                  <Brain size={12} />
                  Workflow Mapper
                </summary>
                <div className="mt-1.5 rounded-md bg-white/80 p-2">
                  <pre className="overflow-x-auto text-[10px] text-slate-600">
                    {JSON.stringify(mapper, null, 2)}
                  </pre>
                </div>
              </details>
            )}

          {/* Workflow history (expandable) */}
          {(kind === "workflow_progress" || kind === "workflow_complete") &&
            Array.isArray(history) &&
            history.length > 0 && (
              <details className="mt-2">
                <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                  <Clock size={12} />
                  State History ({history.length} transitions)
                </summary>
                <div className="mt-1.5 space-y-1">
                  {history.map(
                    (
                      h: {
                        state: string;
                        actions?: string[];
                        timestamp?: number;
                      },
                      i: number
                    ) => (
                      <div
                        key={i}
                        className="flex items-start gap-2 rounded bg-white/80 px-2 py-1 text-[10px]"
                      >
                        <span className="shrink-0 rounded bg-slate-200 px-1.5 py-0.5 font-mono font-medium text-slate-700">
                          {h.state}
                        </span>
                        {h.actions && h.actions.length > 0 && (
                          <span className="text-slate-500">
                            {h.actions.join(", ")}
                          </span>
                        )}
                      </div>
                    )
                  )}
                </div>
              </details>
            )}

          {/* Hierarchical/AOP results */}
          {kind === "hierarchical" && message.aopData && (
            <details className="mt-2">
              <summary className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700">
                <Brain size={12} />
                Orchestration Results
              </summary>
              <div className="mt-1">
                <HierarchicalResults data={message.aopData} />
              </div>
            </details>
          )}

          {/* AOP Task Menu — numbered task list for sequential execution */}
          {kind === "aop_task_menu" && message.aopTaskMenu && (
            <div className="mt-2">
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setTaskPlanOpen(!taskPlanOpen); }}
                className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700"
              >
                {taskPlanOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                <Layers size={12} />
                Task Plan ({message.aopTaskMenu.taskMenu.length} tasks)
              </button>
              {taskPlanOpen && (
                <div className="mt-1 rounded-lg border border-blue-200 bg-blue-50/50 p-3">
                  {message.aopTaskMenu.taskMenu.map((t, i) => {
                    const display_desc = t.subtask
                      .replace(/^INFORMATIONAL:\s*/i, "")
                      .replace(/^ACTION:\s*/i, "")
                      .replace(/^\[INFORMATIONAL\]\s*/i, "")
                      .replace(/^\[ACTION\]\s*/i, "");
                    const agentDisplay = t.agentId ? getAgentDisplay(t.agentId) : null;
                    return (
                      <div
                        key={i}
                        className="flex items-center gap-2 py-1.5 text-sm text-slate-700"
                      >
                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-blue-200 text-xs font-bold text-blue-700">
                          {i + 1}
                        </span>
                        <span className="flex-1 truncate">{display_desc}</span>
                        {agentDisplay && (
                          <span className="rounded bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-600">
                            {agentDisplay.label}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* Remaining AOP tasks (shown for any response with remaining_subtasks) */}
          {Array.isArray(raw?.remaining_subtasks) &&
            raw!.remaining_subtasks.length > 0 && (
              <div className="mt-2">
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); setRemainingOpen(!remainingOpen); }}
                  className="flex cursor-pointer items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700"
                >
                  {remainingOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                  <ListChecks size={12} />
                  Remaining tasks ({raw!.remaining_subtasks.length})
                </button>
                {remainingOpen && (
                  <div className="mt-1 rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
                    {raw!.remaining_subtasks.map(
                      (
                        r: { index: number; subtask: string; agent_id: string | null },
                        i: number
                      ) => {
                        const desc = r.subtask
                          .replace(/^INFORMATIONAL:\s*/i, "")
                          .replace(/^ACTION:\s*/i, "")
                          .replace(/^\[INFORMATIONAL\]\s*/i, "")
                          .replace(/^\[ACTION\]\s*/i, "");
                        return (
                          <div
                            key={i}
                            className="ml-2 mt-1 text-xs text-slate-600"
                          >
                            {i + 1}. {desc}
                          </div>
                        );
                      }
                    )}
                  </div>
                )}
              </div>
            )}

        </div>

        {/* Router plan (expandable — always shown when available) */}
        {routerPlan?.candidates && routerPlan.candidates.length > 0 && (
          <div className="mt-1.5 ml-1">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setRouterPlanOpen(!routerPlanOpen); }}
              className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600"
            >
              {routerPlanOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Route size={12} />
              Router decision ({routerPlan.candidates.length} candidates)
            </button>
            {routerPlanOpen && (
            <div className="mt-1 rounded-lg border border-slate-200 bg-white p-2.5">
              <div className="mb-1.5 flex items-center gap-2 text-[10px] text-slate-400">
                <span>
                  Strategy:{" "}
                  <span className="font-medium text-slate-600">
                    {routerPlan.strategy}
                  </span>
                </span>
                <span>
                  Primary:{" "}
                  <span className="font-medium text-slate-600">
                    {routerPlan.primary}
                  </span>
                </span>
              </div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-100 text-left text-[10px] text-slate-400">
                    <th className="pb-1 pr-2 font-medium">Agent</th>
                    <th className="pb-1 pr-2 font-medium">Routing score</th>
                    <th className="pb-1 font-medium">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {routerPlan.candidates.map((c, i) => {
                    const s = Number(c.score ?? 0);
                    const pct = Math.round(s * 100);
                    return (
                      <tr
                        key={i}
                        className={`border-b border-slate-50 ${i === 0 ? "font-medium" : ""}`}
                      >
                        <td className="py-1 pr-2 text-slate-700">
                          {c.id || "-"}
                        </td>
                        <td className="py-1 pr-2">
                          <div className="flex items-center gap-1.5">
                            <div className="h-1.5 w-12 rounded-full bg-slate-200">
                              <div
                                className="h-1.5 rounded-full bg-blue-500"
                                style={{ width: `${pct}%` }}
                              />
                            </div>
                            <span className="font-mono text-slate-600">
                              {s.toFixed(2)}
                            </span>
                          </div>
                        </td>
                        <td className="py-1 text-slate-500 max-w-[200px] truncate">
                          {c.reason || "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            )}
          </div>
        )}

        {/* Metadata row */}
        {(message.latencyMs || (kind !== "faq" && raw?.score != null) || raw?.rag_state) && (
          <div className="mt-1 ml-1">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setMetadataOpen(!metadataOpen); }}
              className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600"
            >
              {metadataOpen ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              <Info size={10} />
              Metadata
            </button>
            {metadataOpen && (
            <div className="mt-0.5 flex items-center gap-3 text-xs text-slate-400">
              {message.latencyMs && (
                <span>{(message.latencyMs / 1000).toFixed(1)}s</span>
              )}
              {kind !== "faq" && raw?.score != null && (
                <span>Match: {Number(raw.score).toFixed(2)}</span>
              )}
              {raw?.rag_state && (
                <span className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[10px]">
                  {raw.rag_state}
                </span>
              )}
            </div>
            )}
          </div>
        )}

        {/* IEEE Governance compliance badge */}
        <GovernanceBadge message={message} />
      </div>
    </div>
  );
}

/** Inline solvability analysis renderer */
function SolvabilityInline({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data);
  if (!entries.length) return null;

  // Check if it has known structure (scores, assessment, etc.)
  const hasScores =
    typeof data.score === "number" ||
    typeof data.confidence === "number" ||
    typeof data.assessment === "string";

  if (hasScores) {
    const assessment = data.assessment != null ? String(data.assessment) : null;
    const scoreVal = typeof data.score === "number" ? data.score : null;
    const confidence =
      typeof data.confidence === "number" ? data.confidence : null;
    const reason = data.reason != null ? String(data.reason) : null;

    return (
      <div className="space-y-1">
        {assessment && (
          <p className="text-slate-700">
            <span className="font-medium">Assessment:</span> {assessment}
          </p>
        )}
        {scoreVal != null && (
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Score:</span>
            <span className="font-mono font-medium text-slate-700">
              {scoreVal.toFixed(3)}
            </span>
            <div className="h-1.5 w-20 rounded-full bg-slate-200">
              <div
                className={`h-1.5 rounded-full ${scoreVal >= 0.7 ? "bg-green-500" : scoreVal >= 0.4 ? "bg-amber-500" : "bg-red-400"}`}
                style={{
                  width: `${Math.round(scoreVal * 100)}%`,
                }}
              />
            </div>
          </div>
        )}
        {confidence != null && (
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Confidence:</span>
            <span className="font-mono font-medium text-slate-700">
              {confidence.toFixed(3)}
            </span>
          </div>
        )}
        {reason && (
          <p className="text-slate-500">
            <span className="font-medium">Reason:</span> {reason}
          </p>
        )}
      </div>
    );
  }

  // Fallback: render as formatted key-value pairs
  return (
    <div className="space-y-0.5">
      {entries.map(([key, value]) => (
        <div key={key} className="flex items-start gap-2">
          <span className="shrink-0 text-slate-500">
            {key.replace(/_/g, " ")}:
          </span>
          <span className="font-medium text-slate-700">
            {typeof value === "object" ? JSON.stringify(value) : String(value)}
          </span>
        </div>
      ))}
    </div>
  );
}
