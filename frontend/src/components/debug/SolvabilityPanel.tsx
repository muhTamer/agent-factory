"use client";

import type { ChatMessage } from "@/types/chat";
import { BarChart3 } from "lucide-react";
import { getAgentDisplay } from "@/lib/constants";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

export function SolvabilityPanel({ message }: Props) {
  const aop = message.aopData;
  const taskMenu = message.aopTaskMenu;
  const taskResult = message.aopTaskResult;

  // Show for any AOP pattern
  if (!aop && !taskMenu && !taskResult) return null;

  // ── Completed AOP (hierarchical_delegation) ──
  if (aop) {
    const coveragePct = aop.completeness
      ? Math.round(aop.completeness.coverageRatio * 100)
      : null;
    const failedCount = aop.subtaskResults.filter((r) => !r.success).length;
    const sectionStatus: SectionStatus =
      failedCount > 0 || (coveragePct != null && coveragePct < 50)
        ? "error"
        : coveragePct != null && coveragePct < 80
          ? "warning"
          : "ok";

    return (
      <CollapsibleSection
        icon={<BarChart3 size={14} className="text-slate-500" />}
        title="Task Decomposition (AOP)"
        tooltip="Action-Oriented Planning (AOP) decomposes the query into subtasks and scores each for solvability — how likely the system can resolve it"
        status={sectionStatus}
        badge={
          <span className="text-[10px] font-normal text-slate-400">
            {aop.subtaskResults.length} subtask{aop.subtaskResults.length !== 1 ? "s" : ""}
          </span>
        }
        collapsedSummary={
          <div className="flex items-center gap-2 text-xs text-slate-500">
            {coveragePct != null && (
              <span className="font-mono">Coverage: <span className="font-medium text-slate-700">{coveragePct}%</span></span>
            )}
            {failedCount > 0 && (
              <span className="text-red-500 font-medium">{failedCount} failed</span>
            )}
          </div>
        }
      >
        <div className="space-y-1.5">
          {aop.subtaskResults.map((r, i) => {
            const pct = Math.round(r.solvabilityScore * 100);
            const agentLabel = r.agentId ? getAgentDisplay(r.agentId).label : null;
            return (
              <div key={i} className="text-xs">
                <div className="mb-0.5 flex items-center justify-between">
                  <span className="text-slate-700">{r.subtask}</span>
                  <span className="font-mono text-slate-500 cursor-help" title={`Solvability: ${(r.solvabilityScore * 100).toFixed(1)}% — probability this subtask can be resolved by the assigned agent`}>{r.solvabilityScore.toFixed(3)}</span>
                </div>
                {agentLabel && (
                  <p className="text-[10px] text-slate-400 mb-0.5">Handled by: {agentLabel}</p>
                )}
                <div className="h-1.5 w-full rounded-full bg-slate-200 cursor-help" title={r.success ? "Subtask completed successfully" : "Subtask could not be fully resolved"}>
                  <div className={`h-1.5 rounded-full ${r.success ? "bg-green-500" : "bg-red-400"}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        {aop.completeness && (
          <div className="mt-2 text-xs text-slate-500 cursor-help" title="Coverage ratio measures what fraction of the user's request was addressed by the subtask decomposition">
            Coverage: {Math.round(aop.completeness.coverageRatio * 100)}%
            {aop.completeness.missing.length > 0 && (
              <span className="ml-2 text-amber-600 cursor-help" title="Aspects of the user's request that were not covered by any subtask">
                Missing: {aop.completeness.missing.join(", ")}
              </span>
            )}
          </div>
        )}
      </CollapsibleSection>
    );
  }

  // ── Task result (aop_task_result) — single subtask executed ──
  if (taskResult) {
    const ex = taskResult.executedSubtask;
    const pct = Math.round(ex.solvabilityScore * 100);
    const agentLabel = ex.agentId ? getAgentDisplay(ex.agentId).label : "Unknown";
    const remaining = taskResult.remainingSubtasks.length;

    return (
      <CollapsibleSection
        icon={<BarChart3 size={14} className="text-slate-500" />}
        title="Task Execution (AOP)"
        tooltip="This response is a single subtask result from the AOP decomposition plan — the system is executing tasks sequentially"
        status={ex.success ? "ok" : "error"}
        badge={
          <span className="text-[10px] font-normal text-slate-400">
            {remaining > 0 ? `${remaining} remaining` : "last task"}
          </span>
        }
        collapsedSummary={
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <span className="font-medium text-slate-700">{agentLabel}</span>
            <span className={ex.success ? "text-green-600" : "text-red-500"}>{ex.success ? "completed" : "failed"}</span>
          </div>
        }
      >
        {/* Executed subtask */}
        <div className="text-xs space-y-1">
          <div className="mb-0.5 flex items-center justify-between">
            <span className="text-slate-700">{ex.subtask}</span>
            <span
              className="font-mono text-slate-500 cursor-help"
              title={`Solvability: ${(ex.solvabilityScore * 100).toFixed(1)}%`}
            >
              {ex.solvabilityScore.toFixed(3)}
            </span>
          </div>
          <p className="text-[10px] text-slate-400">Handled by: {agentLabel}</p>
          <div
            className="h-1.5 w-full rounded-full bg-slate-200 cursor-help"
            title={ex.success ? "Subtask completed successfully" : "Subtask could not be fully resolved"}
          >
            <div
              className={`h-1.5 rounded-full ${ex.success ? "bg-green-500" : "bg-red-400"}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>

        {/* Remaining subtasks */}
        {taskResult.remainingSubtasks.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-[10px] font-semibold uppercase text-slate-400">Remaining Tasks</p>
            {taskResult.remainingSubtasks.map((r) => {
              const rAgent = r.agentId ? getAgentDisplay(r.agentId).label : "Unassigned";
              return (
                <div key={r.index} className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="font-mono text-slate-400">#{r.index}</span>
                  <span className="text-slate-600">{r.subtask}</span>
                  <span className="text-[10px] text-slate-400 ml-auto">{rAgent}</span>
                </div>
              );
            })}
          </div>
        )}

        {taskResult.planQuery && (
          <p className="mt-2 text-xs text-slate-500 cursor-help" title="The original multi-intent query that triggered AOP decomposition">
            <span className="font-medium">Plan query:</span>{" "}
            <span className="italic">&ldquo;{taskResult.planQuery}&rdquo;</span>
          </p>
        )}
      </CollapsibleSection>
    );
  }

  // ── Task menu (aop_task_menu) — initial decomposition plan ──
  const tasks = taskMenu!.taskMenu;

  return (
    <CollapsibleSection
      icon={<BarChart3 size={14} className="text-slate-500" />}
      title="Task Decomposition (AOP)"
      tooltip="AOP has decomposed this multi-intent query into separate subtasks, each assigned to a specialized agent with a solvability score"
      status="info"
      badge={
        <span className="text-[10px] font-normal text-slate-400">
          {tasks.length} task{tasks.length !== 1 ? "s" : ""}
        </span>
      }
      collapsedSummary={
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>{tasks.length} tasks planned</span>
        </div>
      }
    >
      <div className="space-y-1.5">
        {tasks.map((t) => {
          const pct = Math.round(t.solvabilityScore * 100);
          const agentLabel = t.agentId ? getAgentDisplay(t.agentId).label : "Unassigned";
          return (
            <div key={t.index} className="text-xs">
              <div className="mb-0.5 flex items-center justify-between">
                <span className="text-slate-700">
                  <span className="font-mono text-slate-400 mr-1">#{t.index}</span>
                  {t.subtask}
                </span>
                <span
                  className="font-mono text-slate-500 cursor-help"
                  title={`Solvability: ${(t.solvabilityScore * 100).toFixed(1)}% — predicted likelihood this subtask can be resolved`}
                >
                  {t.solvabilityScore.toFixed(3)}
                </span>
              </div>
              <p className="text-[10px] text-slate-400 mb-0.5">Assigned to: {agentLabel}</p>
              <div
                className="h-1.5 w-full rounded-full bg-slate-200 cursor-help"
                title={`Solvability: ${pct}%`}
              >
                <div
                  className={`h-1.5 rounded-full ${pct >= 70 ? "bg-blue-500" : pct >= 40 ? "bg-amber-400" : "bg-red-400"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {taskMenu!.planQuery && (
        <p className="mt-2 text-xs text-slate-500 cursor-help" title="The original multi-intent query that triggered AOP decomposition">
          <span className="font-medium">Plan query:</span>{" "}
          <span className="italic">&ldquo;{taskMenu!.planQuery}&rdquo;</span>
        </p>
      )}
    </CollapsibleSection>
  );
}
