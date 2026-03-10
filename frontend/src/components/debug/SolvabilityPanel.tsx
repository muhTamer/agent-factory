"use client";

import type { ChatMessage } from "@/types/chat";
import { BarChart3 } from "lucide-react";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

export function SolvabilityPanel({ message }: Props) {
  const aop = message.aopData;
  if (!aop) return null;

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
      title="Solvability Scores (AOP)"
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
          return (
            <div key={i} className="text-xs">
              <div className="mb-0.5 flex items-center justify-between">
                <span className="text-slate-700">{r.subtask}</span>
                <span className="font-mono text-slate-500 cursor-help" title={`Solvability: ${(r.solvabilityScore * 100).toFixed(1)}% — probability this subtask can be resolved by the assigned agent`}>{r.solvabilityScore.toFixed(3)}</span>
              </div>
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
