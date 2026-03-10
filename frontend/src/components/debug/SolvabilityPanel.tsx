"use client";

import type { ChatMessage } from "@/types/chat";
import { BarChart3 } from "lucide-react";
import { CollapsibleSection } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

export function SolvabilityPanel({ message }: Props) {
  const aop = message.aopData;
  if (!aop) return null;

  return (
    <CollapsibleSection
      icon={<BarChart3 size={14} className="text-slate-500" />}
      title="Solvability Scores (AOP)"
      tooltip="Action-Oriented Planning (AOP) decomposes the query into subtasks and scores each for solvability — how likely the system can resolve it"
    >
      <div className="space-y-1.5">
        {aop.subtaskResults.map((r, i) => {
          const pct = Math.round(r.solvabilityScore * 100);
          return (
            <div key={i} className="text-xs">
              <div className="mb-0.5 flex items-center justify-between">
                <span className="text-slate-700">{r.subtask}</span>
                <span
                  className="font-mono text-slate-500 cursor-help"
                  title={`Solvability: ${(r.solvabilityScore * 100).toFixed(1)}% — probability this subtask can be resolved by the assigned agent`}
                >
                  {r.solvabilityScore.toFixed(3)}
                </span>
              </div>
              <div
                className="h-1.5 w-full rounded-full bg-slate-200 cursor-help"
                title={r.success ? "Subtask completed successfully" : "Subtask could not be fully resolved"}
              >
                <div
                  className={`h-1.5 rounded-full ${
                    r.success ? "bg-green-500" : "bg-red-400"
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
      {aop.completeness && (
        <div
          className="mt-2 text-xs text-slate-500 cursor-help"
          title="Coverage ratio measures what fraction of the user's request was addressed by the subtask decomposition"
        >
          Coverage: {Math.round(aop.completeness.coverageRatio * 100)}%
          {aop.completeness.missing.length > 0 && (
            <span
              className="ml-2 text-amber-600 cursor-help"
              title="Aspects of the user's request that were not covered by any subtask"
            >
              Missing: {aop.completeness.missing.join(", ")}
            </span>
          )}
        </div>
      )}
    </CollapsibleSection>
  );
}
