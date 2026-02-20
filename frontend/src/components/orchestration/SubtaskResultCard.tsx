"use client";

import { Check, X, Loader2 } from "lucide-react";
import { getAgentDisplay } from "@/lib/constants";

interface Props {
  index: number;
  subtask: string;
  agentId: string | null;
  success: boolean;
  latencyMs: number;
  solvabilityScore: number;
}

export function SubtaskResultCard({
  index,
  subtask,
  agentId,
  success,
  latencyMs,
  solvabilityScore,
}: Props) {
  const display = agentId
    ? getAgentDisplay(agentId)
    : { label: "Unknown", icon: "Bot" };

  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="text-sm font-bold text-slate-600">{index + 1}.</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          {success ? (
            <Check size={14} className="text-green-600 shrink-0" />
          ) : (
            <X size={14} className="text-red-500 shrink-0" />
          )}
          <span className="text-sm font-medium text-slate-700 truncate">
            {subtask}
          </span>
        </div>
        <div className="mt-0.5 flex gap-3 text-xs text-slate-500">
          <span>{display.label}</span>
          <span>{(latencyMs / 1000).toFixed(1)}s</span>
          <span>Score: {solvabilityScore.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
