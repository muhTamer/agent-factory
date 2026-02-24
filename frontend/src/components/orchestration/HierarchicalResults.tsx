"use client";

import type { AopSnapshot } from "@/types/chat";
import { SubtaskResultCard } from "./SubtaskResultCard";
import { Check, Loader2, Clock } from "lucide-react";

interface Props {
  data: AopSnapshot;
}

export function HierarchicalResults({ data }: Props) {
  const total = data.subtaskResults.length;
  const allComplete = data.subtaskResults.every((s) => s.success !== undefined);
  const succeeded = data.subtaskResults.filter((s) => s.success).length;

  return (
    <details className="mt-3">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100">
        {allComplete ? (
          <Check size={13} className="shrink-0 text-green-600" />
        ) : (
          <Loader2 size={13} className="shrink-0 animate-spin text-blue-500" />
        )}
        <span className="flex-1">
          {allComplete
            ? `${succeeded}/${total} tasks completed`
            : `Processing ${total} tasks…`}
        </span>
        <span className="flex items-center gap-1 font-normal text-slate-400">
          <Clock size={11} />
          {(data.totalLatencyMs / 1000).toFixed(1)}s
        </span>
      </summary>

      <div className="mt-1 rounded-lg border border-slate-200 bg-white p-3">
        {data.subtaskResults.map((s, i) => (
          <SubtaskResultCard
            key={i}
            index={i}
            subtask={s.subtask}
            agentId={s.agentId}
            success={s.success}
            latencyMs={s.latencyMs}
            solvabilityScore={s.solvabilityScore}
          />
        ))}
      </div>
    </details>
  );
}
