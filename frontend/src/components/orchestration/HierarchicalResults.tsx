"use client";

import type { AopSnapshot } from "@/types/chat";
import { ParallelAgentsIndicator } from "./ParallelAgentsIndicator";
import { SubtaskResultCard } from "./SubtaskResultCard";
import { Clock } from "lucide-react";

interface Props {
  data: AopSnapshot;
}

export function HierarchicalResults({ data }: Props) {
  const allComplete = data.subtaskResults.every((s) => s.success !== undefined);

  return (
    <div className="mt-3 space-y-2">
      <ParallelAgentsIndicator
        agents={data.subtaskResults.map((s) => ({
          subtask: s.subtask,
          agentId: s.agentId,
          success: s.success,
        }))}
        isComplete={allComplete}
      />

      <div className="rounded-lg border border-slate-200 bg-white p-3">
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

      <div className="flex items-center gap-1 text-xs text-slate-400">
        <Clock size={12} />
        <span>
          Total: {(data.totalLatencyMs / 1000).toFixed(1)}s (parallel execution)
        </span>
      </div>
    </div>
  );
}
