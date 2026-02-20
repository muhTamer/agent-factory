"use client";

import { Check, X, Loader2 } from "lucide-react";

interface Agent {
  subtask: string;
  agentId: string | null;
  success: boolean;
}

interface Props {
  agents: Agent[];
  isComplete: boolean;
}

export function ParallelAgentsIndicator({ agents, isComplete }: Props) {
  return (
    <div className="mt-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center gap-2 mb-2">
        {isComplete ? (
          <Check size={14} className="text-green-600" />
        ) : (
          <Loader2 size={14} className="animate-spin text-blue-600" />
        )}
        <span className="text-xs font-semibold text-slate-600">
          {isComplete
            ? `All ${agents.length} tasks completed`
            : `Processing ${agents.length} tasks in parallel...`}
        </span>
      </div>
      <div className="space-y-1">
        {agents.map((a, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="font-bold text-slate-500">{i + 1}.</span>
            {a.success ? (
              <Check size={12} className="text-green-600" />
            ) : isComplete ? (
              <X size={12} className="text-red-500" />
            ) : (
              <Loader2 size={12} className="animate-spin text-blue-500" />
            )}
            <span className="text-slate-600 truncate">{a.subtask}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
