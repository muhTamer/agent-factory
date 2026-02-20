"use client";

import type { ChatMessage } from "@/types/chat";

interface Props {
  message: ChatMessage;
}

export function SolvabilityPanel({ message }: Props) {
  const aop = message.aopData;
  if (!aop) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase text-slate-500">
        Solvability Scores (AOP)
      </h3>
      <div className="space-y-1.5">
        {aop.subtaskResults.map((r, i) => {
          const pct = Math.round(r.solvabilityScore * 100);
          return (
            <div key={i} className="text-xs">
              <div className="mb-0.5 flex items-center justify-between">
                <span className="text-slate-700">{r.subtask}</span>
                <span className="font-mono text-slate-500">
                  {r.solvabilityScore.toFixed(3)}
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-slate-200">
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
        <div className="mt-2 text-xs text-slate-500">
          Coverage: {Math.round(aop.completeness.coverageRatio * 100)}%
          {aop.completeness.missing.length > 0 && (
            <span className="ml-2 text-amber-600">
              Missing: {aop.completeness.missing.join(", ")}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
