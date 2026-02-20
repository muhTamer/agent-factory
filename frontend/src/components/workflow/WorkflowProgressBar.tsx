"use client";

import type { WorkflowSnapshot } from "@/types/chat";
import { Check, Circle, Loader2 } from "lucide-react";

interface Props {
  snapshot: WorkflowSnapshot;
}

export function WorkflowProgressBar({ snapshot }: Props) {
  const { allStates, currentState, terminal } = snapshot;

  if (!allStates.length) return null;

  const currentIdx = allStates.indexOf(currentState);

  return (
    <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50/50 p-3">
      <p className="mb-2 text-xs font-semibold text-blue-700">
        Workflow: {snapshot.workflowId}
      </p>
      <div className="flex items-center gap-1 overflow-x-auto">
        {allStates.map((state, i) => {
          const isCompleted = i < currentIdx || terminal;
          const isCurrent = i === currentIdx && !terminal;

          return (
            <div key={state} className="flex items-center">
              <div className="flex items-center gap-1">
                {isCompleted ? (
                  <Check size={14} className="text-green-600" />
                ) : isCurrent ? (
                  <Loader2 size={14} className="animate-spin text-blue-600" />
                ) : (
                  <Circle size={14} className="text-slate-300" />
                )}
                <span
                  className={`whitespace-nowrap text-xs ${
                    isCompleted
                      ? "font-medium text-green-700"
                      : isCurrent
                      ? "font-medium text-blue-700"
                      : "text-slate-400"
                  }`}
                >
                  {state.replace(/_/g, " ")}
                </span>
              </div>
              {i < allStates.length - 1 && (
                <span className="mx-1.5 text-slate-300">&rarr;</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
