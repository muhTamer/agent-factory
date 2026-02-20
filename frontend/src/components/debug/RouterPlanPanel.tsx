"use client";

import type { ChatMessage } from "@/types/chat";

interface Props {
  message: ChatMessage;
}

export function RouterPlanPanel({ message }: Props) {
  const plan = message.routerPlan;
  if (!plan?.candidates?.length) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase text-slate-500">
        Router Plan
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-left text-slate-500">
              <th className="pb-1 pr-2 font-medium">Agent</th>
              <th className="pb-1 pr-2 font-medium">Score</th>
              <th className="pb-1 font-medium">Reason</th>
            </tr>
          </thead>
          <tbody>
            {plan.candidates.map(
              (c, i) => {
                const score = Number(c.score ?? 0);
                const pct = Math.round(score * 100);
                return (
                  <tr
                    key={i}
                    className={`border-b border-slate-100 ${
                      i === 0 ? "font-medium" : ""
                    }`}
                  >
                    <td className="py-1.5 pr-2 text-slate-700">
                      {c.id || "-"}
                    </td>
                    <td className="py-1.5 pr-2">
                      <div className="flex items-center gap-1.5">
                        <div className="h-1.5 w-16 rounded-full bg-slate-200">
                          <div
                            className="h-1.5 rounded-full bg-blue-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="font-mono text-slate-600">
                          {score.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="py-1.5 text-slate-500">
                      {c.reason || "-"}
                    </td>
                  </tr>
                );
              }
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
