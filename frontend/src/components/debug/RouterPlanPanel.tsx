"use client";

import type { ChatMessage } from "@/types/chat";
import { getAgentDisplay } from "@/lib/constants";
import { Route, Zap, Target, Clock } from "lucide-react";

interface Props {
  message: ChatMessage;
}

export function RouterPlanPanel({ message }: Props) {
  const plan = message.routerPlan;
  const raw = message.raw;
  if (!plan?.candidates?.length) return null;

  const orchestrationPattern = raw?.orchestration_pattern ?? "direct";
  const winnerAgent = plan.candidates[0];
  const winnerDisplay = winnerAgent?.id
    ? getAgentDisplay(winnerAgent.id)
    : null;

  return (
    <div className="space-y-3">
      <h3
        className="text-sm font-semibold uppercase text-slate-500 flex items-center gap-1.5 cursor-help"
        title="The LLM router evaluates the query and ranks candidate agents by relevance, then selects a routing strategy"
      >
        <Route size={14} />
        Router Plan
      </h3>

      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-2">
        <div
          className="rounded-lg bg-slate-50 px-3 py-2 cursor-help"
          title="'single' runs only the top agent; 'fanout' runs all candidates in parallel and picks the best scoring response"
        >
          <p className="text-xs text-slate-400">Strategy</p>
          <p className="text-sm font-medium text-slate-700 capitalize">
            {plan.strategy}
          </p>
        </div>
        <div
          className="rounded-lg bg-slate-50 px-3 py-2 cursor-help"
          title="'Direct' routes to a single agent; 'Multi-agent' delegates subtasks across multiple specialized agents"
        >
          <p className="text-xs text-slate-400">Pattern</p>
          <p className="text-sm font-medium text-slate-700">
            {orchestrationPattern === "hierarchical_delegation"
              ? "Multi-agent"
              : "Direct"}
          </p>
        </div>
      </div>

      {/* Winner highlight */}
      {winnerAgent && (
        <div
          className="flex items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2.5 cursor-help"
          title="The agent chosen to handle this query, selected based on the highest router confidence score"
        >
          <Target size={16} className="shrink-0 text-blue-500" />
          <div className="min-w-0 flex-1">
            <p className="text-xs text-blue-500">Selected agent</p>
            <p className="text-sm font-semibold text-blue-700">
              {winnerDisplay?.label ?? winnerAgent.id}
            </p>
          </div>
          <div className="text-right">
            <p
              className="text-lg font-bold text-blue-600 cursor-help"
              title={`Router confidence: ${(Number(winnerAgent.score) * 100).toFixed(1)}% — how well this agent matches the query intent`}
            >
              {(Number(winnerAgent.score) * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      )}

      {/* Candidates table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-400">
              <th className="pb-1.5 pr-2 font-medium">Agent</th>
              <th
                className="pb-1.5 pr-2 font-medium cursor-help"
                title="The LLM router's confidence score (0-1) for how well each agent can handle this query"
              >
                Score
              </th>
              <th
                className="pb-1.5 font-medium cursor-help"
                title="The router's explanation for why this agent was considered for the query"
              >
                Reason
              </th>
            </tr>
          </thead>
          <tbody>
            {plan.candidates.map((c, i) => {
              const score = Number(c.score ?? 0);
              const pct = Math.round(score * 100);
              const display = c.id ? getAgentDisplay(c.id) : null;
              return (
                <tr
                  key={i}
                  className={`border-b border-slate-100 ${
                    i === 0 ? "font-medium" : ""
                  }`}
                >
                  <td className="py-2 pr-2 text-slate-700">
                    {display?.label ?? c.id ?? "-"}
                  </td>
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-16 rounded-full bg-slate-200">
                        <div
                          className={`h-2 rounded-full ${
                            i === 0 ? "bg-blue-500" : "bg-slate-400"
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="font-mono text-sm text-slate-600">
                        {score.toFixed(2)}
                      </span>
                    </div>
                  </td>
                  <td className="py-2 text-sm text-slate-500">
                    {c.reason || "-"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Latency */}
      {message.latencyMs && (
        <div
          className="flex items-center gap-1.5 text-xs text-slate-400 cursor-help"
          title="End-to-end time from sending the query to receiving the final response, including routing, agent execution, and governance checks"
        >
          <Clock size={12} />
          <span>
            Total response time: {(message.latencyMs / 1000).toFixed(1)}s
          </span>
        </div>
      )}
    </div>
  );
}
