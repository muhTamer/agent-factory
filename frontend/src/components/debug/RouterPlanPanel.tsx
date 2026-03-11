"use client";

import type { ChatMessage } from "@/types/chat";
import { getAgentDisplay } from "@/lib/constants";
import { Route, Zap, Target, Clock } from "lucide-react";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

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

  const score = Number(winnerAgent?.score ?? 0);
  const pctTop = Math.round(score * 100);
  const sectionStatus: SectionStatus =
    pctTop >= 70 ? "ok" : pctTop >= 40 ? "warning" : "error";

  return (
    <CollapsibleSection
      icon={<Route size={14} className="text-slate-500" />}
      title="Router Plan"
      tooltip="The LLM router evaluates the query and ranks candidate agents by relevance, then selects a routing strategy"
      status={sectionStatus}
      collapsedSummary={
        winnerAgent ? (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Target size={11} className="text-blue-500" />
            <span className="font-medium text-slate-700">
              {winnerDisplay?.label ?? winnerAgent.id}
            </span>
            <span className="font-mono text-blue-600">{pctTop}%</span>
            <span className="text-slate-300">|</span>
            <span>{orchestrationPattern === "hierarchical_delegation" ? "Multi-agent" : "Single"}</span>
          </div>
        ) : undefined
      }
    >
      <div className="space-y-3">
      {/* Orchestration pattern */}
      <div
        className="rounded-lg bg-slate-50 px-3 py-2 cursor-help"
        title={
          orchestrationPattern === "hierarchical_delegation"
            ? "The router delegates subtasks across multiple specialized agents via AOP decomposition"
            : "The router sends the query to a single best-matched agent"
        }
      >
        <p className="text-xs text-slate-400">Orchestration Pattern</p>
        <p className="text-sm font-medium text-slate-700">
          {orchestrationPattern === "hierarchical_delegation"
            ? "Multi-agent (AOP)"
            : "Single Agent"}
        </p>
      </div>

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
            <p className="text-[10px] text-blue-400">Router Confidence</p>
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs text-slate-400">
              <th className="pb-1.5 pr-2 font-medium">Agent</th>
              <th className="pb-1.5 pr-2 font-medium cursor-help" title="The LLM router's confidence score (0-1) for how well each agent can handle this query">Score</th>
              <th className="pb-1.5 font-medium cursor-help" title="The router's explanation for why this agent was considered for the query">Reason</th>
            </tr>
          </thead>
          <tbody>
            {plan.candidates.map((c, i) => {
              const score = Number(c.score ?? 0);
              const pct = Math.round(score * 100);
              const display = c.id ? getAgentDisplay(c.id) : null;
              return (
                <tr key={i} className={`border-b border-slate-100 ${i === 0 ? "font-medium" : ""}`}>
                  <td className="py-2 pr-2 text-slate-700">{display?.label ?? c.id ?? "-"}</td>
                  <td className="py-2 pr-2">
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-16 rounded-full bg-slate-200">
                        <div className={`h-2 rounded-full ${i === 0 ? "bg-blue-500" : "bg-slate-400"}`} style={{ width: `${pct}%` }} />
                      </div>
                      <span className="font-mono text-sm text-slate-600">{score.toFixed(2)}</span>
                    </div>
                  </td>
                  <td className="py-2 text-sm text-slate-500">{c.reason || "-"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {message.latencyMs && (
        <div className="flex items-center gap-1.5 text-xs text-slate-400 cursor-help" title="End-to-end time from sending the query to receiving the final response, including routing, agent execution, and governance checks">
          <Clock size={12} />
          <span>Total response time: {(message.latencyMs / 1000).toFixed(1)}s</span>
        </div>
      )}
      </div>
    </CollapsibleSection>
  );
}
