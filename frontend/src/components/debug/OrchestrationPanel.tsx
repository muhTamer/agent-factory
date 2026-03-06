"use client";

import type { ChatMessage } from "@/types/chat";
import { Badge } from "@/components/ui/badge";

interface Props {
  message: ChatMessage;
}

export function OrchestrationPanel({ message }: Props) {
  const raw = message.raw;
  if (!raw) return null;

  const pattern = raw.orchestration_pattern || raw.pattern || "single_routing";
  const strategy = raw.router_plan?.strategy || raw.strategy || raw.orchestration_strategy || "-";
  const agentId = raw.agent_id || message.agentId || "-";
  const confidence = raw.confidence ?? raw.score;

  return (
    <div className="space-y-3">
      <h3
        className="text-xs font-semibold uppercase text-slate-500 cursor-help"
        title="How the runtime orchestrated the multi-agent pipeline for this request"
      >
        Orchestration
      </h3>
      <div className="space-y-2 text-sm">
        <Row
          label="Pattern"
          tooltip="The orchestration pattern used: 'direct' routes to a single agent, 'hierarchical delegation' breaks the query into subtasks handled by multiple agents, 'aop task menu' presents an action-oriented plan"
        >
          <Badge variant="secondary" className="text-xs">
            {String(pattern).replace(/_/g, " ")}
          </Badge>
        </Row>
        <Row
          label="Strategy"
          tooltip="'single' executes only the top-ranked agent; 'fanout' runs all candidates and picks the best response by score"
        >
          {String(strategy)}
        </Row>
        <Row
          label="Primary Agent"
          tooltip="The agent selected by the router as the best match for this query"
        >
          {String(agentId)}
        </Row>
        {confidence != null && (
          <Row
            label="Confidence"
            tooltip="The router's confidence score (0-1) for the selected agent's ability to handle this query"
          >
            <span className="font-mono">
              {Number(confidence).toFixed(3)}
            </span>
          </Row>
        )}
      </div>
    </div>
  );
}

function Row({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between">
      <span
        className={`text-slate-500 ${tooltip ? "cursor-help underline decoration-dotted decoration-slate-300" : ""}`}
        title={tooltip}
      >
        {label}
      </span>
      <span className="text-slate-800">{children}</span>
    </div>
  );
}
