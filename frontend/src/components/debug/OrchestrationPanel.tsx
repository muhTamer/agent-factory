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
  const strategy = raw.strategy || raw.orchestration_strategy || "-";
  const agentId = raw.agent_id || message.agentId || "-";
  const confidence = raw.confidence ?? raw.score;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase text-slate-500">
        Orchestration
      </h3>
      <div className="space-y-2 text-sm">
        <Row label="Pattern">
          <Badge variant="secondary" className="text-xs">
            {String(pattern).replace(/_/g, " ")}
          </Badge>
        </Row>
        <Row label="Strategy">{String(strategy)}</Row>
        <Row label="Primary Agent">{String(agentId)}</Row>
        {confidence != null && (
          <Row label="Confidence">
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
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-800">{children}</span>
    </div>
  );
}
