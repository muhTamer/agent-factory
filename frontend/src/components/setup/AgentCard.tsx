"use client";

import type { PlanAgent } from "@/types/concierge";
import { ConfidenceBar } from "./ConfidenceBar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const STATUS_STYLE: Record<string, string> = {
  ready: "bg-green-100 text-green-800",
  partial: "bg-amber-100 text-amber-800",
  missing_docs: "bg-red-100 text-red-800",
};

const STATUS_LABEL: Record<string, string> = {
  ready: "Ready",
  partial: "Partial",
  missing_docs: "Missing Docs",
};

interface AgentCardProps {
  agent: PlanAgent;
}

export function AgentCard({ agent }: AgentCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <span>{agent.icon}</span>
            <span>{agent.display_name}</span>
          </CardTitle>
          <Badge
            className={STATUS_STYLE[agent.status] || "bg-slate-100 text-slate-600"}
            variant="secondary"
          >
            {STATUS_LABEL[agent.status] || agent.status}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        <ConfidenceBar value={agent.confidence} />

        {agent.docs_detected.length > 0 && (
          <div>
            <span className="text-xs font-medium text-slate-500">
              Detected:{" "}
            </span>
            <span className="text-xs text-slate-700">
              {agent.docs_detected.join(", ")}
            </span>
          </div>
        )}

        {agent.docs_missing.length > 0 && (
          <div>
            <span className="text-xs font-medium text-red-500">
              Missing:{" "}
            </span>
            <span className="text-xs text-red-700">
              {agent.docs_missing.join(", ")}
            </span>
          </div>
        )}

        <p className="text-xs text-slate-500">{agent.why}</p>
      </CardContent>
    </Card>
  );
}
