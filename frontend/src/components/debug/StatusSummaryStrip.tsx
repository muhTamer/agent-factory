"use client";

import type { ChatMessage } from "@/types/chat";
import {
  Shield,
  Target,
  Clock,
  Brain,
  BarChart3,
  AlertTriangle,
  CheckCircle,
} from "lucide-react";

interface Props {
  message: ChatMessage;
}

/** Compact KPI indicators for at-a-glance status overview. */
export function StatusSummaryStrip({ message }: Props) {
  const raw = message.raw;
  const plan = message.routerPlan;
  const aop = message.aopData;

  // Router confidence
  const topScore = plan?.candidates?.[0]?.score;
  const confidence = topScore != null ? Math.round(Number(topScore) * 100) : null;
  const confidenceColor =
    confidence != null
      ? confidence >= 80
        ? "text-green-600 bg-green-50 border-green-200"
        : confidence >= 50
          ? "text-amber-600 bg-amber-50 border-amber-200"
          : "text-red-600 bg-red-50 border-red-200"
      : "";

  // Policy check
  const blocked = message.responseKind === "guardrails_block";

  // Compliance rate
  const complianceRate = raw?.governance?.compliance?.compliance_rate;
  const compliancePct =
    complianceRate != null ? Math.round(Number(complianceRate) * 100) : null;
  const complianceColor =
    compliancePct != null
      ? compliancePct >= 80
        ? "text-green-600 bg-green-50 border-green-200"
        : compliancePct >= 50
          ? "text-amber-600 bg-amber-50 border-amber-200"
          : "text-red-600 bg-red-50 border-red-200"
      : "";

  // Response time
  const latency = message.latencyMs;

  // ReAct steps
  const reactSteps = (raw?.react_trace as unknown[] | undefined)?.length ?? 0;

  // AOP coverage
  const coverage = aop?.completeness?.coverageRatio;
  const coveragePct =
    coverage != null ? Math.round(Number(coverage) * 100) : null;

  // Determine if anything needs attention
  const hasAttention =
    blocked ||
    (confidence != null && confidence < 50) ||
    (compliancePct != null && compliancePct < 50) ||
    (coveragePct != null && coveragePct < 80);

  return (
    <div className="space-y-2">
      {/* Attention banner */}
      {hasAttention && (
        <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
          <AlertTriangle size={14} className="shrink-0 text-amber-500" />
          <span className="text-xs font-medium text-amber-700">
            {blocked
              ? "Response was blocked by guardrails"
              : confidence != null && confidence < 50
                ? "Low router confidence — agent match may be poor"
                : compliancePct != null && compliancePct < 50
                  ? "Low IEEE compliance — review governance gaps"
                  : "Incomplete coverage — some subtasks unresolved"}
          </span>
        </div>
      )}

      {/* KPI pills */}
      <div className="flex flex-wrap gap-1.5">
        {confidence != null && (
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${confidenceColor}`}
            title={`Router confidence: ${confidence}% — how well the selected agent matches the query`}
          >
            <Target size={12} />
            {confidence}%
          </div>
        )}

        <div
          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
            blocked
              ? "text-red-600 bg-red-50 border-red-200"
              : "text-green-600 bg-green-50 border-green-200"
          }`}
          title={blocked ? "Response was blocked by the guardrail policy" : "Response passed all guardrail checks"}
        >
          {blocked ? <Shield size={12} /> : <CheckCircle size={12} />}
          {blocked ? "Blocked" : "Passed"}
        </div>

        {compliancePct != null && (
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${complianceColor}`}
            title={`IEEE compliance: ${compliancePct}% of standard requirements satisfied`}
          >
            <Shield size={12} />
            IEEE {compliancePct}%
          </div>
        )}

        {latency != null && (
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
              latency > 5000
                ? "text-amber-600 bg-amber-50 border-amber-200"
                : "text-slate-600 bg-slate-50 border-slate-200"
            }`}
            title={`End-to-end response time: ${(latency / 1000).toFixed(1)}s`}
          >
            <Clock size={12} />
            {(latency / 1000).toFixed(1)}s
          </div>
        )}

        {reactSteps > 0 && (
          <div
            className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-600"
            title={`Agent reasoning chain: ${reactSteps} step${reactSteps !== 1 ? "s" : ""}`}
          >
            <Brain size={12} />
            {reactSteps} step{reactSteps !== 1 ? "s" : ""}
          </div>
        )}

        {coveragePct != null && (
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${
              coveragePct >= 80
                ? "text-green-600 bg-green-50 border-green-200"
                : coveragePct >= 50
                  ? "text-amber-600 bg-amber-50 border-amber-200"
                  : "text-red-600 bg-red-50 border-red-200"
            }`}
            title={`AOP coverage: ${coveragePct}% of subtasks resolved`}
          >
            <BarChart3 size={12} />
            {coveragePct}% covered
          </div>
        )}
      </div>
    </div>
  );
}
