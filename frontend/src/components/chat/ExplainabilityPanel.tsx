"use client";

import { useChatStore } from "@/store/chatStore";
import { RouterPlanPanel } from "@/components/debug/RouterPlanPanel";
import { SolvabilityPanel } from "@/components/debug/SolvabilityPanel";
import { PolicyCheckPanel } from "@/components/debug/PolicyCheckPanel";
import { SourcesPanel } from "@/components/debug/SourcesPanel";
import { ReActTracePanel } from "@/components/debug/ReActTracePanel";
import { GovernancePanel } from "@/components/debug/GovernancePanel";
import { GuardrailsAdminPanel } from "@/components/debug/GuardrailsAdminPanel";
import { EstimatorTogglePanel } from "@/components/debug/EstimatorTogglePanel";
import { RawJsonViewer } from "@/components/debug/RawJsonViewer";
import { StatusSummaryStrip } from "@/components/debug/StatusSummaryStrip";
import { X, Eye, Bug } from "lucide-react";
import { useRef, useEffect, useState } from "react";
import { getAgentDisplay } from "@/lib/constants";

interface Props {
  onClose?: () => void;
}

function SectionGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h3 className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 px-1">{label}</h3>
      <div className="space-y-2">{children}</div>
    </div>
  );
}

export function ExplainabilityPanel({ onClose }: Props) {
  const messages = useChatStore((s) => s.messages);
  const selectedId = useChatStore((s) => s.selectedMessageId);
  const debugMode = useChatStore((s) => s.debugMode);
  const toggleDebugMode = useChatStore((s) => s.toggleDebugMode);

  const selected = messages.find((m) => m.id === selectedId);
  const agentDisplay = selected?.agentId ? getAgentDisplay(selected.agentId) : null;
  const timeStr = selected ? new Date(selected.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";

  return (
    <div className="flex h-full flex-col bg-slate-50/50">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-white px-4 py-3">
        <div className="flex items-center gap-2">
          <Eye size={16} className="text-blue-500" />
          <h2 className="text-base font-semibold text-slate-700 cursor-help" title="Inspect how the system processed this response — routing decisions, agent selection, governance compliance, and policy checks">Explainability</h2>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={toggleDebugMode} title="Toggle raw debug data" className={`rounded p-1.5 transition-colors ${debugMode ? "bg-blue-50 text-blue-600" : "text-slate-400 hover:bg-slate-100 hover:text-slate-600"}`}>
            <Bug size={14} />
          </button>
          {onClose && (
            <button onClick={onClose} className="rounded p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600">
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-4">
        {/* Global config panels — always visible */}
        <SectionGroup label="Configuration">
          <EstimatorTogglePanel />
        </SectionGroup>

        {!selected || selected.role !== "agent" ? (
          <div className="py-8 text-center">
            <Eye size={32} className="mx-auto mb-2 text-slate-200" />
            <p className="text-sm text-slate-400 cursor-help" title="Select any agent message in the chat to see the routing, governance, and policy details for that response">Click an agent response to inspect how it was generated.</p>
          </div>
        ) : (
          <>
            {/* Selected message indicator */}
            <div className="rounded-lg border border-slate-200 bg-white px-3 py-2.5 shadow-sm">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="font-medium text-slate-700">{agentDisplay?.label ?? selected.agentName ?? "Agent"}</span>
                <span>&middot;</span>
                <span>{timeStr}</span>
                {selected.latencyMs && (<><span>&middot;</span><span>{(selected.latencyMs / 1000).toFixed(1)}s</span></>)}
              </div>
              <p className="mt-1 line-clamp-2 text-xs text-slate-400">{selected.content}</p>
            </div>

            <StatusSummaryStrip message={selected} />

            <SectionGroup label="Routing & Orchestration">
              <RouterPlanPanel message={selected} />
              <SolvabilityPanel message={selected} />
            </SectionGroup>

            <SectionGroup label="Safety & Compliance">
              <PolicyCheckPanel message={selected} />
              <GuardrailsAdminPanel />
              <GovernancePanel message={selected} />
            </SectionGroup>

            <SectionGroup label="Evidence & Reasoning">
              <SourcesPanel message={selected} />
              <ReActTracePanel message={selected} />
            </SectionGroup>

            {debugMode && selected.raw && <RawJsonViewer data={selected.raw} />}
          </>
        )}
      </div>
    </div>
  );
}
