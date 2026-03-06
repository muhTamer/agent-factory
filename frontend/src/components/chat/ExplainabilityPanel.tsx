"use client";

import { useChatStore } from "@/store/chatStore";
import { OrchestrationPanel } from "@/components/debug/OrchestrationPanel";
import { RouterPlanPanel } from "@/components/debug/RouterPlanPanel";
import { SolvabilityPanel } from "@/components/debug/SolvabilityPanel";
import { PolicyCheckPanel } from "@/components/debug/PolicyCheckPanel";
import { GovernancePanel } from "@/components/debug/GovernancePanel";
import { RawJsonViewer } from "@/components/debug/RawJsonViewer";
import { Separator } from "@/components/ui/separator";
import { X, Eye, Bug } from "lucide-react";
import { getAgentDisplay } from "@/lib/constants";

interface Props {
  onClose?: () => void;
}

export function ExplainabilityPanel({ onClose }: Props) {
  const messages = useChatStore((s) => s.messages);
  const selectedId = useChatStore((s) => s.selectedMessageId);
  const debugMode = useChatStore((s) => s.debugMode);
  const toggleDebugMode = useChatStore((s) => s.toggleDebugMode);

  const selected = messages.find((m) => m.id === selectedId);

  const agentDisplay = selected?.agentId
    ? getAgentDisplay(selected.agentId)
    : null;

  const timeStr = selected
    ? new Date(selected.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <Eye size={16} className="text-blue-500" />
          <h2
            className="text-base font-semibold text-slate-700 cursor-help"
            title="Inspect how the system processed this response — routing decisions, agent selection, governance compliance, and policy checks"
          >
            Explainability
          </h2>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={toggleDebugMode}
            title="Toggle raw debug data"
            className={`rounded p-1.5 transition-colors ${
              debugMode
                ? "bg-blue-50 text-blue-600"
                : "text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            }`}
          >
            <Bug size={14} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="rounded p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {!selected || selected.role !== "agent" ? (
          <div className="py-8 text-center">
            <Eye size={32} className="mx-auto mb-2 text-slate-200" />
            <p
              className="text-sm text-slate-400 cursor-help"
              title="Select any agent message in the chat to see the routing, governance, and policy details for that response"
            >
              Click an agent response to inspect how it was generated.
            </p>
          </div>
        ) : (
          <>
            {/* Selected message indicator */}
            <div className="rounded-lg bg-slate-50 px-3 py-2">
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <span className="font-medium text-slate-700">
                  {agentDisplay?.label ?? selected.agentName ?? "Agent"}
                </span>
                <span>&middot;</span>
                <span>{timeStr}</span>
                {selected.latencyMs && (
                  <>
                    <span>&middot;</span>
                    <span>{(selected.latencyMs / 1000).toFixed(1)}s</span>
                  </>
                )}
              </div>
              <p className="mt-1 line-clamp-2 text-sm text-slate-400">
                {selected.content}
              </p>
            </div>

            <Separator />
            <OrchestrationPanel message={selected} />
            <Separator />
            <RouterPlanPanel message={selected} />
            <Separator />
            <SolvabilityPanel message={selected} />
            <Separator />
            <PolicyCheckPanel message={selected} />
            <Separator />
            <GovernancePanel message={selected} />
            {debugMode && selected.raw && (
              <>
                <Separator />
                <RawJsonViewer data={selected.raw} />
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
