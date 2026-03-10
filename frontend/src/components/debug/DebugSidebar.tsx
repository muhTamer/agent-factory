"use client";

import { useChatStore } from "@/store/chatStore";
import { RouterPlanPanel } from "./RouterPlanPanel";
import { SolvabilityPanel } from "./SolvabilityPanel";
import { PolicyCheckPanel } from "./PolicyCheckPanel";
import { SourcesPanel } from "./SourcesPanel";
import { ReActTracePanel } from "./ReActTracePanel";
import { GovernancePanel } from "./GovernancePanel";
import { RawJsonViewer } from "./RawJsonViewer";
import { X } from "lucide-react";

export function DebugSidebar() {
  const debugMode = useChatStore((s) => s.debugMode);
  const toggleDebugMode = useChatStore((s) => s.toggleDebugMode);
  const messages = useChatStore((s) => s.messages);
  const selectedId = useChatStore((s) => s.selectedMessageId);

  if (!debugMode) return null;

  const selected = messages.find((m) => m.id === selectedId);

  return (
    <div className="flex h-full w-80 shrink-0 flex-col border-l bg-white">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-700">Debug</h2>
        <button
          onClick={toggleDebugMode}
          className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
        >
          <X size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {!selected ? (
          <p className="text-xs text-slate-400">
            Click an agent message to inspect it.
          </p>
        ) : (
          <>
            <RouterPlanPanel message={selected} />
            <SolvabilityPanel message={selected} />
            <PolicyCheckPanel message={selected} />
            <SourcesPanel message={selected} />
            <ReActTracePanel message={selected} />
            <GovernancePanel message={selected} />
            {selected.raw && <RawJsonViewer data={selected.raw} />}
          </>
        )}
      </div>
    </div>
  );
}
