"use client";

import { useState } from "react";
import type { ChatMessage } from "@/types/chat";
import {
  Brain,
  Search,
  Wrench,
  MessageSquare,
  AlertTriangle,
  Send,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

interface TraceStep {
  step: number;
  thought: string;
  action: string;
  action_input: Record<string, unknown>;
  observation: string;
}

interface ToolResult {
  step: number;
  tool: string;
  args: Record<string, unknown>;
  result: string;
}

const ACTION_CONFIG: Record<
  string,
  { icon: typeof Brain; color: string; label: string; bg: string; tip: string }
> = {
  retrieve_knowledge: {
    icon: Search,
    color: "text-blue-600",
    label: "Knowledge Retrieval",
    bg: "bg-blue-50 border-blue-200",
    tip: "The agent searched its knowledge base for relevant documents or policy content",
  },
  call_tool: {
    icon: Wrench,
    color: "text-amber-600",
    label: "Tool Call",
    bg: "bg-amber-50 border-amber-200",
    tip: "The agent executed an external tool (API call, database lookup, etc.) and received a result",
  },
  respond: {
    icon: Send,
    color: "text-green-600",
    label: "Response",
    bg: "bg-green-50 border-green-200",
    tip: "The agent generated its final answer to the user based on the information gathered",
  },
  ask_user: {
    icon: MessageSquare,
    color: "text-purple-600",
    label: "Ask User",
    bg: "bg-purple-50 border-purple-200",
    tip: "The agent needs additional information from the user before it can proceed to the next step",
  },
  escalate: {
    icon: AlertTriangle,
    color: "text-red-600",
    label: "Escalate",
    bg: "bg-red-50 border-red-200",
    tip: "The agent could not resolve this request and is escalating for human review",
  },
};

function tryParseJson(s: string): unknown | null {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function StepCard({
  step,
  toolResults,
  defaultOpen,
}: {
  step: TraceStep;
  toolResults: ToolResult[];
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const cfg = ACTION_CONFIG[step.action] || ACTION_CONFIG.respond;
  const Icon = cfg.icon;
  const toolResult = toolResults.find((tr) => tr.step === step.step);
  const parsedObservation = tryParseJson(step.observation);

  return (
    <div className={`rounded-lg border ${cfg.bg}`}>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left cursor-help"
        title={cfg.tip}
      >
        {open ? (
          <ChevronDown size={14} className="shrink-0 text-slate-400" />
        ) : (
          <ChevronRight size={14} className="shrink-0 text-slate-400" />
        )}
        <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-500">
          <span className="rounded-full bg-white/80 px-1.5 py-0.5 text-[10px] font-bold text-slate-400">{step.step}</span>
          <Icon size={14} className={cfg.color} />
          <span className={cfg.color}>{cfg.label}</span>
        </span>
        <span className="ml-auto truncate text-[11px] text-slate-400 max-w-[200px]">
          {step.action === "call_tool"
            ? step.action_input.tool ? String(step.action_input.tool) : ""
            : step.action === "retrieve_knowledge"
              ? step.action_input.query ? `"${String(step.action_input.query).slice(0, 40)}"` : ""
              : ""}
        </span>
      </button>

      {open && (
        <div className="space-y-2 border-t border-inherit px-3 py-2.5">
          {step.thought && (
            <div>
              <p className="mb-0.5 text-[10px] font-semibold uppercase text-slate-400 cursor-help" title="The agent's internal reasoning about what to do next — this references policy steps and gathered information">Thought</p>
              <p className="text-xs leading-relaxed text-slate-600">{step.thought}</p>
            </div>
          )}
          {step.action === "call_tool" && toolResult ? (
            <div>
              <p className="mb-0.5 text-[10px] font-semibold uppercase text-slate-400 cursor-help" title="The tool that was called and the parameters passed to it">Tool: {toolResult.tool}</p>
              {Object.keys(toolResult.args).length > 0 && (
                <div className="rounded border border-amber-100 bg-white px-2.5 py-1.5 font-mono text-[11px] text-slate-600">
                  {Object.entries(toolResult.args).map(([k, v]) => (
                    <div key={k}><span className="text-amber-700">{k}</span>: <span className="text-slate-700">{typeof v === "object" ? JSON.stringify(v) : String(v)}</span></div>
                  ))}
                </div>
              )}
            </div>
          ) : step.action === "retrieve_knowledge" && step.action_input.query ? (
            <div>
              <p className="mb-0.5 text-[10px] font-semibold uppercase text-slate-400 cursor-help" title="The search query used to find relevant documents in the knowledge base">Query</p>
              <p className="text-xs italic text-blue-600">&ldquo;{String(step.action_input.query)}&rdquo;</p>
            </div>
          ) : null}
          {step.observation && (
            <div>
              <p className="mb-0.5 text-[10px] font-semibold uppercase text-slate-400 cursor-help" title={step.action === "call_tool" ? "The data returned by the tool — this is what the agent uses to make decisions" : "The result of the agent's action"}>
                {step.action === "call_tool" ? "Tool Result" : "Observation"}
              </p>
              {parsedObservation && typeof parsedObservation === "object" ? (
                <div className="rounded border border-slate-200 bg-white px-2.5 py-1.5 font-mono text-[11px] text-slate-600 max-h-48 overflow-y-auto">
                  <pre className="whitespace-pre-wrap">{JSON.stringify(parsedObservation, null, 2)}</pre>
                </div>
              ) : (
                <div className="rounded border border-slate-200 bg-white px-2.5 py-1.5 text-xs leading-relaxed text-slate-600 max-h-48 overflow-y-auto">{step.observation}</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ReActTracePanel({ message }: Props) {
  const raw = message.raw;
  if (!raw) return null;

  const reactTrace = (raw.react_trace || []) as TraceStep[];
  const toolResults = (raw.tool_results || []) as ToolResult[];
  if (reactTrace.length === 0) return null;

  const hasEscalation = reactTrace.some((s) => s.action === "escalate");
  const actionTypes = [...new Set(reactTrace.map((s) => s.action))];
  const sectionStatus: SectionStatus = hasEscalation ? "warning" : "info";

  return (
    <CollapsibleSection
      icon={<Brain size={14} className="text-indigo-500" />}
      title="Agent Reasoning Trace"
      tooltip="Step-by-step trace of the agent's ReAct reasoning loop — shows what the agent thought, which tools it called, and what results it received"
      status={sectionStatus}
      badge={<span className="text-[10px] font-normal text-slate-400">{reactTrace.length} step{reactTrace.length !== 1 ? "s" : ""}</span>}
      collapsedSummary={
        <div className="flex flex-wrap gap-1 text-[10px]">
          {actionTypes.map((action) => {
            const cfg = ACTION_CONFIG[action];
            if (!cfg) return null;
            return (
              <span key={action} className={`rounded-full border px-2 py-0.5 font-medium ${cfg.bg} ${cfg.color}`}>
                {cfg.label}
              </span>
            );
          })}
        </div>
      }
    >
      <div className="space-y-2">
        {reactTrace.map((step, i) => (
          <StepCard key={`step-${step.step}`} step={step} toolResults={toolResults} defaultOpen={i === reactTrace.length - 1} />
        ))}
      </div>
    </CollapsibleSection>
  );
}
