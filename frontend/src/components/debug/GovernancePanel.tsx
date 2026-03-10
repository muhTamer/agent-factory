"use client";

import type { ChatMessage } from "@/types/chat";
import { Badge } from "@/components/ui/badge";
import {
  Shield,
  CheckCircle,
  XCircle,
  Eye,
  FileText,
  Code,
  Fingerprint,
  Activity,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Zap,
} from "lucide-react";
import { useState } from "react";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

/* ── IEEE Standard metadata with hyperlinks ───────────── */

const IEEE_STANDARDS: Record<string, { label: string; tooltip: string; url: string }> = {
  P3394: { label: "IEEE P3394", tooltip: "Standard for Large Language Model Agent Interface", url: "https://standards.ieee.org/ieee/3394/11377/" },
  "2894-2024": { label: "IEEE 2894-2024", tooltip: "Guide for an Architectural Framework for Explainable Artificial Intelligence", url: "https://standards.ieee.org/ieee/2894/11296/" },
  "3152-2024": { label: "IEEE 3152-2024", tooltip: "Standard for Transparency of Autonomous Systems", url: "https://standards.ieee.org/ieee/3152/11718/" },
};

function StandardLink({ std, compact }: { std: string; compact?: boolean }) {
  const info = IEEE_STANDARDS[std];
  if (!info) return <span className="text-slate-500">{std}</span>;
  return (
    <a href={info.url} target="_blank" rel="noopener noreferrer" className={`inline-flex items-center gap-0.5 text-blue-600 hover:text-blue-800 hover:underline font-medium ${compact ? "" : ""}`} title={`${info.label} — ${info.tooltip} (click to view standard)`}>
      {std}
      <ExternalLink size={compact ? 8 : 10} className="opacity-60" />
    </a>
  );
}

interface Props { message: ChatMessage; }
interface ComplianceStandard { rate: number; total: number; compliant: number; }

export function GovernancePanel({ message }: Props) {
  const governance = message.raw?.governance;
  if (!governance) {
    return (
      <CollapsibleSection icon={<Shield size={14} className="text-slate-400" />} title="IEEE Governance" tooltip="IEEE governance layer evaluates compliance, explainability, and message provenance for each response" status="neutral" defaultOpen={false}>
        <p className="text-xs text-slate-400">No governance data available for this message.</p>
      </CollapsibleSection>
    );
  }

  const compRate = Number(governance.compliance?.compliance_rate ?? 0);
  const govStatus: SectionStatus = compRate >= 0.8 ? "ok" : compRate >= 0.5 ? "warning" : "error";

  return (
    <CollapsibleSection
      icon={<Shield size={14} className="text-emerald-500" />}
      title="IEEE Governance"
      tooltip="IEEE governance layer evaluates compliance, explainability, and message provenance for each response"
      status={govStatus}
      badge={<span className={`text-[10px] font-bold ${compRate >= 0.8 ? "text-green-600" : compRate >= 0.5 ? "text-amber-600" : "text-red-600"}`}>{Math.round(compRate * 100)}%</span>}
      collapsedSummary={
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span>Compliance: <span className={`font-mono font-medium ${compRate >= 0.8 ? "text-green-600" : compRate >= 0.5 ? "text-amber-600" : "text-red-600"}`}>{Math.round(compRate * 100)}%</span></span>
          {governance.explanations && (<><span className="text-slate-300">|</span><span>{Object.keys(governance.explanations).length}/3 explainability levels</span></>)}
        </div>
      }
    >
      <div className="space-y-4">
        <ComplianceSection compliance={governance.compliance} />
        <ExplainabilitySection explanations={governance.explanations} />
        <EnvelopeSection envelope={governance.envelope} />
      </div>
    </CollapsibleSection>
  );
}

/* ── Compliance Section ──────────────────────────────── */

function ComplianceSection({ compliance }: { compliance: Record<string, unknown> }) {
  const [expanded, setExpanded] = useState(false);
  if (!compliance) return null;

  const rate = Number(compliance.compliance_rate ?? 0);
  const byStandard = (compliance.by_standard || {}) as Record<string, ComplianceStandard>;
  const results = (compliance.results || []) as Array<{ standard: string; requirement_id: string; description: string; severity: string; compliant: boolean; evidence?: string; gap?: string; }>;

  const rateColor = rate >= 0.8 ? "text-green-600" : rate >= 0.5 ? "text-amber-600" : "text-red-600";
  const rateBg = rate >= 0.8 ? "bg-green-500" : rate >= 0.5 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600 cursor-help" title="Percentage of IEEE standard requirements that this response satisfies — higher is better">Overall Compliance</span>
        <span className={`text-sm font-bold ${rateColor}`}>{(rate * 100).toFixed(0)}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200">
        <div className={`h-2 rounded-full ${rateBg} transition-all`} style={{ width: `${Math.round(rate * 100)}%` }} />
      </div>
      <div className="space-y-1.5 pt-1">
        {Object.entries(byStandard).map(([std, data]) => {
          const stdRate = typeof data === "object" && data !== null ? Number((data as ComplianceStandard).rate ?? 0) : Number(data ?? 0);
          const color = stdRate >= 0.8 ? "text-green-600" : stdRate >= 0.5 ? "text-amber-600" : "text-red-600";
          return (
            <div key={std} className="flex items-center justify-between text-xs">
              <StandardLink std={std} />
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-12 rounded-full bg-slate-200">
                  <div className={`h-1.5 rounded-full ${stdRate >= 0.8 ? "bg-green-500" : stdRate >= 0.5 ? "bg-amber-500" : "bg-red-500"}`} style={{ width: `${Math.round(stdRate * 100)}%` }} />
                </div>
                <span className={`font-mono font-medium ${color}`}>{(stdRate * 100).toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>
      {results.length > 0 && (
        <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600">
          {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          {results.length} requirements checked
        </button>
      )}
      {expanded && (
        <div className="max-h-64 space-y-1.5 overflow-y-auto rounded-md border border-slate-100 bg-slate-50 p-2">
          {results.map((r, i) => (
            <div key={i} className="flex items-start gap-1.5 text-xs">
              {r.compliant ? <CheckCircle size={12} className="mt-0.5 shrink-0 text-green-500" /> : <XCircle size={12} className="mt-0.5 shrink-0 text-red-500" />}
              <div className="flex-1">
                <a href={IEEE_STANDARDS[r.standard]?.url || "#"} target="_blank" rel="noopener noreferrer" className="font-medium text-blue-600 hover:underline" title={`${IEEE_STANDARDS[r.standard]?.label || r.standard} — ${r.description} (${r.severity})`}>{r.requirement_id}</a>
                <span className="ml-1 text-slate-600">{r.description}</span>
                {r.evidence && !r.gap && <p className="text-slate-400 mt-0.5">{r.evidence}</p>}
                {r.gap && <p className="text-red-500 mt-0.5">{r.gap}</p>}
              </div>
              <Badge variant="secondary" className="shrink-0 text-[9px] px-1.5 py-0">{r.severity}</Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Explainability Section ──────────────────────────── */

function ExplainabilitySection({ explanations }: { explanations: Record<string, Record<string, unknown>> | undefined }) {
  const [activeLevel, setActiveLevel] = useState<string | null>("summary");
  if (!explanations) return null;

  const levels = [
    { key: "summary", label: "Summary", icon: Eye, desc: "User-facing", tooltip: "Plain-language explanation suitable for end-users — what the system did and why" },
    { key: "detailed", label: "Detailed", icon: FileText, desc: "Auditor", tooltip: "Auditor-level explanation with decision rationale, agents involved, and data provenance" },
    { key: "full", label: "Full", icon: Code, desc: "Developer", tooltip: "Developer-level trace with full metrics, decision chain, and technical details" },
  ];

  return (
    <div className="space-y-2">
      <h4 className="text-[10px] font-semibold uppercase text-slate-400 flex items-center gap-1.5">Explainability (<StandardLink std="2894-2024" compact />)</h4>
      <div className="flex gap-1.5">
        {levels.map(({ key, label, icon: Icon, desc, tooltip }) => {
          const available = key in explanations;
          const active = activeLevel === key;
          return (
            <button key={key} disabled={!available} title={tooltip} onClick={() => setActiveLevel(active ? null : key)}
              className={`flex flex-1 items-center gap-2 rounded-lg border px-3 py-2.5 text-left transition-colors ${active ? "border-blue-300 bg-blue-50 text-blue-700 shadow-sm" : available ? "border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50/50" : "border-slate-100 bg-slate-50 text-slate-300 cursor-not-allowed"}`}>
              <Icon size={16} className="shrink-0" />
              <div className="min-w-0">
                <span className="text-sm font-medium block leading-tight">{label}</span>
                <span className="text-[11px] text-slate-400 block leading-tight">{desc}</span>
              </div>
            </button>
          );
        })}
      </div>
      {activeLevel && explanations[activeLevel] && <ExplanationContent data={explanations[activeLevel]} />}
    </div>
  );
}

function ExplanationContent({ data }: { data: Record<string, unknown> }) {
  const narrative = String(data.narrative || "");
  const agents = (data.agents_involved || []) as string[];
  const decisions = (data.decisions || []) as Array<Record<string, unknown>>;
  const metrics = (data.metrics || {}) as Record<string, unknown>;

  return (
    <div className="space-y-3 rounded-lg border border-slate-100 bg-white p-3">
      {narrative && <p className="text-sm leading-relaxed text-slate-700">{narrative}</p>}
      {agents.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {agents.map((a) => <Badge key={a} variant="secondary" className="text-[10px]">{a}</Badge>)}
        </div>
      )}
      {decisions.length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-semibold text-slate-500 uppercase flex items-center gap-1.5"><Zap size={12} />Decisions ({decisions.length})</h5>
          <div className="space-y-1.5">
            {decisions.map((d, i) => (
              <div key={i} className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
                <span className="text-xs font-semibold text-slate-700">{String(d.stage || d.type || `Step ${i + 1}`)}</span>
                {d.rationale != null && <p className="mt-0.5 text-xs text-slate-500">{String(d.rationale)}</p>}
              </div>
            ))}
          </div>
        </div>
      )}
      {Object.keys(metrics).filter((k) => k !== "event_log").length > 0 && (
        <div className="space-y-2">
          <h5 className="text-xs font-semibold text-slate-500 uppercase">Metrics</h5>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(metrics).filter(([k]) => k !== "event_log").map(([k, v]) => (
              <div key={k} className="rounded-lg bg-slate-50 px-3 py-2">
                <p className="text-[11px] text-slate-400 leading-tight">{k.replace(/_/g, " ")}</p>
                <p className="text-sm font-mono font-medium text-slate-700">{typeof v === "number" ? v.toFixed(3) : String(v)}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Envelope Section ────────────────────────────────── */

function EnvelopeSection({ envelope }: { envelope: Record<string, unknown> | undefined }) {
  const [expanded, setExpanded] = useState(false);
  if (!envelope) return null;

  const sender = envelope.sender as Record<string, unknown> | undefined;
  const receiver = envelope.receiver as Record<string, unknown> | undefined;
  const aiGenerated = envelope.ai_generated as boolean | undefined;
  const agentsChain = (envelope.agents_chain || []) as string[];

  return (
    <div className="space-y-2">
      <h4 className="text-[10px] font-semibold uppercase text-slate-400 flex items-center gap-1.5">Message Envelope (<StandardLink std="P3394" compact />)</h4>
      <div className="space-y-1.5 text-[11px]">
        <div className="flex items-center gap-2">
          <Fingerprint size={12} className="text-slate-400" />
          <span className="text-slate-500 cursor-help" title="Whether this response was generated by an AI model, as required by transparency standards">AI-generated:</span>
          {aiGenerated ? <Badge variant="secondary" className="bg-purple-100 text-purple-700 text-[9px]">Yes</Badge> : <Badge variant="secondary" className="text-[9px]">No</Badge>}
        </div>
        {sender && (
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-slate-400" />
            <span className="text-slate-500 cursor-help" title="The agent that produced this response">Sender:</span>
            <span className="font-medium text-slate-700">{String(sender.agent_id || "unknown")}</span>
            <span className="text-[9px] text-slate-400">({String(sender.agent_type || "")})</span>
          </div>
        )}
        {receiver && (
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-slate-400" />
            <span className="text-slate-500 cursor-help" title="The intended recipient of this message — either the end-user or another agent in the pipeline">Receiver:</span>
            <span className="font-medium text-slate-700">{receiver.is_human ? "User" : String(receiver.agent_id || "unknown")}</span>
          </div>
        )}
        {agentsChain.length > 0 && (
          <div className="flex items-center gap-2">
            <Shield size={12} className="text-slate-400" />
            <span className="text-slate-500 cursor-help" title="The sequence of agents that processed this request, from router to final responder">Chain:</span>
            <div className="flex flex-wrap gap-1">{agentsChain.map((a, i) => <Badge key={i} variant="secondary" className="text-[9px]">{a}</Badge>)}</div>
          </div>
        )}
      </div>
      <button onClick={() => setExpanded(!expanded)} className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600">
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        Raw envelope
      </button>
      {expanded && (
        <pre className="max-h-32 overflow-auto rounded-md border border-slate-100 bg-slate-50 p-2 text-[9px] text-slate-600">{JSON.stringify(envelope, null, 2)}</pre>
      )}
    </div>
  );
}

/* ── Expandable inline badge for use in MessageBubble ──── */

export function GovernanceBadge({ message }: Props) {
  const [badgeOpen, setBadgeOpen] = useState(false);
  const governance = message.raw?.governance;
  if (!governance?.compliance) return null;

  const rate = Number(governance.compliance.compliance_rate ?? 0);
  const pct = Math.round(rate * 100);
  const color = rate >= 0.8 ? "bg-green-100 text-green-700 border-green-200" : rate >= 0.5 ? "bg-amber-100 text-amber-700 border-amber-200" : "bg-red-100 text-red-700 border-red-200";
  const rateBg = rate >= 0.8 ? "bg-green-500" : rate >= 0.5 ? "bg-amber-500" : "bg-red-500";
  const explLevels = governance.explanations ? Object.keys(governance.explanations).length : 0;
  const byStandard = (governance.compliance.by_standard || {}) as Record<string, unknown>;
  const results = (governance.compliance.results || []) as Array<{ standard: string; requirement_id: string; description: string; severity: string; compliant: boolean; gap?: string; }>;
  const nonCompliant = results.filter((r) => !r.compliant);

  return (
    <div className="mt-2">
      <button onClick={(e) => { e.stopPropagation(); setBadgeOpen(!badgeOpen); }} className="flex cursor-pointer items-center gap-2">
        <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${color}`}><Shield size={10} />IEEE {pct}%</span>
        {explLevels > 0 && <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-medium text-blue-700"><Eye size={10} />{explLevels}/3 levels</span>}
      </button>
      {badgeOpen && (
        <div className="mt-2 space-y-3 rounded-lg border border-slate-200 bg-white p-3">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="font-medium text-slate-600">Overall Compliance</span>
              <span className={`font-bold ${rate >= 0.8 ? "text-green-600" : rate >= 0.5 ? "text-amber-600" : "text-red-600"}`}>{pct}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-slate-200"><div className={`h-1.5 rounded-full ${rateBg} transition-all`} style={{ width: `${pct}%` }} /></div>
            {Object.entries(byStandard).map(([std, data]) => {
              const stdRate = typeof data === "object" && data !== null ? Number((data as ComplianceStandard).rate ?? 0) : Number(data ?? 0);
              const stdPct = Math.round(stdRate * 100);
              return (
                <div key={std} className="flex items-center justify-between text-[10px]">
                  <StandardLink std={std} compact />
                  <div className="flex items-center gap-1.5">
                    <div className="h-1 w-10 rounded-full bg-slate-200"><div className={`h-1 rounded-full ${stdRate >= 0.8 ? "bg-green-500" : stdRate >= 0.5 ? "bg-amber-500" : "bg-red-500"}`} style={{ width: `${stdPct}%` }} /></div>
                    <span className="w-7 text-right font-mono font-medium text-slate-600">{stdPct}%</span>
                  </div>
                </div>
              );
            })}
            {nonCompliant.length > 0 && (
              <div className="mt-1 space-y-0.5">
                <span className="text-[10px] font-medium text-red-500">Gaps ({nonCompliant.length}):</span>
                {nonCompliant.map((r, i) => (
                  <div key={i} className="flex items-start gap-1 text-[10px]">
                    <XCircle size={10} className="mt-0.5 shrink-0 text-red-400" />
                    <span className="text-slate-600">
                      <a href={IEEE_STANDARDS[r.standard]?.url || "#"} target="_blank" rel="noopener noreferrer" className="font-medium text-blue-600 hover:underline" title={`${IEEE_STANDARDS[r.standard]?.label || r.standard} — ${r.description}`}>{r.requirement_id}</a>
                      {r.gap ? ` — ${r.gap}` : ` — ${r.description}`}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          {governance.explanations && <InlineExplainability explanations={governance.explanations} />}
          {governance.envelope && <InlineEnvelope envelope={governance.envelope} />}
        </div>
      )}
    </div>
  );
}

/* ── Inline explainability (for MessageBubble expand) ── */

function InlineExplainability({ explanations }: { explanations: Record<string, Record<string, unknown>> }) {
  const [activeLevel, setActiveLevel] = useState<string | null>(null);
  const levels = [
    { key: "summary", label: "Summary", icon: Eye },
    { key: "detailed", label: "Detailed", icon: FileText },
    { key: "full", label: "Full", icon: Code },
  ];

  return (
    <div className="space-y-1.5">
      <span className="text-[10px] font-semibold uppercase text-slate-400">Explainability (<StandardLink std="2894-2024" compact />)</span>
      <div className="flex gap-1">
        {levels.map(({ key, label, icon: Icon }) => {
          const available = key in explanations;
          const active = activeLevel === key;
          return (
            <button key={key} disabled={!available} onClick={() => setActiveLevel(active ? null : key)}
              className={`flex flex-1 items-center justify-center gap-1 rounded border px-2 py-1 text-xs transition-colors ${active ? "border-blue-300 bg-blue-50 text-blue-700 font-medium" : available ? "border-slate-200 text-slate-600 hover:border-blue-200 hover:bg-blue-50/50" : "border-slate-100 bg-slate-50 text-slate-300"}`}>
              <Icon size={10} />{label}
            </button>
          );
        })}
      </div>
      {activeLevel && explanations[activeLevel] && (
        <div className="rounded border border-slate-100 bg-slate-50 p-2 text-xs text-slate-700 leading-relaxed space-y-1.5">
          {String(explanations[activeLevel].narrative || "No narrative.")}
          {Array.isArray(explanations[activeLevel].decisions) && (explanations[activeLevel].decisions as Array<Record<string, unknown>>).length > 0 && (
            <details className="mt-1.5">
              <summary className="cursor-pointer font-medium text-slate-500">Decisions ({(explanations[activeLevel].decisions as Array<unknown>).length})</summary>
              <div className="mt-1 space-y-0.5">
                {(explanations[activeLevel].decisions as Array<Record<string, unknown>>).map((d, i) => (
                  <div key={i} className="text-slate-600">
                    <span className="font-medium">{String(d.stage || `Step ${i + 1}`)}</span>
                    {d.rationale != null && <span className="text-slate-400">{` — ${String(d.rationale)}`}</span>}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Inline envelope (for MessageBubble expand) ── */

function InlineEnvelope({ envelope }: { envelope: Record<string, unknown> }) {
  const sender = envelope.sender as Record<string, unknown> | undefined;
  const aiGenerated = envelope.ai_generated as boolean | undefined;
  const agentsChain = (envelope.agents_chain || []) as string[];

  return (
    <div className="space-y-1">
      <span className="text-[10px] font-semibold uppercase text-slate-400">Message Envelope (<StandardLink std="P3394" compact />)</span>
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        {aiGenerated && <span className="inline-flex items-center gap-1 rounded bg-purple-50 px-1.5 py-0.5 text-purple-700 border border-purple-200"><Fingerprint size={9} />AI-generated</span>}
        {sender && <span className="text-slate-600">Sender: <span className="font-medium">{String(sender.agent_id || "unknown")}</span><span className="text-slate-400"> ({String(sender.agent_type || "")})</span></span>}
        {agentsChain.length > 0 && <span className="text-slate-600">Chain: {agentsChain.join(" → ")}</span>}
      </div>
    </div>
  );
}
