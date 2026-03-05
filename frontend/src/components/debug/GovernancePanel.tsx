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
} from "lucide-react";
import { useState } from "react";

/* Tooltip descriptions for IEEE standard short codes */
const STANDARD_TOOLTIPS: Record<string, string> = {
  "P3394": "IEEE P3394 — Universal Message Format for multi-agent communication",
  "2894-2024": "IEEE 2894-2024 — Guide for Explainable Artificial Intelligence",
  "3152-2024": "IEEE 3152-2024 — Transparent Human/Machine Agency Framework",
};

interface Props {
  message: ChatMessage;
}

interface ComplianceStandard {
  rate: number;
  total: number;
  compliant: number;
}

export function GovernancePanel({ message }: Props) {
  const governance = message.raw?.governance;
  if (!governance) {
    return (
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase text-slate-500">
          IEEE Governance
        </h3>
        <p className="text-xs text-slate-400">
          No governance data available for this message.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-semibold uppercase text-slate-500">
        IEEE Governance
      </h3>
      <ComplianceSection compliance={governance.compliance} />
      <ExplainabilitySection explanations={governance.explanations} />
      <EnvelopeSection envelope={governance.envelope} />
    </div>
  );
}

/* ── Compliance Section ──────────────────────────────── */

function ComplianceSection({ compliance }: { compliance: Record<string, unknown> }) {
  const [expanded, setExpanded] = useState(false);

  if (!compliance) return null;

  const rate = Number(compliance.compliance_rate ?? 0);
  const byStandard = (compliance.by_standard || {}) as Record<string, ComplianceStandard>;
  const results = (compliance.results || []) as Array<{
    standard: string; requirement_id: string; description: string; severity: string;
    compliant: boolean;
    evidence?: string;
    gap?: string;
  }>;

  const rateColor =
    rate >= 0.8 ? "text-green-600" : rate >= 0.5 ? "text-amber-600" : "text-red-600";
  const rateBg =
    rate >= 0.8 ? "bg-green-500" : rate >= 0.5 ? "bg-amber-500" : "bg-red-500";

  return (
    <div className="space-y-2">
      {/* Overall rate */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600">Overall Compliance</span>
        <span className={`text-sm font-bold ${rateColor}`}>
          {(rate * 100).toFixed(0)}%
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-200">
        <div
          className={`h-2 rounded-full ${rateBg} transition-all`}
          style={{ width: `${Math.round(rate * 100)}%` }}
        />
      </div>

      {/* By standard */}
      <div className="space-y-1.5 pt-1">
        {Object.entries(byStandard).map(([std, data]) => {
          const stdRate = typeof data === "object" && data !== null ? Number((data as ComplianceStandard).rate ?? 0) : Number(data ?? 0);
          const color =
            stdRate >= 0.8 ? "text-green-600" : stdRate >= 0.5 ? "text-amber-600" : "text-red-600";
          return (
            <div key={std} className="flex items-center justify-between text-xs">
              <span className="text-slate-500 cursor-help" title={STANDARD_TOOLTIPS[std] || std}>{std}</span>
              <div className="flex items-center gap-2">
                <div className="h-1.5 w-12 rounded-full bg-slate-200">
                  <div
                    className={`h-1.5 rounded-full ${stdRate >= 0.8 ? "bg-green-500" : stdRate >= 0.5 ? "bg-amber-500" : "bg-red-500"}`}
                    style={{ width: `${Math.round(stdRate * 100)}%` }}
                  />
                </div>
                <span className={`font-mono font-medium ${color}`}>
                  {(stdRate * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Expandable requirement details */}
      {results.length > 0 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600"
        >
          {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
          {results.length} requirements checked
        </button>
      )}
      {expanded && (
        <div className="max-h-48 space-y-1 overflow-y-auto rounded-md border border-slate-100 bg-slate-50 p-2">
          {results.map((r, i) => (
            <div key={i} className="flex items-start gap-1.5 text-[10px]">
              {r.compliant ? (
                <CheckCircle size={10} className="mt-0.5 shrink-0 text-green-500" />
              ) : (
                <XCircle size={10} className="mt-0.5 shrink-0 text-red-500" />
              )}
              <div className="flex-1">
                <span
                  className="font-medium text-slate-700 underline decoration-dotted cursor-help"
                  title={`[${r.standard}] ${r.description} (${r.severity})`}
                >
                  {r.requirement_id}
                </span>
                <span className="ml-1 text-slate-500">
                  {r.description}
                </span>
                {r.gap && (
                  <p className="text-red-500">{r.gap}</p>
                )}
              </div>
              <Badge
                variant="secondary"
                className="shrink-0 text-[8px] px-1 py-0"
              >
                {r.severity}
              </Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Explainability Section ──────────────────────────── */

function ExplainabilitySection({
  explanations,
}: {
  explanations: Record<string, Record<string, unknown>> | undefined;
}) {
  const [activeLevel, setActiveLevel] = useState<string | null>(null);

  if (!explanations) return null;

  const levels = [
    { key: "summary", label: "Summary", icon: Eye, desc: "User-facing" },
    { key: "detailed", label: "Detailed", icon: FileText, desc: "Auditor" },
    { key: "full", label: "Full", icon: Code, desc: "Developer" },
  ];

  return (
    <div className="space-y-2">
      <h4 className="text-[10px] font-semibold uppercase text-slate-400">
        Explainability (IEEE 2894)
      </h4>

      <div className="flex gap-1">
        {levels.map(({ key, label, icon: Icon, desc }) => {
          const available = key in explanations;
          const active = activeLevel === key;
          return (
            <button
              key={key}
              disabled={!available}
              onClick={() => setActiveLevel(active ? null : key)}
              className={`flex flex-1 flex-col items-center gap-0.5 rounded-md border p-1.5 text-[10px] transition-colors ${
                active
                  ? "border-blue-300 bg-blue-50 text-blue-700"
                  : available
                    ? "border-slate-200 bg-white text-slate-600 hover:border-blue-200 hover:bg-blue-50/50"
                    : "border-slate-100 bg-slate-50 text-slate-300"
              }`}
            >
              <Icon size={12} />
              <span className="font-medium">{label}</span>
              <span className="text-[8px] text-slate-400">{desc}</span>
            </button>
          );
        })}
      </div>

      {/* Active explanation content */}
      {activeLevel && explanations[activeLevel] && (
        <ExplanationContent data={explanations[activeLevel]} />
      )}
    </div>
  );
}

function ExplanationContent({ data }: { data: Record<string, unknown> }) {
  const narrative = String(data.narrative || "");
  const agents = (data.agents_involved || []) as string[];
  const decisions = (data.decisions || []) as Array<Record<string, unknown>>;
  const provenance = (data.provenance || []) as Array<Record<string, unknown>>;
  const metrics = (data.metrics || {}) as Record<string, unknown>;

  return (
    <div className="space-y-2 rounded-md border border-slate-100 bg-white p-2">
      {/* Narrative */}
      {narrative && (
        <p className="text-[11px] leading-relaxed text-slate-700">{narrative}</p>
      )}

      {/* Agents involved */}
      {agents.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {agents.map((a) => (
            <Badge key={a} variant="secondary" className="text-[9px]">
              {a}
            </Badge>
          ))}
        </div>
      )}

      {/* Key decisions */}
      {decisions.length > 0 && (
        <details className="text-[10px]">
          <summary className="cursor-pointer font-medium text-slate-500 hover:text-slate-700">
            Decisions ({decisions.length})
          </summary>
          <div className="mt-1 space-y-1">
            {decisions.map((d, i) => (
              <div key={i} className="rounded bg-slate-50 px-2 py-1">
                <span className="font-medium text-slate-600">
                  {String(d.stage || d.type || `Step ${i + 1}`)}
                </span>
                {d.rationale != null && (
                  <span className="ml-1 text-slate-500">{`— ${String(d.rationale)}`}</span>
                )}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Provenance */}
      {provenance.length > 0 && (
        <details className="text-[10px]">
          <summary className="cursor-pointer font-medium text-slate-500 hover:text-slate-700">
            Provenance ({provenance.length})
          </summary>
          <div className="mt-1 space-y-1">
            {provenance.map((p, i) => (
              <div key={i} className="rounded bg-slate-50 px-2 py-1 text-slate-600">
                {String(p.source || p.type || `Source ${i + 1}`)}
                {p.detail != null && <span className="ml-1 text-slate-400">{`— ${String(p.detail)}`}</span>}
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Metrics */}
      {Object.keys(metrics).length > 0 && (
        <details className="text-[10px]">
          <summary className="cursor-pointer font-medium text-slate-500 hover:text-slate-700">
            Metrics
          </summary>
          <div className="mt-1 space-y-0.5">
            {Object.entries(metrics)
              .filter(([k]) => k !== "event_log")
              .map(([k, v]) => (
                <div key={k} className="flex justify-between text-slate-600">
                  <span>{k.replace(/_/g, " ")}</span>
                  <span className="font-mono">
                    {typeof v === "number" ? v.toFixed(3) : String(v)}
                  </span>
                </div>
              ))}
          </div>
        </details>
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
      <h4 className="text-[10px] font-semibold uppercase text-slate-400">
        Message Envelope (IEEE P3394)
      </h4>

      <div className="space-y-1.5 text-[11px]">
        {/* AI disclosure */}
        <div className="flex items-center gap-2">
          <Fingerprint size={12} className="text-slate-400" />
          <span className="text-slate-500">AI-generated:</span>
          {aiGenerated ? (
            <Badge variant="secondary" className="bg-purple-100 text-purple-700 text-[9px]">
              Yes
            </Badge>
          ) : (
            <Badge variant="secondary" className="text-[9px]">No</Badge>
          )}
        </div>

        {/* Sender */}
        {sender && (
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-slate-400" />
            <span className="text-slate-500">Sender:</span>
            <span className="font-medium text-slate-700">
              {String(sender.agent_id || "unknown")}
            </span>
            <span className="text-[9px] text-slate-400">
              ({String(sender.agent_type || "")})
            </span>
          </div>
        )}

        {/* Receiver */}
        {receiver && (
          <div className="flex items-center gap-2">
            <Activity size={12} className="text-slate-400" />
            <span className="text-slate-500">Receiver:</span>
            <span className="font-medium text-slate-700">
              {(receiver.is_human ? "User" : String(receiver.agent_id || "unknown"))}
            </span>
          </div>
        )}

        {/* Agents chain */}
        {agentsChain.length > 0 && (
          <div className="flex items-center gap-2">
            <Shield size={12} className="text-slate-400" />
            <span className="text-slate-500">Chain:</span>
            <div className="flex flex-wrap gap-1">
              {agentsChain.map((a, i) => (
                <Badge key={i} variant="secondary" className="text-[9px]">
                  {a}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Expandable raw envelope */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-600"
      >
        {expanded ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
        Raw envelope
      </button>
      {expanded && (
        <pre className="max-h-32 overflow-auto rounded-md border border-slate-100 bg-slate-50 p-2 text-[9px] text-slate-600">
          {JSON.stringify(envelope, null, 2)}
        </pre>
      )}
    </div>
  );
}

/* ── Expandable inline badge for use in MessageBubble ──── */

export function GovernanceBadge({ message }: Props) {
  const governance = message.raw?.governance;
  if (!governance?.compliance) return null;

  const rate = Number(governance.compliance.compliance_rate ?? 0);
  const pct = Math.round(rate * 100);
  const color =
    rate >= 0.8
      ? "bg-green-100 text-green-700 border-green-200"
      : rate >= 0.5
        ? "bg-amber-100 text-amber-700 border-amber-200"
        : "bg-red-100 text-red-700 border-red-200";

  const rateBg =
    rate >= 0.8 ? "bg-green-500" : rate >= 0.5 ? "bg-amber-500" : "bg-red-500";

  const explLevels = governance.explanations
    ? Object.keys(governance.explanations).length
    : 0;

  const byStandard = (governance.compliance.by_standard || {}) as Record<string, unknown>;
  const results = (governance.compliance.results || []) as Array<{
    standard: string; requirement_id: string; description: string; severity: string;
    compliant: boolean;
    gap?: string;
  }>;
  const nonCompliant = results.filter((r) => !r.compliant);

  return (
    <details className="mt-2" onClick={(e) => e.stopPropagation()}>
      <summary className="flex cursor-pointer items-center gap-2 list-none [&::-webkit-details-marker]:hidden">
        <span
          className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${color}`}
        >
          <Shield size={10} />
          IEEE {pct}%
        </span>
        {explLevels > 0 && (
          <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-medium text-blue-700">
            <Eye size={10} />
            {explLevels}/3 levels
          </span>
        )}
      </summary>

      <div className="mt-2 space-y-3 rounded-lg border border-slate-200 bg-white p-3">
        {/* ── Compliance breakdown ── */}
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="font-medium text-slate-600">Overall Compliance</span>
            <span className={`font-bold ${rate >= 0.8 ? "text-green-600" : rate >= 0.5 ? "text-amber-600" : "text-red-600"}`}>
              {pct}%
            </span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-slate-200">
            <div
              className={`h-1.5 rounded-full ${rateBg} transition-all`}
              style={{ width: `${pct}%` }}
            />
          </div>

          {/* Per-standard bars */}
          {Object.entries(byStandard).map(([std, data]) => {
            const stdRate = typeof data === "object" && data !== null
              ? Number((data as ComplianceStandard).rate ?? 0)
              : Number(data ?? 0);
            const stdPct = Math.round(stdRate * 100);
            return (
              <div key={std} className="flex items-center justify-between text-[10px]">
                <span className="text-slate-500 cursor-help" title={STANDARD_TOOLTIPS[std] || std}>{std}</span>
                <div className="flex items-center gap-1.5">
                  <div className="h-1 w-10 rounded-full bg-slate-200">
                    <div
                      className={`h-1 rounded-full ${stdRate >= 0.8 ? "bg-green-500" : stdRate >= 0.5 ? "bg-amber-500" : "bg-red-500"}`}
                      style={{ width: `${stdPct}%` }}
                    />
                  </div>
                  <span className="w-7 text-right font-mono font-medium text-slate-600">
                    {stdPct}%
                  </span>
                </div>
              </div>
            );
          })}

          {/* Non-compliant items */}
          {nonCompliant.length > 0 && (
            <div className="mt-1 space-y-0.5">
              <span className="text-[10px] font-medium text-red-500">
                Gaps ({nonCompliant.length}):
              </span>
              {nonCompliant.map((r, i) => (
                <div key={i} className="flex items-start gap-1 text-[10px]">
                  <XCircle size={10} className="mt-0.5 shrink-0 text-red-400" />
                  <span className="text-slate-600">
                    <span
                      className="font-medium underline decoration-dotted cursor-help"
                      title={`[${r.standard}] ${r.description} (${r.severity})`}
                    >
                      {r.requirement_id}
                    </span>
                    {r.gap ? ` — ${r.gap}` : ` — ${r.description}`}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Explainability levels ── */}
        {governance.explanations && (
          <InlineExplainability explanations={governance.explanations} />
        )}

        {/* ── Envelope summary ── */}
        {governance.envelope && (
          <InlineEnvelope envelope={governance.envelope} />
        )}
      </div>
    </details>
  );
}

/* ── Inline explainability (for MessageBubble expand) ── */

function InlineExplainability({
  explanations,
}: {
  explanations: Record<string, Record<string, unknown>>;
}) {
  const [activeLevel, setActiveLevel] = useState<string | null>(null);

  const levels = [
    { key: "summary", label: "Summary", icon: Eye },
    { key: "detailed", label: "Detailed", icon: FileText },
    { key: "full", label: "Full", icon: Code },
  ];

  return (
    <div className="space-y-1.5">
      <span className="text-[10px] font-semibold uppercase text-slate-400">
        Explainability (IEEE 2894)
      </span>
      <div className="flex gap-1">
        {levels.map(({ key, label, icon: Icon }) => {
          const available = key in explanations;
          const active = activeLevel === key;
          return (
            <button
              key={key}
              disabled={!available}
              onClick={() => setActiveLevel(active ? null : key)}
              className={`flex flex-1 items-center justify-center gap-1 rounded border px-2 py-1 text-[10px] transition-colors ${
                active
                  ? "border-blue-300 bg-blue-50 text-blue-700 font-medium"
                  : available
                    ? "border-slate-200 text-slate-600 hover:border-blue-200 hover:bg-blue-50/50"
                    : "border-slate-100 bg-slate-50 text-slate-300"
              }`}
            >
              <Icon size={10} />
              {label}
            </button>
          );
        })}
      </div>

      {activeLevel && explanations[activeLevel] && (
        <div className="rounded border border-slate-100 bg-slate-50 p-2 text-[10px] text-slate-700 leading-relaxed">
          {String(explanations[activeLevel].narrative || "No narrative.")}

          {/* Decisions for detailed/full */}
          {Array.isArray(explanations[activeLevel].decisions) &&
            (explanations[activeLevel].decisions as Array<Record<string, unknown>>).length > 0 && (
              <details className="mt-1.5">
                <summary className="cursor-pointer font-medium text-slate-500">
                  Decisions ({(explanations[activeLevel].decisions as Array<unknown>).length})
                </summary>
                <div className="mt-1 space-y-0.5">
                  {(explanations[activeLevel].decisions as Array<Record<string, unknown>>).map((d, i) => (
                    <div key={i} className="text-slate-600">
                      <span className="font-medium">{String(d.stage || `Step ${i + 1}`)}</span>
                      {d.rationale != null && (
                        <span className="text-slate-400">{` — ${String(d.rationale)}`}</span>
                      )}
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
      <span className="text-[10px] font-semibold uppercase text-slate-400">
        Message Envelope (P3394)
      </span>
      <div className="flex flex-wrap items-center gap-2 text-[10px]">
        {aiGenerated && (
          <span className="inline-flex items-center gap-1 rounded bg-purple-50 px-1.5 py-0.5 text-purple-700 border border-purple-200">
            <Fingerprint size={9} />
            AI-generated
          </span>
        )}
        {sender && (
          <span className="text-slate-600">
            Sender: <span className="font-medium">{String(sender.agent_id || "unknown")}</span>
            <span className="text-slate-400"> ({String(sender.agent_type || "")})</span>
          </span>
        )}
        {agentsChain.length > 0 && (
          <span className="text-slate-600">
            Chain: {agentsChain.join(" → ")}
          </span>
        )}
      </div>
    </div>
  );
}
