"use client";

import type { ChatMessage } from "@/types/chat";
import {
  BookOpen,
  FileText,
  CheckCircle,
  Clock,
  ScrollText,
} from "lucide-react";
import { CollapsibleSection } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

interface PolicySource {
  name: string;
  type: string;
  active_entries?: string[];
}

export function SourcesPanel({ message }: Props) {
  const raw = message.raw;
  if (!raw) return null;

  // Gather knowledge sources from all possible locations
  const directKs = (raw.knowledge_sources || []) as Array<
    Record<string, unknown>
  >;
  const directCitations = (raw.citations ||
    raw.grounded_citations ||
    []) as Array<Record<string, unknown>>;

  // Governance provenance across all explanation levels
  const provenanceKs: Array<Record<string, unknown>> = [];
  const provenanceCitations: Array<Record<string, unknown>> = [];

  const governance = raw.governance as Record<string, unknown> | undefined;
  if (governance?.explanations) {
    const explanations = governance.explanations as Record<
      string,
      Record<string, unknown>
    >;
    for (const levelData of Object.values(explanations)) {
      const prov = (levelData?.provenance || []) as Array<
        Record<string, unknown>
      >;
      for (const p of prov) {
        if (Array.isArray(p.knowledge_sources)) {
          provenanceKs.push(
            ...(p.knowledge_sources as Array<Record<string, unknown>>),
          );
        }
        if (Array.isArray(p.citations)) {
          provenanceCitations.push(
            ...(p.citations as Array<Record<string, unknown>>),
          );
        }
      }
    }
  }

  // AOP subtask results may also carry knowledge sources
  const subtaskResults = (raw.subtask_results || []) as Array<
    Record<string, unknown>
  >;
  for (const st of subtaskResults) {
    const result = st.result as Record<string, unknown> | undefined;
    if (result) {
      const ks = result.knowledge_sources as
        | Array<Record<string, unknown>>
        | undefined;
      if (Array.isArray(ks)) provenanceKs.push(...ks);
      const cits = (result.grounded_citations || result.citations) as
        | Array<Record<string, unknown>>
        | undefined;
      if (Array.isArray(cits)) provenanceCitations.push(...cits);
    }
  }

  // Deduplicate
  const allKs = deduplicateByQuery([...directKs, ...provenanceKs]);
  const allCitations = deduplicateCitations([
    ...directCitations,
    ...provenanceCitations,
  ]);

  // Policy sources (always present when agent follows a configured policy)
  const policySources = (raw.policy_sources || []) as PolicySource[];

  const hasAnything =
    allKs.length > 0 || allCitations.length > 0 || policySources.length > 0;
  if (!hasAnything) return null;

  return (
    <CollapsibleSection
      icon={<BookOpen size={14} className="text-blue-500" />}
      title="Sources & Evidence"
      tooltip="Knowledge documents, policy entries, and citations the agent used to generate this response"
    >
      <div className="space-y-3">
        {/* Policy source badges — always shown when agent is policy-driven */}
        {policySources.length > 0 &&
          policySources.map((ps, i) => (
            <div
              key={`policy-${i}`}
              className="rounded-lg border border-indigo-100 bg-indigo-50/30 p-3 space-y-1.5"
            >
              <div className="flex items-center gap-1.5 flex-wrap">
                <ScrollText size={12} className="text-indigo-500 shrink-0" />
                <span
                  className="text-[11px] font-medium text-indigo-600 cursor-help"
                  title="The agent follows a structured workflow policy that dictates which steps to take, which tools to call, and what information to collect"
                >
                  Policy Grounding
                </span>
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <span
                  className="rounded-full bg-indigo-100 px-2 py-0.5 text-[11px] font-medium text-indigo-700 cursor-help"
                  title="The YAML policy file that defines the agent's step-by-step workflow"
                >
                  {ps.name}
                </span>
              </div>

              {/* Active policy entries — the specific steps driving this turn */}
              {ps.active_entries && ps.active_entries.length > 0 ? (
                <div className="space-y-1">
                  <p
                    className="text-[10px] font-semibold uppercase text-indigo-400 cursor-help"
                    title="The specific workflow steps the agent referenced in its reasoning for this turn"
                  >
                    Active Steps This Turn
                  </p>
                  {ps.active_entries.map((entry, j) => (
                    <div
                      key={j}
                      className="rounded border border-indigo-100 bg-white px-2.5 py-1.5 text-xs leading-relaxed text-slate-600"
                    >
                      {entry}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-[11px] text-slate-400">
                  Agent decisions follow the workflow steps defined in this
                  policy.
                </p>
              )}
            </div>
          ))}

        {/* Knowledge sources from domain agent retrieval */}
        {allKs.map((ks, i) => (
          <div
            key={`ks-${i}`}
            className="rounded-lg border border-blue-100 bg-blue-50/30 p-3 space-y-2"
          >
            {/* Prior-turn label */}
            {ks.from_prior_turn ? (
              <div
                className="flex items-center gap-1 text-[10px] text-slate-400 cursor-help"
                title="This knowledge was retrieved in a previous conversation turn, not the current one — it provides context for the agent's ongoing reasoning"
              >
                <Clock size={10} className="shrink-0" />
                Retrieved in an earlier turn
              </div>
            ) : null}

            {/* Source file badges */}
            {Array.isArray(ks.sources) &&
            (ks.sources as string[]).length > 0 ? (
              <div className="flex items-center gap-1.5 flex-wrap">
                <FileText size={12} className="text-blue-500 shrink-0" />
                {(ks.sources as string[]).map((src, j) => (
                  <span
                    key={j}
                    className="rounded-full bg-blue-100 px-2 py-0.5 text-[11px] font-medium text-blue-700 cursor-help"
                    title="The knowledge base document that was searched by the retrieval system"
                  >
                    {src}
                  </span>
                ))}
              </div>
            ) : null}

            {/* Retrieval query */}
            {ks.query ? (
              <p
                className="text-xs text-slate-500 cursor-help"
                title="The search query the agent used to find relevant knowledge passages"
              >
                <span className="font-medium">Retrieval query:</span>{" "}
                <span className="italic">
                  &ldquo;{String(ks.query)}&rdquo;
                </span>
              </p>
            ) : null}

            {/* Full passages — no truncation */}
            {Array.isArray(ks.passages) &&
            (ks.passages as string[]).length > 0 ? (
              <div className="space-y-1.5">
                {(ks.passages as string[]).map((passage, k) => (
                  <div
                    key={k}
                    className="rounded border border-blue-100 bg-white px-3 py-2 text-xs leading-relaxed text-slate-700"
                  >
                    {passage}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ))}

        {/* RAG citations */}
        {allCitations.length > 0 && (
          <div className="space-y-1.5">
            {allCitations.map((c, j) => (
              <div
                key={`cit-${j}`}
                className="flex items-start gap-2 rounded-lg border border-green-100 bg-green-50/30 px-3 py-2"
              >
                <CheckCircle
                  size={12}
                  className="mt-0.5 shrink-0 text-green-500"
                />
                <div className="text-xs leading-relaxed text-slate-700">
                  {c.source ? (
                    <span className="font-semibold text-green-700 mr-1">
                      [{String(c.source)}]
                    </span>
                  ) : null}
                  {String(c.question || c.text || `Citation ${j + 1}`)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}

function deduplicateByQuery(
  items: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key =
      String(item.query || "") + "|" + JSON.stringify(item.sources || []);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function deduplicateCitations(
  items: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = String(item.question || item.text || "");
    if (!key || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
