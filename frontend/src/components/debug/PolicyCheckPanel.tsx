"use client";

import type { ChatMessage } from "@/types/chat";
import { Shield, CheckCircle } from "lucide-react";
import { CollapsibleSection, type SectionStatus } from "./CollapsibleSection";

interface Props {
  message: ChatMessage;
}

export function PolicyCheckPanel({ message }: Props) {
  const raw = message.raw;
  if (!raw) return null;

  const blocked = message.responseKind === "guardrails_block";
  const reason = raw.reason || raw.guardrail_reason;
  const sectionStatus: SectionStatus = blocked ? "error" : "ok";

  return (
    <CollapsibleSection
      icon={<Shield size={14} className={blocked ? "text-red-500" : "text-green-500"} />}
      title="Policy Check"
      tooltip="The guardrail layer checks the query and response against safety policies before delivery — blocks harmful, off-topic, or policy-violating content"
      status={sectionStatus}
      badge={
        <span className={`text-[10px] font-medium ${blocked ? "text-red-500" : "text-green-500"}`}>
          {blocked ? "Blocked" : "Passed"}
        </span>
      }
      collapsedSummary={
        <div className="flex items-center gap-1.5 text-xs">
          {blocked ? (
            <>
              <Shield size={11} className="text-red-500" />
              <span className="text-red-600 font-medium">Blocked</span>
              {reason && <span className="text-slate-400 truncate max-w-[200px]">&mdash; {String(reason)}</span>}
            </>
          ) : (
            <>
              <CheckCircle size={11} className="text-green-500" />
              <span className="text-green-600">All checks passed</span>
            </>
          )}
        </div>
      }
    >
      <div className="flex items-center gap-2 text-sm">
        {blocked ? (
          <>
            <Shield size={14} className="text-red-500" />
            <span className="text-red-600 cursor-help" title="The response was blocked by the guardrail because it violated a safety or policy rule">Blocked</span>
          </>
        ) : (
          <>
            <CheckCircle size={14} className="text-green-500" />
            <span className="text-green-600 cursor-help" title="The response passed all guardrail checks and was delivered to the user">Passed</span>
          </>
        )}
      </div>
      {reason && (
        <p className="text-xs text-slate-600 cursor-help" title="The specific policy rule or reason that triggered the guardrail decision">{String(reason)}</p>
      )}
    </CollapsibleSection>
  );
}
