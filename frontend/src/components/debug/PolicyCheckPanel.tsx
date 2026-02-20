"use client";

import type { ChatMessage } from "@/types/chat";
import { Shield, CheckCircle } from "lucide-react";

interface Props {
  message: ChatMessage;
}

export function PolicyCheckPanel({ message }: Props) {
  const raw = message.raw;
  if (!raw) return null;

  const blocked = message.responseKind === "guardrails_block";
  const reason = raw.reason || raw.guardrail_reason;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase text-slate-500">
        Policy Check
      </h3>
      <div className="flex items-center gap-2 text-sm">
        {blocked ? (
          <>
            <Shield size={14} className="text-red-500" />
            <span className="text-red-600">Blocked</span>
          </>
        ) : (
          <>
            <CheckCircle size={14} className="text-green-500" />
            <span className="text-green-600">Passed</span>
          </>
        )}
      </div>
      {reason && (
        <p className="text-xs text-slate-600">{String(reason)}</p>
      )}
    </div>
  );
}
