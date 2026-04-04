"use client";

import { useChatStore } from "@/store/chatStore";

export function UsageBadge() {
  const usage = useChatStore((s) => s.usage);

  if (!usage) return null;

  const pct = Math.round(
    (usage.session_llm_calls / usage.session_llm_limit) * 100
  );
  const isWarning = pct >= 70;
  const isCritical = pct >= 90;

  const color = isCritical
    ? "text-red-500"
    : isWarning
      ? "text-amber-500"
      : "text-zinc-400";

  return (
    <div className={`flex items-center gap-1.5 text-xs ${color}`}>
      <div
        className="h-1.5 w-16 rounded-full bg-zinc-700 overflow-hidden"
        title={`${usage.session_llm_calls} / ${usage.session_llm_limit} queries used`}
      >
        <div
          className={`h-full rounded-full transition-all ${
            isCritical
              ? "bg-red-500"
              : isWarning
                ? "bg-amber-500"
                : "bg-emerald-500"
          }`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span>
        {usage.session_remaining} left
      </span>
    </div>
  );
}
