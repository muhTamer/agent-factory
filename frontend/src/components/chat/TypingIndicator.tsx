"use client";

export function TypingIndicator({ agentName }: { agentName?: string }) {
  return (
    <div className="flex items-start gap-2 mb-4">
      <div className="rounded-xl rounded-tl-sm bg-slate-100 px-4 py-3">
        {agentName && (
          <span className="text-xs font-semibold text-slate-400 block mb-1">
            {agentName}
          </span>
        )}
        <div className="flex gap-1">
          <span className="h-2 w-2 rounded-full bg-slate-400 animate-bounce" />
          <span className="h-2 w-2 rounded-full bg-slate-400 animate-bounce [animation-delay:0.15s]" />
          <span className="h-2 w-2 rounded-full bg-slate-400 animate-bounce [animation-delay:0.3s]" />
        </div>
      </div>
    </div>
  );
}
