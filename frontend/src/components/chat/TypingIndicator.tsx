"use client";

export function TypingIndicator() {
  return (
    <div className="flex items-start gap-2 mb-4">
      <div className="rounded-xl rounded-tl-sm bg-slate-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-slate-500">Thinking</span>
          <div className="flex gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-bounce" />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.15s]" />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-bounce [animation-delay:0.3s]" />
          </div>
        </div>
      </div>
    </div>
  );
}
