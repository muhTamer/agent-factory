"use client";

import { cn } from "@/lib/utils";

interface ConfidenceBarProps {
  value: number; // 0..1
  className?: string;
}

export function ConfidenceBar({ value, className }: ConfidenceBarProps) {
  const pct = Math.round(value * 100);
  const color =
    value >= 0.8
      ? "bg-green-500"
      : value >= 0.5
        ? "bg-amber-500"
        : "bg-red-500";

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="h-2 flex-1 rounded-full bg-slate-200">
        <div
          className={cn("h-2 rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-medium text-slate-500 tabular-nums">
        {pct}%
      </span>
    </div>
  );
}
