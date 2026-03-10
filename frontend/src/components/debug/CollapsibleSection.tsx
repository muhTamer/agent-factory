"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

export type SectionStatus = "ok" | "warning" | "error" | "info" | "neutral";

const STATUS_STYLES: Record<SectionStatus, string> = {
  ok: "border-l-green-400",
  warning: "border-l-amber-400",
  error: "border-l-red-400",
  info: "border-l-blue-400",
  neutral: "border-l-transparent",
};

interface Props {
  icon: ReactNode;
  title: string;
  tooltip: string;
  /** Optional badge shown to the right of the title */
  badge?: ReactNode;
  /** Collapsed-state summary shown when the section is closed */
  collapsedSummary?: ReactNode;
  /** Status indicator — adds a left border color accent */
  status?: SectionStatus;
  /** Default collapsed/expanded state */
  defaultOpen?: boolean;
  children: ReactNode;
}

export function CollapsibleSection({
  icon,
  title,
  tooltip,
  badge,
  collapsedSummary,
  status = "neutral",
  defaultOpen = true,
  children,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      className={`rounded-lg border border-slate-100 border-l-[3px] ${STATUS_STYLES[status]} bg-white transition-colors`}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1.5 px-3 py-2.5 text-left cursor-help group"
        title={tooltip}
      >
        {open ? (
          <ChevronDown
            size={12}
            className="shrink-0 text-slate-300 group-hover:text-slate-500 transition-colors"
          />
        ) : (
          <ChevronRight
            size={12}
            className="shrink-0 text-slate-300 group-hover:text-slate-500 transition-colors"
          />
        )}
        {icon}
        <span className="text-xs font-semibold uppercase text-slate-500">
          {title}
        </span>
        {badge && <span className="ml-auto">{badge}</span>}
      </button>
      {/* Collapsed summary — visible only when closed */}
      {!open && collapsedSummary && (
        <div className="px-3 pb-2.5 -mt-1">{collapsedSummary}</div>
      )}
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  );
}
