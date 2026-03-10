"use client";

import { useState, type ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface Props {
  icon: ReactNode;
  title: string;
  tooltip: string;
  /** Optional badge shown to the right of the title */
  badge?: ReactNode;
  /** Default collapsed/expanded state */
  defaultOpen?: boolean;
  children: ReactNode;
}

export function CollapsibleSection({
  icon,
  title,
  tooltip,
  badge,
  defaultOpen = true,
  children,
}: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1.5 text-left cursor-help group"
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
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}
