"use client";

import type { WizardStep } from "@/types/concierge";
import { Check, Circle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

const STEPS: { key: WizardStep; label: string }[] = [
  { key: "welcome", label: "Welcome" },
  { key: "upload", label: "Upload" },
  { key: "analysis", label: "Analysis" },
  { key: "deploy", label: "Deploy" },
  { key: "runtime", label: "Launch" },
];

interface WizardProgressBarProps {
  currentStep: WizardStep;
}

export function WizardProgressBar({ currentStep }: WizardProgressBarProps) {
  const currentIdx = STEPS.findIndex((s) => s.key === currentStep);

  return (
    <div className="mb-6 sm:mb-10 flex items-center justify-center gap-0.5 sm:gap-1">
      {STEPS.map((step, i) => {
        const isComplete = i < currentIdx;
        const isCurrent = i === currentIdx;

        return (
          <div key={step.key} className="flex items-center">
            <div className="flex flex-col items-center gap-1">
              <div
                className={cn(
                  "flex h-7 w-7 sm:h-8 sm:w-8 items-center justify-center rounded-full text-sm transition-colors",
                  isComplete && "bg-green-500 text-white",
                  isCurrent && "bg-blue-500 text-white",
                  !isComplete && !isCurrent && "bg-slate-200 text-slate-400"
                )}
              >
                {isComplete ? (
                  <Check size={16} />
                ) : isCurrent ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Circle size={12} />
                )}
              </div>
              <span
                className={cn(
                  "text-[10px] sm:text-[11px] font-medium",
                  isComplete && "text-green-600",
                  isCurrent && "text-blue-600",
                  !isComplete && !isCurrent && "text-slate-400"
                )}
              >
                {step.label}
              </span>
            </div>

            {i < STEPS.length - 1 && (
              <div
                className={cn(
                  "mx-1 sm:mx-2 h-0.5 w-5 sm:w-8 md:w-12 rounded",
                  i < currentIdx ? "bg-green-400" : "bg-slate-200"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
