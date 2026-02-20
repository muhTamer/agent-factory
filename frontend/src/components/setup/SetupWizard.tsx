"use client";

import { useSetupStore } from "@/store/setupStore";
import { WizardProgressBar } from "./WizardProgressBar";
import { WelcomeStep } from "./WelcomeStep";
import { UploadStep } from "./UploadStep";
import { AnalysisStep } from "./AnalysisStep";
import { DeployStep } from "./DeployStep";
import { RuntimeStep } from "./RuntimeStep";
import { AlertTriangle, X } from "lucide-react";

export function SetupWizard() {
  const currentStep = useSetupStore((s) => s.currentStep);
  const error = useSetupStore((s) => s.error);
  const setError = useSetupStore((s) => s.setError);

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-3xl px-4 py-12">
        <WizardProgressBar currentStep={currentStep} />

        {/* Global error banner */}
        {error && (
          <div className="mb-6 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-red-500" />
            <p className="flex-1 text-sm text-red-700">{error}</p>
            <button
              onClick={() => setError(null)}
              className="shrink-0 rounded p-1 text-red-400 hover:bg-red-100"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {currentStep === "welcome" && <WelcomeStep />}
        {currentStep === "upload" && <UploadStep />}
        {currentStep === "analysis" && <AnalysisStep />}
        {currentStep === "deploy" && <DeployStep />}
        {currentStep === "runtime" && <RuntimeStep />}
      </div>
    </div>
  );
}
