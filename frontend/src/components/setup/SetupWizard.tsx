"use client";

import { useEffect, useState } from "react";
import { useSetupStore } from "@/store/setupStore";
import { getSession } from "@/lib/concierge-api";
import type { Vertical } from "@/types/concierge";
import { WizardProgressBar } from "./WizardProgressBar";
import { WelcomeStep } from "./WelcomeStep";
import { UploadStep } from "./UploadStep";
import { AnalysisStep } from "./AnalysisStep";
import { DeployStep } from "./DeployStep";
import { RuntimeStep } from "./RuntimeStep";
import { UserMenu } from "../UserMenu";
import { AlertTriangle, X, Loader2 } from "lucide-react";

export function SetupWizard() {
  const currentStep = useSetupStore((s) => s.currentStep);
  const error = useSetupStore((s) => s.error);
  const setError = useSetupStore((s) => s.setError);
  const setStep = useSetupStore((s) => s.setStep);
  const setVertical = useSetupStore((s) => s.setVertical);
  const setDeployment = useSetupStore((s) => s.setDeployment);
  const setDeployMessage = useSetupStore((s) => s.setDeployMessage);

  const [restoring, setRestoring] = useState(true);

  // On mount, check if the user has an existing deployment
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const session = await getSession();
        if (cancelled) return;
        if (session.status === "deployed" && session.deployment_request) {
          setVertical(session.vertical as Vertical);
          setDeployment(session.deployment_request);
          setDeployMessage(session.deploy_text ?? "");
          setStep("runtime");
        }
      } catch (err) {
        console.warn("[Session] restore failed:", err);
      } finally {
        if (!cancelled) setRestoring(false);
      }
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (restoring) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-2">
        <Loader2 size={20} className="animate-spin text-blue-500" />
        <span className="text-sm text-slate-500">Loading your workspace...</span>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Top bar with user menu */}
      <div className="flex items-center justify-end px-4 sm:px-6 py-3">
        <UserMenu />
      </div>

      <div className="mx-auto max-w-3xl px-3 sm:px-4 pb-12">
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
