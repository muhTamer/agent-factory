"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSetupStore } from "@/store/setupStore";
import { useThreadStore } from "@/store/threadStore";
import { getSession, getActiveJob, startRuntime } from "@/lib/concierge-api";
import { authFetch } from "@/lib/auth-fetch";
import type { Vertical, DeploymentInfo } from "@/types/concierge";
import { WizardProgressBar } from "./WizardProgressBar";
import { WelcomeStep } from "./WelcomeStep";
import { UploadStep } from "./UploadStep";
import { AnalysisStep } from "./AnalysisStep";
import { DeployStep } from "./DeployStep";
import { RuntimeStep } from "./RuntimeStep";
import { UserMenu } from "../UserMenu";
import { AlertTriangle, X, Loader2 } from "lucide-react";

async function waitForAgents(
  cancelled: () => boolean,
  setStatus: (s: string) => void,
) {
  const maxWait = 120_000;
  const t0 = Date.now();
  while (Date.now() - t0 < maxWait && !cancelled()) {
    try {
      const h = await authFetch("/api/runtime/health", { cache: "no-store" });
      if (h.ok) {
        const data = await h.json();
        if (data.status === "ok" && Object.keys(data.agents || {}).length > 0) {
          return;
        }
      }
    } catch { /* runtime may not be up yet */ }
    await new Promise((r) => setTimeout(r, 2000));
    const elapsed = Math.round((Date.now() - t0) / 1000);
    setStatus(`Loading agents... (${elapsed}s)`);
  }
}

export function SetupWizard() {
  const router = useRouter();
  const currentStep = useSetupStore((s) => s.currentStep);
  const error = useSetupStore((s) => s.error);
  const setError = useSetupStore((s) => s.setError);
  const setStep = useSetupStore((s) => s.setStep);
  const setVertical = useSetupStore((s) => s.setVertical);
  const setDeployment = useSetupStore((s) => s.setDeployment);
  const setDeployMessage = useSetupStore((s) => s.setDeployMessage);

  const setPlan = useSetupStore((s) => s.setPlan);
  const setAnalysisSummaryText = useSetupStore((s) => s.setAnalysisSummaryText);
  const setQuickstart = useSetupStore((s) => s.setQuickstart);

  const [restoring, setRestoring] = useState(true);
  const [resumeStatus, setResumeStatus] = useState("");

  // Helper: poll a running job then start runtime and navigate to chat
  async function resumeJob(jobId: string, cancelled: () => boolean) {
    setResumeStatus("Setting up your system...");

    // Poll job until done
    let result: Record<string, unknown> | null = null;
    while (!cancelled()) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const res = await authFetch(`/api/concierge/job/${jobId}`);
        const data = await res.json();
        if (data.status === "done") {
          result = data.result;
          break;
        }
        if (data.status === "error") throw new Error(data.error);
        const elapsed = Math.round(data.elapsed ?? 0);
        setResumeStatus(
          elapsed < 15
            ? `Analyzing documents... (${elapsed}s)`
            : `Generating & deploying agents... (${elapsed}s)`
        );
      } catch (err) {
        if (err instanceof Error && err.message) throw err;
      }
    }
    if (cancelled() || !result) return;

    // Persist state
    const vertical = (result.vertical as string) ?? "fintech";
    setVertical(vertical as Vertical);
    setQuickstart(true);
    if (result.plan) setPlan(result.plan as never);
    if (result.text) setAnalysisSummaryText(result.text as string);
    if (result.deployment_request) {
      setDeployment(result.deployment_request as unknown as DeploymentInfo);
      setDeployMessage((result.deploy_text as string) ?? "");
    }

    // Start runtime + poll health
    setResumeStatus("Starting agents...");
    try { await startRuntime(); } catch { /* ignore */ }
    await waitForAgents(cancelled, setResumeStatus);
    if (!cancelled()) {
      setStep("runtime");
      router.push("/chat");
    }
  }

  // On mount: check deployed session, then check active job, then show wizard
  useEffect(() => {
    let cancelled = false;
    const isCancelled = () => cancelled;
    (async () => {
      try {
        // Case 1: session already deployed (job finished while browser was away)
        const session = await getSession();
        if (cancelled) return;
        if (session.status === "deployed" && session.deployment_request) {
          setVertical(session.vertical as Vertical);
          setDeployment(session.deployment_request);
          setDeployMessage(session.deploy_text ?? "");
          setStep("runtime");
          useThreadStore.getState().loadFromBackend();

          try { await startRuntime(); } catch { /* ignore */ }
          try {
            const h = await authFetch("/api/runtime/health", { cache: "no-store" });
            if (h.ok) {
              const data = await h.json();
              if (data.status === "ok" && Object.keys(data.agents || {}).length > 0) {
                if (!cancelled) router.push("/chat");
              }
            }
          } catch { /* ignore — RuntimeStep will handle it */ }
          if (!cancelled) setRestoring(false);
          return;
        }

        // Case 2: quickstart job still running in background
        const activeJob = await getActiveJob();
        if (cancelled) return;
        if (activeJob.active && activeJob.job_id) {
          if (activeJob.status === "processing") {
            try {
              await resumeJob(activeJob.job_id, isCancelled);
            } catch (err) {
              console.warn("[Session] resume job failed:", err);
            }
          } else if (activeJob.status === "done" && activeJob.result) {
            // Job finished but session wasn't loaded yet — persist and go
            const r = activeJob.result;
            const v = (r.vertical as string) ?? "fintech";
            setVertical(v as Vertical);
            setQuickstart(true);
            if (r.deployment_request) {
              setDeployment(r.deployment_request as unknown as DeploymentInfo);
              setDeployMessage((r.deploy_text as string) ?? "");
            }
            setResumeStatus("Starting agents...");
            try { await startRuntime(); } catch { /* ignore */ }
            await waitForAgents(isCancelled, setResumeStatus);
            if (!cancelled) {
              setStep("runtime");
              router.push("/chat");
            }
          }
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
      <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-white/95 backdrop-blur-sm">
        <div className="flex flex-col items-center gap-4 text-center px-6">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-blue-100">
            <Loader2 size={32} className="animate-spin text-blue-600" />
          </div>
          <h2 className="text-xl font-semibold text-slate-800">
            {resumeStatus ? "Setting up your system" : "Loading your workspace..."}
          </h2>
          {resumeStatus && (
            <p className="text-sm text-slate-500 max-w-sm">{resumeStatus}</p>
          )}
          {resumeStatus && (
            <p className="text-xs text-slate-400 mt-2">
              This usually takes 30–60 seconds
            </p>
          )}
        </div>
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
