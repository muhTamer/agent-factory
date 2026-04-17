"use client";

import { useSetupStore } from "@/store/setupStore";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ShoppingBag,
  Landmark,
  Smartphone,
  Headphones,
  Zap,
  ArrowRight,
  Loader2,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { Vertical } from "@/types/concierge";
import { authFetch } from "@/lib/auth-fetch";
import { startRuntime } from "@/lib/concierge-api";

const DOMAINS: {
  value: Vertical;
  label: string;
  description: string;
  icon: React.ElementType;
}[] = [
  {
    value: "retail",
    label: "Retail",
    description: "E-commerce, orders, returns",
    icon: ShoppingBag,
  },
  {
    value: "fintech",
    label: "Fintech",
    description: "Banking, payments, refunds",
    icon: Landmark,
  },
  {
    value: "telco",
    label: "Telecom",
    description: "Plans, billing, support",
    icon: Smartphone,
  },
  {
    value: "general_service",
    label: "General",
    description: "Custom service domain",
    icon: Headphones,
  },
];

export function WelcomeStep() {
  const vertical = useSetupStore((s) => s.vertical);
  const setVertical = useSetupStore((s) => s.setVertical);
  const setStep = useSetupStore((s) => s.setStep);
  const setPlan = useSetupStore((s) => s.setPlan);
  const setAnalysisSummaryText = useSetupStore((s) => s.setAnalysisSummaryText);
  const setQuickstart = useSetupStore((s) => s.setQuickstart);
  const setDeployment = useSetupStore((s) => s.setDeployment);
  const setDeployMessage = useSetupStore((s) => s.setDeployMessage);
  const setError = useSetupStore((s) => s.setError);

  const [quickLoading, setQuickLoading] = useState(false);
  const [quickStatus, setQuickStatus] = useState("");
  const [quickstartOpen, setQuickstartOpen] = useState(false);

  async function handleQuickstart(variant: "fintech" | "retail") {
    if (quickLoading) return; // guard against double tap
    setQuickLoading(true);
    setError(null);
    setQuickStatus("Starting quickstart...");
    // Let React render the loading state before starting fetch
    await new Promise((r) => setTimeout(r, 100));
    try {
      // Step 1: Start quickstart job
      const endpoint = variant === "retail"
        ? "/api/concierge/quickstart-retail"
        : "/api/concierge/quickstart-fintech";
      const startRes = await authFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ use_llm: true, model: "gpt-5-mini" }),
      });
      if (!startRes.ok) {
        const errText = await startRes.text().catch(() => startRes.statusText);
        throw new Error(`Quickstart failed (${startRes.status}): ${errText}`);
      }
      const startData = await startRes.json();

      // Step 2: Poll until quickstart completes (or use synchronous result)
      let qsResult: Record<string, unknown>;
      if (startData.job_id) {
        let polled: Record<string, unknown> | null = null;
        while (!polled) {
          await new Promise((r) => setTimeout(r, 2000));
          const pollRes = await authFetch(`/api/concierge/job/${startData.job_id}`);
          const pollData = await pollRes.json();
          if (pollData.status === "done") {
            polled = pollData.result;
          } else if (pollData.status === "error") {
            throw new Error(`Quickstart: ${pollData.error}`);
          } else {
            setQuickStatus(`Analyzing preset documents... (${Math.round(pollData.elapsed ?? 0)}s)`);
          }
        }
        qsResult = polled;
      } else {
        // Synchronous response (fallback)
        qsResult = startData;
      }
      setVertical(variant);
      setQuickstart(true);
      setPlan((qsResult as Record<string, never>).plan);
      setAnalysisSummaryText((qsResult as Record<string, never>).text);

      // Step 3: Start deploy job
      setQuickStatus("Generating agents & deploying...");
      const depRes = await authFetch("/api/concierge/deploy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "dry", doc_visibility: null }),
      });
      if (!depRes.ok) {
        const errText = await depRes.text().catch(() => depRes.statusText);
        throw new Error(`Deploy failed (${depRes.status}): ${errText}`);
      }
      const depStart = await depRes.json();

      // Step 4: Poll until deploy completes
      if (depStart.job_id) {
        let depResult: Record<string, unknown> | null = null;
        while (!depResult) {
          await new Promise((r) => setTimeout(r, 2000));
          const pollRes = await authFetch(`/api/concierge/job/${depStart.job_id}`);
          const pollData = await pollRes.json();
          if (pollData.status === "done") {
            depResult = pollData.result;
          } else if (pollData.status === "error") {
            throw new Error(`Deploy: ${pollData.error}`);
          } else {
            setQuickStatus(`Generating agents & deploying... (${Math.round(pollData.elapsed ?? 0)}s)`);
          }
        }
        setDeployment((depResult as Record<string, never>).deployment_request);
        setDeployMessage((depResult as Record<string, never>).text);
      }

      // Start runtime automatically so user doesn't have to click "Start"
      setQuickStatus("Starting runtime...");
      try {
        await startRuntime();
      } catch {
        // Non-fatal — RuntimeStep will show Start button as fallback
      }

      // Jump straight to runtime step
      setStep("runtime");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Quickstart failed");
    } finally {
      setQuickLoading(false);
      setQuickStatus("");
    }
  }

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center">
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900">Agent Factory</h1>
        <p className="mt-2 text-slate-500">
          Build your multi-agent customer service system in minutes
        </p>
      </div>

      {/* Domain picker */}
      <div>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">
          Select your business domain
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {DOMAINS.map((d) => {
            const Icon = d.icon;
            const selected = vertical === d.value;
            return (
              <Card
                key={d.value}
                onClick={() => setVertical(d.value)}
                className={cn(
                  "cursor-pointer transition-all",
                  selected
                    ? "ring-2 ring-blue-500 bg-blue-50/50"
                    : "hover:border-slate-300"
                )}
              >
                <CardContent className="flex items-center gap-3 p-4">
                  <div
                    className={cn(
                      "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg",
                      selected
                        ? "bg-blue-500 text-white"
                        : "bg-slate-100 text-slate-500"
                    )}
                  >
                    <Icon size={20} />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-800">
                      {d.label}
                    </p>
                    <p className="text-xs text-slate-400">{d.description}</p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>

      {/* Quickstart */}
      <div className="space-y-3">
        <Card className="border-slate-200">
          <button
            type="button"
            onClick={() => setQuickstartOpen((o) => !o)}
            className="flex w-full items-center justify-between p-4 text-left"
          >
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-violet-500 text-white">
                <Zap size={20} />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  Quickstart with Preset Data
                </p>
                <p className="text-xs text-slate-500">
                  Test the system instantly using built-in sample documents
                </p>
              </div>
            </div>
            <ChevronDown
              size={18}
              className={cn(
                "text-slate-400 transition-transform",
                quickstartOpen && "rotate-180"
              )}
            />
          </button>

          {quickstartOpen && (
            <div className="border-t border-slate-100 px-4 pb-4 pt-3 space-y-4">
              <p className="text-xs text-slate-500 leading-relaxed">
                Quickstart uses preset data so you can explore the full pipeline
                without uploading anything. <strong>Fintech</strong> loads
                synthetic bank FAQs and a refund policy.{" "}
                <strong>Retail</strong> loads FAQs scraped from IKEA alongside
                auto-generated refund and complaint policies.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {/* Fintech */}
                <Card className="border-amber-200 bg-amber-50/50">
                  <CardContent className="flex flex-col gap-3 p-4">
                    <div className="flex items-center gap-3">
                      <Landmark size={18} className="text-amber-600 shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-slate-800">Fintech</p>
                        <p className="text-xs text-slate-500">
                          Bank FAQs & refund policy
                        </p>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => handleQuickstart("fintech")}
                      disabled={quickLoading}
                      className="w-full"
                    >
                      {quickLoading ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <>
                          <Zap size={14} />
                          Launch
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>

                {/* Retail */}
                <Card className="border-emerald-200 bg-emerald-50/50">
                  <CardContent className="flex flex-col gap-3 p-4">
                    <div className="flex items-center gap-3">
                      <ShoppingBag size={18} className="text-emerald-600 shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-slate-800">Retail</p>
                        <p className="text-xs text-slate-500">
                          IKEA FAQs, refund & complaint policies
                        </p>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => handleQuickstart("retail")}
                      disabled={quickLoading}
                      className="w-full"
                    >
                      {quickLoading ? (
                        <Loader2 size={16} className="animate-spin" />
                      ) : (
                        <>
                          <ShoppingBag size={14} />
                          Launch
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </Card>

        {quickLoading && quickStatus && (
          <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-700">
            <Loader2 size={14} className="animate-spin" />
            {quickStatus}
            <p className="mt-1 text-xs text-amber-600">
              You can switch apps — progress won&apos;t be lost.
            </p>
          </div>
        )}
      </div>

      {/* Continue */}
      <div className="flex justify-end">
        <Button onClick={() => setStep("upload")} disabled={quickLoading}>
          Continue
          <ArrowRight size={16} />
        </Button>
      </div>

    </div>
  );
}
