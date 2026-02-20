"use client";

import { useSetupStore } from "@/store/setupStore";
import { quickstartFintech, deployFactory } from "@/lib/concierge-api";
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
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { Vertical } from "@/types/concierge";

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

  async function handleQuickstart() {
    setQuickLoading(true);
    setError(null);
    try {
      // Step 1: Analyze
      setQuickStatus("Analyzing preset documents...");
      const res = await quickstartFintech();
      setVertical("fintech");
      setQuickstart(true);
      setPlan(res.plan);
      setAnalysisSummaryText(res.text);

      // Step 2: Auto-deploy
      setQuickStatus("Generating agents & deploying...");
      const dep = await deployFactory("dry");
      setDeployment(dep.deployment_request);
      setDeployMessage(dep.text);

      // Jump straight to runtime step
      setStep("runtime");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Quickstart failed"
      );
    } finally {
      setQuickLoading(false);
      setQuickStatus("");
    }
  }

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center">
        <h1 className="text-3xl font-bold text-slate-900">Agent Factory</h1>
        <p className="mt-2 text-slate-500">
          Build your multi-agent customer service system in minutes
        </p>
      </div>

      {/* Domain picker */}
      <div>
        <h2 className="mb-3 text-sm font-semibold text-slate-700">
          Select your business domain
        </h2>
        <div className="grid grid-cols-2 gap-3">
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
      <Card className="border-amber-200 bg-amber-50/50">
        <CardContent className="flex items-center justify-between gap-4 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-amber-500 text-white">
              <Zap size={20} />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-800">
                Quickstart: Fintech
              </p>
              <p className="text-xs text-slate-500">
                Load preset bank FAQs & refund policy, analyze & deploy in one
                click
              </p>
            </div>
          </div>
          <Button
            size="sm"
            onClick={handleQuickstart}
            disabled={quickLoading}
            className="shrink-0"
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
        {quickLoading && quickStatus && (
          <div className="flex items-center gap-2 border-t border-amber-200 px-4 py-2 text-sm text-amber-700">
            <Loader2 size={14} className="animate-spin" />
            {quickStatus}
          </div>
        )}
      </Card>

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
