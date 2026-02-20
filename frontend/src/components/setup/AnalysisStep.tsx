"use client";

import { useSetupStore } from "@/store/setupStore";
import { deployFactory } from "@/lib/concierge-api";
import { AgentCard } from "./AgentCard";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft, Rocket, Loader2 } from "lucide-react";

export function AnalysisStep() {
  const plan = useSetupStore((s) => s.plan);
  const analysisSummaryText = useSetupStore((s) => s.analysisSummaryText);
  const isDeploying = useSetupStore((s) => s.isDeploying);
  const setDeploying = useSetupStore((s) => s.setDeploying);
  const setDeployment = useSetupStore((s) => s.setDeployment);
  const setDeployMessage = useSetupStore((s) => s.setDeployMessage);
  const setStep = useSetupStore((s) => s.setStep);
  const setError = useSetupStore((s) => s.setError);
  const isQuickstart = useSetupStore((s) => s.isQuickstart);

  async function handleDeploy() {
    setError(null);
    setDeploying(true);
    try {
      const res = await deployFactory("dry");
      setDeployment(res.deployment_request);
      setDeployMessage(res.text);
      setStep("deploy");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Deployment failed");
    } finally {
      setDeploying(false);
    }
  }

  if (!plan) {
    return (
      <div className="text-center text-sm text-slate-400">
        No analysis results yet.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Analysis Results</h2>
        <p className="mt-1 text-sm text-slate-500">
          {plan.agents.length} agent{plan.agents.length !== 1 ? "s" : ""}{" "}
          proposed for <span className="font-medium">{plan.vertical}</span>{" "}
          domain
        </p>
      </div>

      {/* Summary */}
      <Card className="bg-slate-50">
        <CardContent className="p-4">
          <p className="whitespace-pre-wrap text-sm text-slate-600">
            {plan.summary}
          </p>
        </CardContent>
      </Card>

      {/* Agent cards */}
      <div className="grid gap-4 md:grid-cols-2">
        {plan.agents.map((agent) => (
          <AgentCard key={agent.id} agent={agent} />
        ))}
      </div>

      {isDeploying && (
        <div className="flex items-center gap-2 text-sm text-blue-600">
          <Loader2 size={16} className="animate-spin" />
          Generating agents & building deployment spec...
        </div>
      )}

      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          onClick={() => setStep(isQuickstart ? "welcome" : "upload")}
          disabled={isDeploying}
        >
          <ArrowLeft size={16} />
          Back
        </Button>
        <Button onClick={handleDeploy} disabled={isDeploying}>
          {isDeploying ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Rocket size={16} />
          )}
          Approve & Deploy
        </Button>
      </div>
    </div>
  );
}
