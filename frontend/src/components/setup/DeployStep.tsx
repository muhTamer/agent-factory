"use client";

import { useSetupStore } from "@/store/setupStore";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, ArrowRight, Check, AlertTriangle } from "lucide-react";

export function DeployStep() {
  const deployment = useSetupStore((s) => s.deployment);
  const deployMessage = useSetupStore((s) => s.deployMessage);
  const setStep = useSetupStore((s) => s.setStep);

  if (!deployment) {
    return (
      <div className="text-center text-sm text-slate-400">
        No deployment info yet.
      </div>
    );
  }

  const hasErrors = deployment.generation_errors.length > 0;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Deployment Ready</h2>
        <p className="mt-1 text-sm text-slate-500">{deployMessage}</p>
      </div>

      {/* Generated agents */}
      <Card>
        <CardContent className="space-y-3 p-4">
          <h3 className="text-sm font-semibold text-slate-700">
            Generated Agents
          </h3>
          <ul className="space-y-2">
            {deployment.generated_agents.map((a) => (
              <li key={a.id} className="flex items-center gap-2 text-sm">
                <Check size={14} className="text-green-600" />
                <span className="font-medium text-slate-700">{a.id}</span>
              </li>
            ))}
          </ul>

          {hasErrors && (
            <div className="mt-3 space-y-1">
              <p className="flex items-center gap-1 text-sm font-medium text-amber-700">
                <AlertTriangle size={14} />
                Generation Warnings
              </p>
              {deployment.generation_errors.map((e) => (
                <p key={e.id} className="text-xs text-red-600">
                  {e.id}: {e.error}
                </p>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Spec info */}
      <Card className="bg-slate-50">
        <CardContent className="space-y-2 p-4">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-500">Spec path:</span>
            <code className="rounded bg-slate-200 px-2 py-0.5 text-xs">
              {deployment.spec_path}
            </code>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-slate-500">Runtime URL:</span>
            <Badge variant="outline">{deployment.runtime.base_url}</Badge>
          </div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between">
        <Button variant="outline" onClick={() => setStep("analysis")}>
          <ArrowLeft size={16} />
          Back
        </Button>
        <Button onClick={() => setStep("runtime")}>
          Start Runtime
          <ArrowRight size={16} />
        </Button>
      </div>
    </div>
  );
}
