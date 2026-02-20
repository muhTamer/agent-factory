"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSetupStore } from "@/store/setupStore";
import { useRuntimePoller } from "@/hooks/useRuntimePoller";
import { startRuntime, stopRuntime } from "@/lib/concierge-api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  ArrowLeft,
  MessageSquare,
  Loader2,
  Play,
  Square,
  Check,
  Wifi,
} from "lucide-react";

export function RuntimeStep() {
  const router = useRouter();
  const setStep = useSetupStore((s) => s.setStep);
  const deployment = useSetupStore((s) => s.deployment);
  const setError = useSetupStore((s) => s.setError);

  const [started, setStarted] = useState(false);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);

  const { online, agentCount } = useRuntimePoller(started);

  async function handleStart() {
    setError(null);
    setStarting(true);
    try {
      await startRuntime();
      setStarted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start runtime");
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    setStopping(true);
    try {
      await stopRuntime();
      setStarted(false);
    } catch {
      // best-effort
    } finally {
      setStopping(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Launch Runtime</h2>
        <p className="mt-1 text-sm text-slate-500">
          Start the agent runtime, then open the chat to test your system
        </p>
      </div>

      {/* Status card */}
      <Card
        className={
          online
            ? "border-green-200 bg-green-50/50"
            : "border-slate-200 bg-slate-50/50"
        }
      >
        <CardContent className="flex items-center gap-4 p-6">
          {online ? (
            <>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-500 text-white">
                <Wifi size={24} />
              </div>
              <div>
                <p className="text-lg font-semibold text-green-800">
                  Runtime is online
                </p>
                <p className="text-sm text-green-600">
                  {agentCount} agent{agentCount !== 1 ? "s" : ""} loaded and
                  ready
                </p>
              </div>
            </>
          ) : started ? (
            <>
              <Loader2 size={32} className="animate-spin text-blue-500" />
              <div>
                <p className="text-lg font-semibold text-slate-800">
                  Starting runtime...
                </p>
                <p className="text-sm text-slate-500">
                  Waiting for health check to pass
                </p>
              </div>
            </>
          ) : (
            <>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-200 text-slate-400">
                <Play size={24} />
              </div>
              <div>
                <p className="text-lg font-semibold text-slate-800">
                  Runtime is not running
                </p>
                <p className="text-sm text-slate-500">
                  Click Start to launch the agent runtime
                </p>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Command reference */}
      {deployment && (
        <Card className="bg-slate-50">
          <CardContent className="p-4">
            <p className="mb-2 text-xs font-medium text-slate-500">
              Manual command (if needed):
            </p>
            <code className="block rounded bg-slate-900 px-3 py-2 text-xs text-green-400">
              {deployment.uvicorn_command}
            </code>
          </CardContent>
        </Card>
      )}

      {/* Actions */}
      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          onClick={() => setStep("deploy")}
          disabled={starting}
        >
          <ArrowLeft size={16} />
          Back
        </Button>

        <div className="flex gap-2">
          {!started && (
            <Button onClick={handleStart} disabled={starting}>
              {starting ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <Play size={16} />
              )}
              Start Runtime
            </Button>
          )}

          {started && !online && (
            <Button variant="outline" onClick={handleStop} disabled={stopping}>
              <Square size={16} />
              Stop
            </Button>
          )}

          {online && (
            <Button
              onClick={() => router.push("/chat")}
              className="bg-green-600 hover:bg-green-700"
            >
              <MessageSquare size={16} />
              Open Chat
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
