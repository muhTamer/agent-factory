"use client";

import { useState, useEffect, useCallback } from "react";
import { authFetch } from "@/lib/auth-fetch";
import { Brain, RefreshCw, AlertTriangle } from "lucide-react";
import { CollapsibleSection } from "./CollapsibleSection";

interface EstimatorResponse {
  kind: string;
  options: string[];
}

const KIND_LABELS: Record<string, string> = {
  neural: "Neural (MLP)",
  tfidf: "TF-IDF",
};

const KIND_COLORS: Record<string, string> = {
  neural: "bg-purple-100 text-purple-700 border-purple-300",
  tfidf: "bg-slate-100 text-slate-600 border-slate-300",
};

const DEFAULT_OPTIONS = ["neural", "tfidf"];

export function EstimatorTogglePanel() {
  const [activeKind, setActiveKind] = useState<string>("neural");
  const [options, setOptions] = useState<string[]>(DEFAULT_OPTIONS);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  const fetchEstimator = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await authFetch(`/api/runtime/solvability-estimator`);
      if (!res.ok) throw new Error(`${res.status}`);
      const json: EstimatorResponse = await res.json();
      setActiveKind(json.kind);
      setOptions(json.options?.length ? json.options : DEFAULT_OPTIONS);
      setConnected(true);
    } catch (e) {
      setError(
        e instanceof Error
          ? `Backend unreachable (${e.message})`
          : "Backend unreachable"
      );
      setConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEstimator();
  }, [fetchEstimator]);

  const switchEstimator = async (kind: string) => {
    setSwitching(true);
    setError(null);
    try {
      const res = await authFetch(`/api/runtime/solvability-estimator`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind }),
      });
      if (!res.ok) throw new Error(`Switch failed: ${res.status}`);
      const json = await res.json();
      setActiveKind(json.kind);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Switch failed");
    } finally {
      setSwitching(false);
    }
  };

  return (
    <CollapsibleSection
      icon={<Brain size={14} className="text-purple-500" />}
      title="Solvability Estimator"
      tooltip="Switch between Neural (MLP + embeddings) and TF-IDF solvability estimators. Changes take effect immediately."
      status={error ? "warning" : "info"}
      defaultOpen={true}
      badge={
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${KIND_COLORS[activeKind] || "bg-slate-100 text-slate-500"}`}
        >
          {KIND_LABELS[activeKind] || activeKind}
        </span>
      }
    >
      <div className="space-y-2">
        {/* Header with refresh */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-400">
            Agent selection reward model
          </span>
          <button
            onClick={fetchEstimator}
            disabled={loading}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-slate-400 hover:bg-slate-50 hover:text-slate-600 disabled:opacity-50"
            title="Refresh from backend"
          >
            <RefreshCw size={10} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="flex items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-600">
            <AlertTriangle size={12} className="shrink-0" />
            {error}
          </div>
        )}

        {/* Always show toggle buttons */}
        <div className="space-y-1.5">
          {options.map((kind) => {
            const isActive = kind === activeKind;
            return (
              <button
                key={kind}
                onClick={() => !isActive && switchEstimator(kind)}
                disabled={isActive || switching || !connected}
                className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-xs transition-colors ${
                  isActive
                    ? "border-purple-300 bg-purple-50 text-purple-700"
                    : "border-slate-200 bg-white text-slate-600 hover:border-purple-200 hover:bg-purple-50/50"
                } ${switching || !connected ? "opacity-50" : ""}`}
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`h-2 w-2 rounded-full ${
                      isActive ? "bg-purple-500" : "bg-slate-300"
                    }`}
                  />
                  <span className="font-medium">
                    {KIND_LABELS[kind] || kind}
                  </span>
                </div>
                {isActive && (
                  <span className="text-[10px] font-medium text-purple-500">
                    ACTIVE
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Description */}
        <div className="rounded-lg bg-slate-50 px-2.5 py-2 text-[10px] text-slate-400 leading-relaxed">
          <strong className="text-slate-500">Neural:</strong> Sentence
          embeddings + trained MLP. Better on semantic paraphrases.
          <br />
          <strong className="text-slate-500">TF-IDF:</strong> Token-based
          cosine similarity. Faster, fully deterministic.
        </div>
      </div>
    </CollapsibleSection>
  );
}
