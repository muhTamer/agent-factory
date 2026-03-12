"use client";

import { useState, useEffect, useCallback } from "react";
import { API_BASE } from "@/lib/constants";
import { Shield, ToggleLeft, ToggleRight, RefreshCw, AlertTriangle } from "lucide-react";
import { CollapsibleSection } from "./CollapsibleSection";

interface GuardrailRule {
  id: string;
  label: string;
  description: string;
  category: string;
  severity: string;
  enabled: boolean;
  patterns: string[];
}

interface GuardrailsResponse {
  rules: GuardrailRule[];
  transaction_slot_keys: string[];
  policy_pack: string;
  version: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  high: "bg-red-100 text-red-700",
  medium: "bg-amber-100 text-amber-700",
  low: "bg-slate-100 text-slate-600",
};

const CATEGORY_COLORS: Record<string, string> = {
  safety: "bg-red-50 text-red-600 border-red-200",
  tone: "bg-purple-50 text-purple-600 border-purple-200",
  internal: "bg-slate-50 text-slate-600 border-slate-200",
  privacy: "bg-blue-50 text-blue-600 border-blue-200",
  general: "bg-slate-50 text-slate-500 border-slate-200",
};

export function GuardrailsAdminPanel() {
  const [data, setData] = useState<GuardrailsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);

  const fetchRules = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/guardrails`);
      if (!res.ok) throw new Error(`Failed to fetch guardrails: ${res.status}`);
      const json: GuardrailsResponse = await res.json();
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load guardrails");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRules();
  }, [fetchRules]);

  const toggleRule = async (ruleId: string, enabled: boolean) => {
    setToggling(ruleId);
    try {
      const res = await fetch(`${API_BASE}/guardrails/${ruleId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) throw new Error(`Toggle failed: ${res.status}`);
      setData((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          rules: prev.rules.map((r) =>
            r.id === ruleId ? { ...r, enabled } : r
          ),
        };
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Toggle failed");
    } finally {
      setToggling(null);
    }
  };

  const enabledCount = data?.rules.filter((r) => r.enabled).length ?? 0;
  const totalCount = data?.rules.length ?? 0;

  return (
    <CollapsibleSection
      icon={<Shield size={14} className="text-amber-500" />}
      title="Guardrail Rules"
      tooltip="Configure which guardrail rules are active — toggle rules on/off for testing. Changes persist to disk and take effect immediately."
      status={error ? "error" : "info"}
      defaultOpen={false}
      badge={
        data ? (
          <span className="text-[10px] font-normal text-slate-400">
            {enabledCount}/{totalCount} active
          </span>
        ) : undefined
      }
      collapsedSummary={
        data ? (
          <div className="flex flex-wrap gap-1.5 text-[10px]">
            {Object.entries(
              data.rules.reduce<Record<string, { enabled: number; total: number }>>((acc, r) => {
                if (!acc[r.category]) acc[r.category] = { enabled: 0, total: 0 };
                acc[r.category].total++;
                if (r.enabled) acc[r.category].enabled++;
                return acc;
              }, {})
            ).map(([cat, counts]) => (
              <span
                key={cat}
                className={`rounded-full border px-2 py-0.5 font-medium ${CATEGORY_COLORS[cat] || CATEGORY_COLORS.general}`}
              >
                {cat}: {counts.enabled}/{counts.total}
              </span>
            ))}
          </div>
        ) : undefined
      }
    >
      <div className="space-y-2">
        {/* Header with refresh */}
        <div className="flex items-center justify-between">
          {data && (
            <span className="text-[10px] text-slate-400">
              Pack: <span className="font-medium text-slate-600">{data.policy_pack}</span> v{data.version}
            </span>
          )}
          <button
            onClick={fetchRules}
            disabled={loading}
            className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] text-slate-400 hover:bg-slate-50 hover:text-slate-600 disabled:opacity-50"
            title="Refresh rules from backend"
          >
            <RefreshCw size={10} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        {error && (
          <div className="flex items-center gap-1.5 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
            <AlertTriangle size={12} className="shrink-0" />
            {error}
          </div>
        )}

        {loading && !data && (
          <div className="py-4 text-center text-xs text-slate-400">
            <RefreshCw size={14} className="mx-auto mb-1 animate-spin text-slate-300" />
            Loading guardrail rules...
          </div>
        )}

        {data && (
          <div className="space-y-1.5">
            {data.rules.map((rule) => (
              <div
                key={rule.id}
                className={`rounded-lg border px-3 py-2 transition-colors ${
                  rule.enabled
                    ? "border-slate-200 bg-white"
                    : "border-slate-100 bg-slate-50/50"
                }`}
              >
                <div className="flex items-center gap-2">
                  {/* Toggle button */}
                  <button
                    onClick={() => toggleRule(rule.id, !rule.enabled)}
                    disabled={toggling === rule.id}
                    className="shrink-0 transition-colors disabled:opacity-50"
                    title={rule.enabled ? "Click to disable this rule" : "Click to enable this rule"}
                  >
                    {rule.enabled ? (
                      <ToggleRight size={20} className="text-green-500" />
                    ) : (
                      <ToggleLeft size={20} className="text-slate-300" />
                    )}
                  </button>

                  {/* Rule info */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span
                        className={`text-xs font-medium ${
                          rule.enabled ? "text-slate-700" : "text-slate-400"
                        }`}
                      >
                        {rule.label}
                      </span>
                      <span
                        className={`rounded-full border px-1.5 py-0 text-[9px] font-medium ${
                          CATEGORY_COLORS[rule.category] || CATEGORY_COLORS.general
                        }`}
                      >
                        {rule.category}
                      </span>
                      <span
                        className={`rounded-full px-1.5 py-0 text-[9px] font-medium ${
                          SEVERITY_COLORS[rule.severity] || SEVERITY_COLORS.medium
                        }`}
                      >
                        {rule.severity}
                      </span>
                    </div>
                    {rule.description && (
                      <p
                        className={`mt-0.5 text-[11px] leading-snug ${
                          rule.enabled ? "text-slate-500" : "text-slate-400"
                        }`}
                      >
                        {rule.description}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </CollapsibleSection>
  );
}
