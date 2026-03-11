"use client";

import { useEffect, useState, useCallback } from "react";
import { useSetupStore } from "@/store/setupStore";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChevronDown,
  ChevronRight,
  Plus,
  Trash2,
  Save,
  Loader2,
  Wrench,
  AlertTriangle,
} from "lucide-react";
import {
  getMcpToolsConfig,
  saveMcpToolsConfig,
} from "@/lib/concierge-api";
import type { McpToolDef, McpToolParameter, McpToolScenario } from "@/types/concierge";

// ── Helpers ─────────────────────────────────────────────────────────

function emptyTool(): McpToolDef {
  return {
    name: "",
    description: "",
    parameters: {},
    response: {},
    scenarios: [],
  };
}

function emptyScenario(): McpToolScenario {
  return { _comment: "", when: {}, response: {} };
}

const CONDITION_OPS = [
  "starts_with",
  "equals",
  "contains",
  "greater_than",
  "less_than",
  "in",
] as const;

// ── Main Component ──────────────────────────────────────────────────

export function ToolConfigStep() {
  const mcpTools = useSetupStore((s) => s.mcpTools);
  const setMcpTools = useSetupStore((s) => s.setMcpTools);
  const isSaving = useSetupStore((s) => s.isSavingTools);
  const setSaving = useSetupStore((s) => s.setSavingTools);
  const setError  = useSetupStore((s) => s.setError);

  const [expandedTool, setExpandedTool] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [dirty, setDirty] = useState(false);

  // Load config on mount
  useEffect(() => {
    getMcpToolsConfig()
      .then((data) => {
        setMcpTools(data.tools ?? []);
      })
      .catch((err) => setError(`Failed to load MCP tools: ${err.message}`))
      .finally(() => setIsLoading(false));
  }, [setMcpTools, setError]);

  const updateTool = useCallback(
    (index: number, updated: McpToolDef) => {
      const next = [...mcpTools];
      next[index] = updated;
      setMcpTools(next);
      setDirty(true);
    },
    [mcpTools, setMcpTools]
  );

  const removeTool = useCallback(
    (index: number) => {
      setMcpTools(mcpTools.filter((_, i) => i !== index));
      setDirty(true);
    },
    [mcpTools, setMcpTools]
  );

  const addTool = useCallback(() => {
    setMcpTools([...mcpTools, emptyTool()]);
    setDirty(true);
  }, [mcpTools, setMcpTools]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await saveMcpToolsConfig(mcpTools);
      setDirty(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Save failed: ${msg}`);
    } finally {
      setSaving(false);
    }
  }, [mcpTools, setSaving, setError]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
        <span className="ml-2 text-sm text-slate-500">
          Loading tool configuration...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-slate-500">
        Customize tool responses and scenarios. Changes take effect on next
        runtime start.
      </p>

      {/* Tool list */}
      <div className="space-y-3">
        {mcpTools.map((tool, idx) => (
          <ToolCard
            key={`${tool.name}-${idx}`}
            tool={tool}
            isExpanded={expandedTool === `${idx}`}
            onToggle={() =>
              setExpandedTool(expandedTool === `${idx}` ? null : `${idx}`)
            }
            onChange={(t) => updateTool(idx, t)}
            onRemove={() => removeTool(idx)}
          />
        ))}
      </div>

      {/* Add tool button */}
      <Button
        variant="outline"
        className="w-full border-dashed"
        onClick={addTool}
      >
        <Plus size={16} className="mr-1" />
        Add Tool
      </Button>

      {/* Save button */}
      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={isSaving || !dirty}>
          {isSaving ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Save size={16} />
          )}
          {dirty ? "Save Changes" : "Saved"}
        </Button>
      </div>
    </div>
  );
}

// ── Tool Card ───────────────────────────────────────────────────────

interface ToolCardProps {
  tool: McpToolDef;
  isExpanded: boolean;
  onToggle: () => void;
  onChange: (t: McpToolDef) => void;
  onRemove: () => void;
}

function ToolCard({
  tool,
  isExpanded,
  onToggle,
  onChange,
  onRemove,
}: ToolCardProps) {
  const paramCount = Object.keys(tool.parameters).length;
  const scenarioCount = tool.scenarios?.length ?? 0;

  return (
    <Card className="overflow-hidden">
      <Collapsible open={isExpanded} onOpenChange={onToggle}>
        <CollapsibleTrigger asChild>
          <button className="flex w-full items-center gap-3 p-4 text-left hover:bg-slate-50 overflow-hidden">
            {isExpanded ? (
              <ChevronDown size={16} className="shrink-0 text-slate-400" />
            ) : (
              <ChevronRight size={16} className="shrink-0 text-slate-400" />
            )}
            <Wrench size={16} className="shrink-0 text-blue-500" />
            <div className="min-w-0 flex-1 overflow-hidden">
              <div className="font-medium text-slate-800 truncate">
                {tool.name || "(unnamed)"}
              </div>
              <div className="text-xs text-slate-400 truncate">
                {tool.description}
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Badge variant="outline" className="text-[10px] whitespace-nowrap">
                {paramCount} param{paramCount !== 1 ? "s" : ""}
              </Badge>
              {scenarioCount > 0 && (
                <Badge variant="outline" className="text-[10px] whitespace-nowrap">
                  {scenarioCount} scenario{scenarioCount !== 1 ? "s" : ""}
                </Badge>
              )}
            </div>
          </button>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <CardContent className="space-y-4 border-t px-4 pb-4 pt-3">
            {/* Name & Description — stacked on small widths */}
            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">
                  Tool Name
                </label>
                <Input
                  value={tool.name}
                  onChange={(e) => onChange({ ...tool, name: e.target.value })}
                  placeholder="e.g. lookup_customer"
                  className="font-mono text-sm"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs font-medium text-slate-500">
                  Description
                </label>
                <Input
                  value={tool.description}
                  onChange={(e) =>
                    onChange({ ...tool, description: e.target.value })
                  }
                  placeholder="What this tool does..."
                  className="text-sm"
                />
              </div>
            </div>

            {/* Parameters */}
            <ParametersEditor
              parameters={tool.parameters}
              onChange={(p) => onChange({ ...tool, parameters: p })}
            />

            {/* Default Response */}
            <JsonEditor
              label="Default Response"
              value={tool.response}
              onChange={(r) => onChange({ ...tool, response: r })}
            />

            {/* Scenarios */}
            <ScenariosEditor
              scenarios={tool.scenarios ?? []}
              onChange={(s) => onChange({ ...tool, scenarios: s })}
            />

            {/* Remove tool */}
            <div className="flex justify-end">
              <Button
                variant="outline"
                size="sm"
                className="text-red-600 hover:bg-red-50"
                onClick={onRemove}
              >
                <Trash2 size={14} className="mr-1" />
                Remove Tool
              </Button>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

// ── Parameters Editor ───────────────────────────────────────────────

type ParamDef = McpToolParameter;

interface ParametersEditorProps {
  parameters: Record<string, ParamDef>;
  onChange: (p: Record<string, ParamDef>) => void;
}

function ParametersEditor({ parameters, onChange }: ParametersEditorProps) {
  const entries = Object.entries(parameters);

  const addParam = () => {
    const name = `param_${entries.length + 1}`;
    onChange({ ...parameters, [name]: { type: "string", required: true } });
  };

  const removeParam = (key: string) => {
    const next = { ...parameters };
    delete next[key];
    onChange(next);
  };

  const renameParam = (oldKey: string, newKey: string) => {
    if (newKey === oldKey || !newKey) return;
    const next: typeof parameters = {};
    for (const [k, v] of Object.entries(parameters)) {
      next[k === oldKey ? newKey : k] = v;
    }
    onChange(next);
  };

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <label className="text-xs font-medium text-slate-500">Parameters</label>
        <button
          onClick={addParam}
          className="flex items-center gap-1 text-xs text-blue-600 hover:underline"
        >
          <Plus size={12} /> Add
        </button>
      </div>
      {entries.length === 0 ? (
        <p className="text-xs text-slate-400 italic">No parameters defined.</p>
      ) : (
        <div className="space-y-2">
          {entries.map(([key, param]) => (
            <div
              key={key}
              className="flex flex-wrap items-center gap-2 rounded-md border border-slate-100 bg-slate-50/50 p-2"
            >
              {/* Name */}
              <Input
                value={key}
                onChange={(e) => renameParam(key, e.target.value)}
                className="min-w-0 flex-1 basis-28 font-mono text-xs"
                placeholder="name"
              />
              {/* Type */}
              <select
                value={param.type}
                onChange={(e) =>
                  onChange({
                    ...parameters,
                    [key]: { ...param, type: e.target.value as ParamDef["type"] },
                  })
                }
                className="h-9 shrink-0 rounded-md border border-slate-200 bg-white px-2 text-xs"
              >
                <option value="string">string</option>
                <option value="number">number</option>
                <option value="integer">integer</option>
                <option value="boolean">boolean</option>
              </select>
              {/* Required */}
              <label className="flex shrink-0 items-center gap-1 text-xs text-slate-500">
                <input
                  type="checkbox"
                  checked={param.required ?? false}
                  onChange={(e) =>
                    onChange({
                      ...parameters,
                      [key]: { ...param, required: e.target.checked },
                    })
                  }
                />
                req
              </label>
              {/* Default (only when not required) */}
              {!param.required && (
                <Input
                  value={String(param.default ?? "")}
                  onChange={(e) =>
                    onChange({
                      ...parameters,
                      [key]: { ...param, default: e.target.value },
                    })
                  }
                  className="min-w-0 flex-1 basis-20 text-xs"
                  placeholder="default"
                />
              )}
              {/* Delete */}
              <button
                onClick={() => removeParam(key)}
                className="shrink-0 rounded p-1 text-red-400 hover:bg-red-50 hover:text-red-600"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── JSON Editor (for response objects) ──────────────────────────────

interface JsonEditorProps {
  label: string;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
}

function JsonEditor({ label, value, onChange }: JsonEditorProps) {
  const [raw, setRaw] = useState(() => JSON.stringify(value, null, 2));
  const [parseError, setParseError] = useState<string | null>(null);

  // Sync from parent when value changes externally
  useEffect(() => {
    const parentStr = JSON.stringify(value, null, 2);
    try {
      const currentParsed = JSON.parse(raw);
      if (JSON.stringify(currentParsed, null, 2) !== parentStr) {
        setRaw(parentStr);
      }
    } catch {
      setRaw(parentStr);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  const handleChange = (text: string) => {
    setRaw(text);
    try {
      const parsed = JSON.parse(text);
      if (typeof parsed === "object" && parsed !== null) {
        setParseError(null);
        onChange(parsed);
      } else {
        setParseError("Must be a JSON object");
      }
    } catch {
      setParseError("Invalid JSON");
    }
  };

  return (
    <div className="overflow-hidden">
      <label className="mb-1 block text-xs font-medium text-slate-500">
        {label}
      </label>
      <textarea
        value={raw}
        onChange={(e) => handleChange(e.target.value)}
        rows={Math.min(Object.keys(value).length + 2, 8)}
        className="block w-full resize-y rounded-md border border-slate-200 bg-slate-50 p-2 font-mono text-xs leading-relaxed focus:border-blue-300 focus:outline-none focus:ring-1 focus:ring-blue-300"
        spellCheck={false}
      />
      {parseError && (
        <p className="mt-0.5 flex items-center gap-1 text-xs text-red-500">
          <AlertTriangle size={10} className="shrink-0" />
          {parseError}
        </p>
      )}
    </div>
  );
}

// ── Scenarios Editor ────────────────────────────────────────────────

interface ScenariosEditorProps {
  scenarios: McpToolScenario[];
  onChange: (s: McpToolScenario[]) => void;
}

function ScenariosEditor({ scenarios, onChange }: ScenariosEditorProps) {
  const addScenario = () => {
    onChange([...scenarios, emptyScenario()]);
  };

  const removeScenario = (idx: number) => {
    onChange(scenarios.filter((_, i) => i !== idx));
  };

  const updateScenario = (idx: number, updated: McpToolScenario) => {
    const next = [...scenarios];
    next[idx] = updated;
    onChange(next);
  };

  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <label className="text-xs font-medium text-slate-500">
          Conditional Scenarios
        </label>
        <button
          onClick={addScenario}
          className="flex items-center gap-1 text-xs text-blue-600 hover:underline"
        >
          <Plus size={12} /> Add Scenario
        </button>
      </div>

      {scenarios.length === 0 ? (
        <p className="text-xs text-slate-400 italic">
          No scenarios. Tool always returns the default response.
        </p>
      ) : (
        <div className="space-y-3">
          {scenarios.map((scenario, idx) => (
            <ScenarioCard
              key={idx}
              scenario={scenario}
              onChange={(s) => updateScenario(idx, s)}
              onRemove={() => removeScenario(idx)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Scenario Card ───────────────────────────────────────────────────

interface ScenarioCardProps {
  scenario: McpToolScenario;
  onChange: (s: McpToolScenario) => void;
  onRemove: () => void;
}

function ScenarioCard({ scenario, onChange, onRemove }: ScenarioCardProps) {
  const whenEntries = Object.entries(scenario.when);

  const addCondition = () => {
    onChange({
      ...scenario,
      when: { ...scenario.when, "": { starts_with: "" } },
    });
  };

  const removeCondition = (paramKey: string) => {
    const next = { ...scenario.when };
    delete next[paramKey];
    onChange({ ...scenario, when: next });
  };

  const updateCondition = (
    oldParam: string,
    newParam: string,
    op: string,
    val: unknown
  ) => {
    const next: Record<string, Record<string, unknown>> = {};
    for (const [k, v] of Object.entries(scenario.when)) {
      if (k === oldParam) {
        next[newParam || oldParam] = { [op]: val };
      } else {
        next[k] = v;
      }
    }
    onChange({ ...scenario, when: next });
  };

  return (
    <div className="overflow-hidden rounded-md border border-slate-200 bg-slate-50 p-3">
      {/* Comment + delete */}
      <div className="mb-2 flex items-center gap-2">
        <Input
          value={scenario._comment ?? ""}
          onChange={(e) =>
            onChange({ ...scenario, _comment: e.target.value })
          }
          placeholder="Scenario label (e.g. 'Suspended account')"
          className="min-w-0 flex-1 text-xs"
        />
        <button
          onClick={onRemove}
          className="shrink-0 rounded p-1 text-red-400 hover:bg-red-100 hover:text-red-600"
        >
          <Trash2 size={14} />
        </button>
      </div>

      {/* When conditions */}
      <div className="mb-2">
        <div className="mb-1.5 flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            When
          </span>
          <button
            onClick={addCondition}
            className="flex items-center gap-1 text-[10px] text-blue-600 hover:underline"
          >
            <Plus size={10} /> Condition
          </button>
        </div>
        {whenEntries.length === 0 ? (
          <p className="text-xs text-slate-400 italic">
            No conditions — add one above.
          </p>
        ) : (
          <div className="space-y-2">
            {whenEntries.map(([param, opObj]) => {
              const [op, val] = Object.entries(opObj)[0] ?? [
                "starts_with",
                "",
              ];
              return (
                <div
                  key={param}
                  className="flex flex-wrap items-center gap-1.5"
                >
                  {/* Param name */}
                  <Input
                    value={param}
                    onChange={(e) =>
                      updateCondition(param, e.target.value, op, val)
                    }
                    className="min-w-0 flex-1 basis-24 font-mono text-xs"
                    placeholder="param"
                  />
                  {/* Operator */}
                  <select
                    value={op}
                    onChange={(e) =>
                      updateCondition(param, param, e.target.value, val)
                    }
                    className="h-9 shrink-0 rounded-md border border-slate-200 bg-white px-1.5 text-xs"
                  >
                    {CONDITION_OPS.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                  {/* Value */}
                  <Input
                    value={String(val ?? "")}
                    onChange={(e) => {
                      let parsed: unknown = e.target.value;
                      if (
                        op === "greater_than" ||
                        op === "less_than"
                      ) {
                        const num = Number(e.target.value);
                        if (!isNaN(num)) parsed = num;
                      }
                      updateCondition(param, param, op, parsed);
                    }}
                    className="min-w-0 flex-1 basis-20 text-xs"
                    placeholder="value"
                  />
                  {/* Delete */}
                  <button
                    onClick={() => removeCondition(param)}
                    className="shrink-0 rounded p-1 text-red-400 hover:bg-red-100 hover:text-red-600"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Scenario response override */}
      <JsonEditor
        label="Response Override"
        value={scenario.response}
        onChange={(r) => onChange({ ...scenario, response: r })}
      />
    </div>
  );
}
