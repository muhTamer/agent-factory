"use client";

import { useEffect } from "react";
import { useSetupStore } from "@/store/setupStore";
import { uploadFiles, analyzeDocuments } from "@/lib/concierge-api";
import { FileDropZone } from "./FileDropZone";
import { FileList } from "./FileList";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight, Loader2, FileUp, Eye, EyeOff, FileText } from "lucide-react";

export function UploadStep() {
  const vertical = useSetupStore((s) => s.vertical);
  const stagedFiles = useSetupStore((s) => s.stagedFiles);
  const addStagedFiles = useSetupStore((s) => s.addStagedFiles);
  const removeStagedFile = useSetupStore((s) => s.removeStagedFile);
  const isUploading = useSetupStore((s) => s.isUploading);
  const setUploading = useSetupStore((s) => s.setUploading);
  const isAnalyzing = useSetupStore((s) => s.isAnalyzing);
  const setAnalyzing = useSetupStore((s) => s.setAnalyzing);
  const setPlan = useSetupStore((s) => s.setPlan);
  const setAnalysisSummaryText = useSetupStore((s) => s.setAnalysisSummaryText);
  const setStep = useSetupStore((s) => s.setStep);
  const setError = useSetupStore((s) => s.setError);
  const docVisibility = useSetupStore((s) => s.docVisibility);
  const setDocVisibility = useSetupStore((s) => s.setDocVisibility);
  const initDocVisibilityDefaults = useSetupStore((s) => s.initDocVisibilityDefaults);

  const busy = isUploading || isAnalyzing;

  // Initialise visibility defaults whenever the staged file list changes
  useEffect(() => {
    if (stagedFiles.length > 0) {
      initDocVisibilityDefaults(stagedFiles.map((f) => f.name));
    }
  }, [stagedFiles]);

  async function handleAnalyze() {
    if (!stagedFiles.length) return;
    setError(null);

    try {
      // 1. Upload files
      setUploading(true);
      await uploadFiles(stagedFiles, vertical);
      setUploading(false);

      // 2. Analyze
      setAnalyzing(true);
      const res = await analyzeDocuments();
      setPlan(res.plan);
      setAnalysisSummaryText(res.text);
      setStep("analysis");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload/analysis failed");
    } finally {
      setUploading(false);
      setAnalyzing(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold text-slate-900">Upload Documents</h2>
        <p className="mt-1 text-sm text-slate-500">
          Upload FAQs, policies, or SOPs that will power your agents
        </p>
      </div>

      <FileDropZone onFiles={addStagedFiles} disabled={busy} />

      <FileList files={stagedFiles} onRemove={removeStagedFile} />

      {/* Visibility tagging — shown once at least one file is staged */}
      {stagedFiles.length > 0 && (
        <DocVisibilityPanel
          files={stagedFiles.map((f) => f.name)}
          visibility={docVisibility}
          onChange={(filename, vis) =>
            setDocVisibility({ ...docVisibility, [filename]: vis })
          }
        />
      )}

      {isUploading && (
        <div className="flex items-center gap-2 text-sm text-blue-600">
          <Loader2 size={16} className="animate-spin" />
          Uploading files...
        </div>
      )}
      {isAnalyzing && (
        <div className="flex items-center gap-2 text-sm text-blue-600">
          <Loader2 size={16} className="animate-spin" />
          Analyzing documents — this may take a moment...
        </div>
      )}

      <div className="flex items-center justify-between">
        <Button
          variant="outline"
          onClick={() => setStep("welcome")}
          disabled={busy}
        >
          <ArrowLeft size={16} />
          Back
        </Button>
        <Button
          onClick={handleAnalyze}
          disabled={!stagedFiles.length || busy}
        >
          {busy ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <FileUp size={16} />
          )}
          Upload & Analyze
        </Button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DocVisibilityPanel
// ---------------------------------------------------------------------------
interface DocVisibilityPanelProps {
  files: string[];
  visibility: Record<string, "customer_facing" | "internal">;
  onChange: (filename: string, vis: "customer_facing" | "internal") => void;
}

function DocVisibilityPanel({ files, visibility, onChange }: DocVisibilityPanelProps) {
  const internalCount = files.filter((f) => visibility[f] === "internal").length;

  return (
    <details className="rounded-lg border border-slate-200 bg-slate-50" open>
      <summary className="flex cursor-pointer list-none items-center gap-2 px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-100 rounded-lg">
        <FileText size={15} className="shrink-0 text-slate-400" />
        <span className="flex-1">
          Document visibility
          <span className="ml-2 text-xs font-normal text-slate-400">
            {files.length - internalCount} customer-facing · {internalCount} internal
          </span>
        </span>
      </summary>

      <div className="px-4 pb-4 pt-2 space-y-2">
        <p className="text-xs text-slate-500 mb-3">
          Choose which documents the customer-facing agent can answer questions from.
          Internal documents are used for workflow rules only.
        </p>
        {files.map((filename) => {
          const vis = visibility[filename] ?? "customer_facing";
          return (
            <div
              key={filename}
              className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 rounded-md border border-slate-200 bg-white px-3 py-2"
            >
              <span className="text-sm text-slate-700 truncate" title={filename}>
                {filename}
              </span>
              <div className="flex gap-1 shrink-0">
                <button
                  onClick={() => onChange(filename, "customer_facing")}
                  className={`flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    vis === "customer_facing"
                      ? "bg-green-100 text-green-700 ring-1 ring-green-300"
                      : "text-slate-400 hover:bg-slate-100"
                  }`}
                >
                  <Eye size={11} />
                  <span className="hidden xs:inline">Customer-facing</span>
                  <span className="xs:hidden">Public</span>
                </button>
                <button
                  onClick={() => onChange(filename, "internal")}
                  className={`flex items-center gap-1 rounded-full px-2.5 py-1.5 text-xs font-medium transition-colors ${
                    vis === "internal"
                      ? "bg-amber-100 text-amber-700 ring-1 ring-amber-300"
                      : "text-slate-400 hover:bg-slate-100"
                  }`}
                >
                  <EyeOff size={11} />
                  Internal
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </details>
  );
}
