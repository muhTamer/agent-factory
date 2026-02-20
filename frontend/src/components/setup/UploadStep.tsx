"use client";

import { useSetupStore } from "@/store/setupStore";
import { uploadFiles, analyzeDocuments } from "@/lib/concierge-api";
import { FileDropZone } from "./FileDropZone";
import { FileList } from "./FileList";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight, Loader2, FileUp } from "lucide-react";

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

  const busy = isUploading || isAnalyzing;

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
