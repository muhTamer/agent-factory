import { create } from "zustand";
import type {
  Vertical,
  WizardStep,
  FactoryPlan,
  DeploymentInfo,
  WorkspaceFile,
  McpToolDef,
} from "@/types/concierge";

interface SetupState {
  currentStep: WizardStep;
  setStep: (step: WizardStep) => void;

  vertical: Vertical;
  setVertical: (v: Vertical) => void;

  stagedFiles: File[];
  setStagedFiles: (files: File[]) => void;
  addStagedFiles: (files: File[]) => void;
  removeStagedFile: (name: string) => void;

  workspaceFiles: WorkspaceFile[];
  setWorkspaceFiles: (files: WorkspaceFile[]) => void;

  plan: FactoryPlan | null;
  setPlan: (plan: FactoryPlan | null) => void;

  analysisSummaryText: string;
  setAnalysisSummaryText: (text: string) => void;

  deployment: DeploymentInfo | null;
  setDeployment: (dep: DeploymentInfo | null) => void;

  deployMessage: string;
  setDeployMessage: (msg: string) => void;

  runtimeOnline: boolean;
  setRuntimeOnline: (v: boolean) => void;

  isAnalyzing: boolean;
  setAnalyzing: (v: boolean) => void;

  isDeploying: boolean;
  setDeploying: (v: boolean) => void;

  isUploading: boolean;
  setUploading: (v: boolean) => void;

  isStartingRuntime: boolean;
  setStartingRuntime: (v: boolean) => void;

  error: string | null;
  setError: (e: string | null) => void;

  isQuickstart: boolean;
  setQuickstart: (v: boolean) => void;

  docVisibility: Record<string, "customer_facing" | "internal">;
  setDocVisibility: (v: Record<string, "customer_facing" | "internal">) => void;
  initDocVisibilityDefaults: (filenames: string[]) => void;

  mcpTools: McpToolDef[];
  setMcpTools: (tools: McpToolDef[]) => void;
  isSavingTools: boolean;
  setSavingTools: (v: boolean) => void;

  reset: () => void;
}

const INITIAL: Pick<
  SetupState,
  | "currentStep"
  | "vertical"
  | "stagedFiles"
  | "workspaceFiles"
  | "plan"
  | "analysisSummaryText"
  | "deployment"
  | "deployMessage"
  | "runtimeOnline"
  | "isAnalyzing"
  | "isDeploying"
  | "isUploading"
  | "isStartingRuntime"
  | "error"
  | "isQuickstart"
  | "docVisibility"
  | "mcpTools"
  | "isSavingTools"
> = {
  currentStep: "welcome",
  vertical: "retail",
  stagedFiles: [],
  workspaceFiles: [],
  plan: null,
  analysisSummaryText: "",
  deployment: null,
  deployMessage: "",
  runtimeOnline: false,
  isAnalyzing: false,
  isDeploying: false,
  isUploading: false,
  isStartingRuntime: false,
  error: null,
  isQuickstart: false,
  docVisibility: {},
  mcpTools: [],
  isSavingTools: false,
};

export const useSetupStore = create<SetupState>((set) => ({
  ...INITIAL,

  setStep: (step) => set({ currentStep: step }),
  setVertical: (v) => set({ vertical: v }),

  setStagedFiles: (files) => set({ stagedFiles: files }),
  addStagedFiles: (files) =>
    set((s) => {
      const names = new Set(s.stagedFiles.map((f) => f.name));
      const newFiles = files.filter((f) => !names.has(f.name));
      return { stagedFiles: [...s.stagedFiles, ...newFiles] };
    }),
  removeStagedFile: (name) =>
    set((s) => ({ stagedFiles: s.stagedFiles.filter((f) => f.name !== name) })),

  setWorkspaceFiles: (files) => set({ workspaceFiles: files }),

  setPlan: (plan) => set({ plan }),
  setAnalysisSummaryText: (text) => set({ analysisSummaryText: text }),

  setDeployment: (dep) => set({ deployment: dep }),
  setDeployMessage: (msg) => set({ deployMessage: msg }),

  setRuntimeOnline: (v) => set({ runtimeOnline: v }),

  setAnalyzing: (v) => set({ isAnalyzing: v }),
  setDeploying: (v) => set({ isDeploying: v }),
  setUploading: (v) => set({ isUploading: v }),
  setStartingRuntime: (v) => set({ isStartingRuntime: v }),

  setError: (e) => set({ error: e }),
  setQuickstart: (v) => set({ isQuickstart: v }),

  setDocVisibility: (v) => set({ docVisibility: v }),
  setMcpTools: (tools) => set({ mcpTools: tools }),
  setSavingTools: (v) => set({ isSavingTools: v }),
  initDocVisibilityDefaults: (filenames) =>
    set((s) => {
      const next: Record<string, "customer_facing" | "internal"> = { ...s.docVisibility };
      for (const name of filenames) {
        if (next[name]) continue; // keep existing user choice
        const ext = name.split(".").pop()?.toLowerCase() ?? "";
        next[name] = ext === "yaml" || ext === "yml" ? "internal" : "customer_facing";
      }
      return { docVisibility: next };
    }),

  reset: () => set(INITIAL),
}));
