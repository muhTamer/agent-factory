export type Vertical = "retail" | "fintech" | "telco" | "general_service";

export type WizardStep =
  | "welcome"
  | "upload"
  | "analysis"
  | "deploy"
  | "runtime";

export interface PlanAgent {
  id: string;
  display_name: string;
  icon: string;
  status: "ready" | "partial" | "missing_docs";
  confidence: number;
  docs_detected: string[];
  docs_missing: string[];
  why: string;
  summary: string;
  inputs_typed: {
    knowledge_base: string[];
    policy: string[];
    procedure: string[];
    tool_spec: string[];
  };
}

export interface FactoryPlan {
  vertical: string;
  summary: string;
  agents: PlanAgent[];
}

export interface AnalysisResponse {
  type: string;
  text: string;
  plan: FactoryPlan;
}

export interface DeploymentInfo {
  vertical: string;
  mode: string;
  agents: string[];
  spec_path: string;
  generated_agents: Array<{ id: string; path: string }>;
  generation_errors: Array<{ id: string; error: string }>;
  uvicorn_command: string;
  runtime: {
    base_url: string;
    health: string;
    chat: string;
  };
}

export interface DeployResponse {
  type: string;
  text: string;
  deployment_request: DeploymentInfo;
}

export interface WorkspaceFile {
  name: string;
  size: number;
  extension: string;
}
