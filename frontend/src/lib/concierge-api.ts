import { authFetch } from "./auth-fetch";
import type {
  Vertical,
  AnalysisResponse,
  DeployResponse,
  WorkspaceFile,
  McpToolsConfig,
  McpToolDef,
} from "@/types/concierge";

/** Extract a human-readable error from a failed response. */
async function apiError(label: string, res: Response): Promise<Error> {
  let detail = "";
  try {
    const body = await res.text();
    const json = JSON.parse(body);
    detail = json?.detail ?? json?.message ?? body;
  } catch {
    detail = res.statusText;
  }
  return new Error(`${label} (${res.status}): ${detail}`);
}

export async function initSession(
  vertical: Vertical,
  useLlm = true,
  model = "gpt-5-mini"
) {
  const res = await authFetch(`/api/concierge/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vertical, use_llm: useLlm, model }),
  });
  if (!res.ok) throw await apiError("Init failed", res);
  return res.json();
}

export async function uploadFiles(files: File[], vertical: Vertical) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("vertical", vertical);
  const res = await authFetch(`/api/concierge/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw await apiError("Upload failed", res);
  return res.json() as Promise<{ files_saved: string[]; workspace: string }>;
}

export async function quickstartFintech(
  useLlm = true,
  model = "gpt-5-mini"
): Promise<AnalysisResponse> {
  const res = await authFetch(`/api/concierge/quickstart-fintech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_llm: useLlm, model }),
  });
  if (!res.ok) throw await apiError("Quickstart failed", res);
  return res.json();
}

export async function analyzeDocuments(
  useLlm = true,
  model = "gpt-5-mini"
): Promise<AnalysisResponse> {
  const res = await authFetch(`/api/concierge/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_llm: useLlm, model }),
  });
  if (!res.ok) throw await apiError("Analysis failed", res);
  return res.json();
}

export async function generateTemplates(): Promise<AnalysisResponse> {
  const res = await authFetch(`/api/concierge/generate-templates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw await apiError("Template generation failed", res);
  return res.json();
}

export async function deployFactory(
  mode: "dry" | "live" = "dry",
  docVisibility?: Record<string, "customer_facing" | "internal">
): Promise<DeployResponse> {
  const res = await authFetch(`/api/concierge/deploy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, doc_visibility: docVisibility ?? null }),
  });
  if (!res.ok) throw await apiError("Deploy failed", res);
  return res.json();
}

export async function startRuntime(port = 808) {
  const res = await authFetch(`/api/concierge/runtime/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  if (!res.ok) throw await apiError("Start runtime failed", res);
  return res.json();
}

export async function stopRuntime(port = 808) {
  const res = await authFetch(`/api/concierge/runtime/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  if (!res.ok) throw await apiError("Stop runtime failed", res);
  return res.json();
}

export async function getRuntimeHealth() {
  const res = await authFetch(`/api/concierge/runtime/health`, {
    cache: "no-store",
  });
  if (!res.ok) throw await apiError("Runtime health failed", res);
  return res.json();
}

export async function listWorkspaceFiles(): Promise<WorkspaceFile[]> {
  const res = await authFetch(`/api/concierge/workspace/files`);
  if (!res.ok) throw await apiError("Workspace listing failed", res);
  return res.json();
}

export async function deleteWorkspaceFile(filename: string) {
  const res = await authFetch(
    `/api/concierge/workspace/files/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw await apiError("Delete failed", res);
  return res.json();
}

// ── MCP Tool Configuration ──────────────────────────────────────────

export async function getMcpToolsConfig(): Promise<McpToolsConfig> {
  const res = await authFetch(`/api/concierge/mcp-tools`);
  if (!res.ok) throw await apiError("MCP tools fetch failed", res);
  return res.json();
}

export async function saveMcpToolsConfig(
  tools: McpToolDef[],
  serverName = "demo-server"
) {
  const res = await authFetch(`/api/concierge/mcp-tools`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tools, server_name: serverName }),
  });
  if (!res.ok) throw await apiError("MCP tools save failed", res);
  return res.json();
}

export async function saveSingleMcpTool(toolName: string, tool: McpToolDef) {
  const res = await authFetch(
    `/api/concierge/mcp-tools/${encodeURIComponent(toolName)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool }),
    }
  );
  if (!res.ok) throw await apiError("MCP tool save failed", res);
  return res.json();
}

export async function deleteMcpTool(toolName: string) {
  const res = await authFetch(
    `/api/concierge/mcp-tools/${encodeURIComponent(toolName)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw await apiError("MCP tool delete failed", res);
  return res.json();
}
