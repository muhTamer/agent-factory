import { CONCIERGE_API } from "./constants";
import { authFetch } from "./auth-fetch";
import type {
  Vertical,
  AnalysisResponse,
  DeployResponse,
  WorkspaceFile,
  McpToolsConfig,
  McpToolDef,
} from "@/types/concierge";

export async function initSession(
  vertical: Vertical,
  useLlm = true,
  model = "gpt-5-mini"
) {
  const res = await authFetch(`${CONCIERGE_API}/concierge/init`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vertical, use_llm: useLlm, model }),
  });
  if (!res.ok) throw new Error(`Init failed: ${res.status}`);
  return res.json();
}

export async function uploadFiles(files: File[], vertical: Vertical) {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("vertical", vertical);
  const res = await authFetch(`${CONCIERGE_API}/concierge/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json() as Promise<{ files_saved: string[]; workspace: string }>;
}

export async function quickstartFintech(
  useLlm = true,
  model = "gpt-5-mini"
): Promise<AnalysisResponse> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/quickstart-fintech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_llm: useLlm, model }),
  });
  if (!res.ok) throw new Error(`Quickstart failed: ${res.status}`);
  return res.json();
}

export async function analyzeDocuments(
  useLlm = true,
  model = "gpt-5-mini"
): Promise<AnalysisResponse> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_llm: useLlm, model }),
  });
  if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
  return res.json();
}

export async function generateTemplates(): Promise<AnalysisResponse> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/generate-templates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`Template generation failed: ${res.status}`);
  return res.json();
}

export async function deployFactory(
  mode: "dry" | "live" = "dry",
  docVisibility?: Record<string, "customer_facing" | "internal">
): Promise<DeployResponse> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/deploy`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, doc_visibility: docVisibility ?? null }),
  });
  if (!res.ok) throw new Error(`Deploy failed: ${res.status}`);
  return res.json();
}

export async function startRuntime(port = 808) {
  const res = await authFetch(`${CONCIERGE_API}/concierge/runtime/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  if (!res.ok) throw new Error(`Start runtime failed: ${res.status}`);
  return res.json();
}

export async function stopRuntime(port = 808) {
  const res = await authFetch(`${CONCIERGE_API}/concierge/runtime/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  if (!res.ok) throw new Error(`Stop runtime failed: ${res.status}`);
  return res.json();
}

export async function getRuntimeHealth() {
  const res = await authFetch(`${CONCIERGE_API}/concierge/runtime/health`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Runtime health failed: ${res.status}`);
  return res.json();
}

export async function listWorkspaceFiles(): Promise<WorkspaceFile[]> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/workspace/files`);
  if (!res.ok) throw new Error(`Workspace listing failed: ${res.status}`);
  return res.json();
}

export async function deleteWorkspaceFile(filename: string) {
  const res = await authFetch(
    `${CONCIERGE_API}/concierge/workspace/files/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
  return res.json();
}

// ── MCP Tool Configuration ──────────────────────────────────────────

export async function getMcpToolsConfig(): Promise<McpToolsConfig> {
  const res = await authFetch(`${CONCIERGE_API}/concierge/mcp-tools`);
  if (!res.ok) throw new Error(`MCP tools fetch failed: ${res.status}`);
  return res.json();
}

export async function saveMcpToolsConfig(
  tools: McpToolDef[],
  serverName = "demo-server"
) {
  const res = await authFetch(`${CONCIERGE_API}/concierge/mcp-tools`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tools, server_name: serverName }),
  });
  if (!res.ok) throw new Error(`MCP tools save failed: ${res.status}`);
  return res.json();
}

export async function saveSingleMcpTool(toolName: string, tool: McpToolDef) {
  const res = await authFetch(
    `${CONCIERGE_API}/concierge/mcp-tools/${encodeURIComponent(toolName)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tool }),
    }
  );
  if (!res.ok) throw new Error(`MCP tool save failed: ${res.status}`);
  return res.json();
}

export async function deleteMcpTool(toolName: string) {
  const res = await authFetch(
    `${CONCIERGE_API}/concierge/mcp-tools/${encodeURIComponent(toolName)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error(`MCP tool delete failed: ${res.status}`);
  return res.json();
}
