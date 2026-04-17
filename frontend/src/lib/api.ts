import { authFetch } from "./auth-fetch";
import type { ChatRequest, ChatResponse, HealthResponse } from "@/types/api";

export async function getHealth(): Promise<HealthResponse> {
  const res = await authFetch(`/api/runtime/health`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Health check failed (${res.status}): ${body}`);
  }
  return res.json();
}

export async function postChat(body: ChatRequest): Promise<ChatResponse> {
  const res = await authFetch(`/api/runtime/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      const json = JSON.parse(text);
      detail = json?.detail ?? json?.message ?? text;
    } catch {
      // keep raw text
    }
    throw new Error(`Chat request failed (${res.status}): ${detail}`);
  }

  const data = await res.json();

  // Async job pattern: poll until done
  if (data.job_id && data.status === "processing") {
    return pollChatJob(data.job_id);
  }

  // Synchronous fallback
  return data;
}

async function pollChatJob(jobId: string): Promise<ChatResponse> {
  while (true) {
    await new Promise((r) => setTimeout(r, 1000));
    const res = await authFetch(`/api/runtime/chat/job/${jobId}`);
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Chat poll failed (${res.status}): ${text}`);
    }
    const data = await res.json();
    if (data.status === "done") return data.result;
    if (data.status === "error") throw new Error(data.error);
    // still processing — loop
  }
}
