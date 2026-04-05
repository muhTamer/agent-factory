import { API_BASE } from "./constants";
import { authFetch } from "./auth-fetch";
import type { ChatRequest, ChatResponse, HealthResponse } from "@/types/api";

export async function getHealth(): Promise<HealthResponse> {
  const res = await authFetch(`${API_BASE}/health`, { cache: "no-store" });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Health check failed (${res.status}): ${body}`);
  }
  return res.json();
}

export async function postChat(body: ChatRequest): Promise<ChatResponse> {
  const res = await authFetch(`${API_BASE}/chat`, {
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
  return res.json();
}
