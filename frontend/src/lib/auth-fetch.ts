import { useAuthStore } from "@/store/authStore";

/**
 * Wrapper around fetch that automatically adds the Authorization header
 * with the backend JWT token when available.
 */
export async function authFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  const token = useAuthStore.getState().backendToken;
  const headers = new Headers(init?.headers);

  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;

  try {
    return await fetch(input, { ...init, headers });
  } catch (err) {
    // Network / CORS errors surface as TypeError with unhelpful messages
    // ("Load failed" on Safari, "Failed to fetch" on Chrome).
    // Re-throw with the target URL so the user can diagnose.
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error calling ${url}: ${msg}`);
  }
}
