import { useAuthStore } from "@/store/authStore";

/**
 * Wrapper around fetch that automatically adds the Authorization header
 * with the backend JWT token when available.
 * Includes telemetry: logs every API call with timing, status, and errors.
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

  const url =
    typeof input === "string"
      ? input
      : input instanceof URL
        ? input.href
        : input.url;
  const method = init?.method ?? "GET";
  const t0 = performance.now();

  console.info(`[API] ${method} ${url} token=${token ? "yes" : "NO"}`);

  try {
    const res = await fetch(input, { ...init, headers });
    const elapsed = Math.round(performance.now() - t0);
    const level = res.ok ? "info" : "warn";
    console[level](
      `[API] ${method} ${url} -> ${res.status} (${elapsed}ms)`
    );
    return res;
  } catch (err) {
    const elapsed = Math.round(performance.now() - t0);
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[API] ${method} ${url} -> NETWORK ERROR (${elapsed}ms): ${msg}`);
    // Re-throw with the target URL so the user can diagnose.
    throw new Error(`Network error calling ${url}: ${msg}`);
  }
}
