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

  return fetch(input, { ...init, headers });
}
