"use client";

import { useSession, signOut } from "next-auth/react";
import { useCallback, useEffect } from "react";
import { useAuthStore } from "@/store/authStore";

/** Re-fetch the backend token if older than 20 hours (token has 24h TTL). */
const TOKEN_REFRESH_MS = 20 * 60 * 60 * 1000;
const MAX_TOKEN_RETRIES = 3;

async function fetchBackendToken(): Promise<string> {
  console.info("[AUTH] Fetching backend JWT from /api/token...");
  const res = await fetch("/api/token");
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    console.error(`[AUTH] Token fetch failed: ${res.status} ${body}`);
    throw new Error(`Token fetch failed (${res.status}): ${body}`);
  }
  const { token } = await res.json();
  console.info("[AUTH] Backend JWT acquired (length=%d)", token?.length ?? 0);
  return token;
}

/**
 * Hook that manages the authenticated user session.
 * - Automatically fetches a backend JWT when the user is signed in.
 * - Refreshes the token before it expires.
 * - Provides logout that clears both NextAuth session and backend token.
 */
export function useAuth() {
  const { data: session, status } = useSession();
  const {
    backendToken,
    tokenFetchedAt,
    isFetchingToken,
    tokenRetries,
    tokenError,
    setBackendToken,
    clearBackendToken,
    setFetchingToken,
    setTokenError,
  } = useAuthStore();

  const refreshToken = useCallback(async () => {
    if (isFetchingToken) return;
    const currentRetries = useAuthStore.getState().tokenRetries;
    if (currentRetries >= MAX_TOKEN_RETRIES) return;

    setFetchingToken(true);
    try {
      const token = await fetchBackendToken();
      setBackendToken(token);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("[AUTH] Token refresh failed:", msg);
      const nextRetry = currentRetries + 1;
      setTokenError(msg, nextRetry);
      clearBackendToken();
    } finally {
      setFetchingToken(false);
    }
  }, [isFetchingToken, setBackendToken, clearBackendToken, setFetchingToken, setTokenError]);

  // Fetch token when user signs in
  useEffect(() => {
    if (
      status === "authenticated" &&
      !backendToken &&
      !isFetchingToken &&
      tokenRetries < MAX_TOKEN_RETRIES
    ) {
      refreshToken();
    }
  }, [status, backendToken, isFetchingToken, tokenRetries, refreshToken]);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!tokenFetchedAt || !backendToken) return;
    const age = Date.now() - tokenFetchedAt;
    const delay = Math.max(TOKEN_REFRESH_MS - age, 0);
    const timer = setTimeout(refreshToken, delay);
    return () => clearTimeout(timer);
  }, [tokenFetchedAt, backendToken, refreshToken]);

  const logout = useCallback(async () => {
    clearBackendToken();
    await signOut({ callbackUrl: "/login" });
  }, [clearBackendToken]);

  const retryToken = useCallback(() => {
    setTokenError(null, 0);
  }, [setTokenError]);

  return {
    session,
    status,
    isAuthenticated: status === "authenticated",
    isLoading: status === "loading",
    backendToken,
    tokenError,
    tokenRetries,
    logout,
    refreshToken,
    retryToken,
  };
}

/**
 * Get the current backend token synchronously (for use in API helpers).
 * Returns null if not authenticated.
 */
export function getBackendToken(): string | null {
  return useAuthStore.getState().backendToken;
}
