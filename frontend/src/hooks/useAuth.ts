"use client";

import { useSession, signOut } from "next-auth/react";
import { useCallback, useEffect } from "react";
import { useAuthStore } from "@/store/authStore";

/** Re-fetch the backend token if older than 20 hours (token has 24h TTL). */
const TOKEN_REFRESH_MS = 20 * 60 * 60 * 1000;

async function fetchBackendToken(): Promise<string> {
  const res = await fetch("/api/token");
  if (!res.ok) throw new Error(`Token fetch failed: ${res.status}`);
  const { token } = await res.json();
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
    setBackendToken,
    clearBackendToken,
    setFetchingToken,
  } = useAuthStore();

  const refreshToken = useCallback(async () => {
    if (isFetchingToken) return;
    setFetchingToken(true);
    try {
      const token = await fetchBackendToken();
      setBackendToken(token);
    } catch {
      clearBackendToken();
    } finally {
      setFetchingToken(false);
    }
  }, [isFetchingToken, setBackendToken, clearBackendToken, setFetchingToken]);

  // Fetch token when user signs in
  useEffect(() => {
    if (status === "authenticated" && !backendToken && !isFetchingToken) {
      refreshToken();
    }
  }, [status, backendToken, isFetchingToken, refreshToken]);

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

  return {
    session,
    status,
    isAuthenticated: status === "authenticated",
    isLoading: status === "loading",
    backendToken,
    logout,
    refreshToken,
  };
}

/**
 * Get the current backend token synchronously (for use in API helpers).
 * Returns null if not authenticated.
 */
export function getBackendToken(): string | null {
  return useAuthStore.getState().backendToken;
}
