"use client";

import { useAuth } from "@/hooks/useAuth";
import { Loader2, AlertCircle, RefreshCw } from "lucide-react";

/**
 * Wraps authenticated pages. Fetches the backend JWT token
 * on mount so all subsequent API calls are authenticated.
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, backendToken, tokenError, retryToken, logout } = useAuth();

  // Still checking session
  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 size={24} className="animate-spin text-slate-400" />
      </div>
    );
  }

  // Not authenticated — middleware should redirect, but just in case
  if (!isAuthenticated) {
    return null;
  }

  // Token fetch failed after retries — show error with retry
  if (!backendToken && tokenError) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
        <div className="w-full max-w-md space-y-4 rounded-xl border border-red-200 bg-red-50 p-6">
          <div className="flex items-start gap-3">
            <AlertCircle size={20} className="mt-0.5 shrink-0 text-red-500" />
            <div className="space-y-1">
              <p className="text-sm font-semibold text-red-800">
                Failed to initialize workspace
              </p>
              <p className="text-xs text-red-600 break-all font-mono">
                {tokenError}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={retryToken}
              className="flex items-center gap-1.5 rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-red-700"
            >
              <RefreshCw size={12} />
              Retry
            </button>
            <button
              onClick={logout}
              className="rounded-lg border border-red-200 px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100"
            >
              Sign out
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Waiting for backend token to be fetched
  if (!backendToken) {
    return (
      <div className="flex min-h-screen items-center justify-center gap-2">
        <Loader2 size={20} className="animate-spin text-blue-500" />
        <span className="text-sm text-slate-500">Preparing your workspace...</span>
      </div>
    );
  }

  return <>{children}</>;
}
