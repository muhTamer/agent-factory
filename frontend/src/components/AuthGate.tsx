"use client";

import { useAuth } from "@/hooks/useAuth";
import { Loader2 } from "lucide-react";

/**
 * Wraps authenticated pages. Fetches the backend JWT token
 * on mount so all subsequent API calls are authenticated.
 */
export function AuthGate({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading, backendToken } = useAuth();

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
