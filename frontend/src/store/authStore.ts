import { create } from "zustand";

interface AuthState {
  /** JWT token for backend API calls */
  backendToken: string | null;
  /** Timestamp when the token was fetched */
  tokenFetchedAt: number | null;
  /** Whether a token fetch is in progress */
  isFetchingToken: boolean;
  /** Number of consecutive token fetch failures */
  tokenRetries: number;
  /** Last token fetch error message */
  tokenError: string | null;

  setBackendToken: (token: string) => void;
  clearBackendToken: () => void;
  setFetchingToken: (v: boolean) => void;
  setTokenError: (error: string | null, retries: number) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  backendToken: null,
  tokenFetchedAt: null,
  isFetchingToken: false,
  tokenRetries: 0,
  tokenError: null,

  setBackendToken: (token) =>
    set({ backendToken: token, tokenFetchedAt: Date.now(), tokenRetries: 0, tokenError: null }),
  clearBackendToken: () =>
    set({ backendToken: null, tokenFetchedAt: null }),
  setFetchingToken: (v) => set({ isFetchingToken: v }),
  setTokenError: (error, retries) => set({ tokenError: error, tokenRetries: retries }),
}));
