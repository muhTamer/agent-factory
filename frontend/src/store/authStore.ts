import { create } from "zustand";

interface AuthState {
  /** JWT token for backend API calls */
  backendToken: string | null;
  /** Timestamp when the token was fetched */
  tokenFetchedAt: number | null;
  /** Whether a token fetch is in progress */
  isFetchingToken: boolean;

  setBackendToken: (token: string) => void;
  clearBackendToken: () => void;
  setFetchingToken: (v: boolean) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  backendToken: null,
  tokenFetchedAt: null,
  isFetchingToken: false,

  setBackendToken: (token) =>
    set({ backendToken: token, tokenFetchedAt: Date.now() }),
  clearBackendToken: () =>
    set({ backendToken: null, tokenFetchedAt: null }),
  setFetchingToken: (v) => set({ isFetchingToken: v }),
}));
