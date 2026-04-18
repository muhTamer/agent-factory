"use client";

import { useEffect, useRef } from "react";
import { getHealth } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";

export function useHealth() {
  const setAgents = useChatStore((s) => s.setAgents);
  const setConnected = useChatStore((s) => s.setBackendConnected);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function check() {
      try {
        const data = await getHealth();
        const loaded =
          data.status === "ok" && Object.keys(data.agents || {}).length > 0;
        setAgents(data.agents || {});
        setConnected(loaded);
      } catch {
        setConnected(false);
      }
    }

    async function poll() {
      await check();
      // Poll fast (2s) until agents are loaded, then slow (10s)
      const connected = useChatStore.getState().backendConnected;
      const interval = connected ? 10_000 : 2_000;
      timerRef.current = setTimeout(poll, interval);
    }

    poll();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [setAgents, setConnected]);
}
