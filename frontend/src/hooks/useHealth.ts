"use client";

import { useEffect, useRef } from "react";
import { getHealth } from "@/lib/api";
import { useChatStore } from "@/store/chatStore";

export function useHealth(intervalMs = 10_000) {
  const setAgents = useChatStore((s) => s.setAgents);
  const setConnected = useChatStore((s) => s.setBackendConnected);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function check() {
      try {
        const data = await getHealth();
        setAgents(data.agents || {});
        setConnected(true);
      } catch {
        setConnected(false);
      }
    }
    check();
    timerRef.current = setInterval(check, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [intervalMs, setAgents, setConnected]);
}
