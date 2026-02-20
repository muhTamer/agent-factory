"use client";

import { useEffect, useRef, useState } from "react";
import { getRuntimeHealth } from "@/lib/concierge-api";

export function useRuntimePoller(enabled: boolean, intervalMs = 2000) {
  const [online, setOnline] = useState(false);
  const [agentCount, setAgentCount] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      setOnline(false);
      return;
    }

    async function check() {
      try {
        const data = await getRuntimeHealth();
        if (data.status === "ok") {
          setOnline(true);
          setAgentCount(Object.keys(data.agents || {}).length);
        } else {
          setOnline(false);
        }
      } catch {
        setOnline(false);
      }
    }

    check();
    timerRef.current = setInterval(check, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled, intervalMs]);

  return { online, agentCount };
}
