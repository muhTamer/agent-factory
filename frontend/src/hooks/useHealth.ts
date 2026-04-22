"use client";

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { getHealth } from "@/lib/api";
import { startRuntime } from "@/lib/concierge-api";
import { useChatStore } from "@/store/chatStore";

export function useHealth() {
  const router = useRouter();
  const setAgents = useChatStore((s) => s.setAgents);
  const setConnected = useChatStore((s) => s.setBackendConnected);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reloadTriggered = useRef(false);
  const waitingCount = useRef(0);

  useEffect(() => {
    async function check() {
      try {
        const data = await getHealth();
        const loaded =
          data.status === "ok" && Object.keys(data.agents || {}).length > 0;
        setAgents(data.agents || {});
        setConnected(loaded);

        if (loaded) {
          waitingCount.current = 0;
          reloadTriggered.current = false;
        } else {
          waitingCount.current++;
          if (waitingCount.current >= 3 && !reloadTriggered.current) {
            reloadTriggered.current = true;
            try { await startRuntime(); } catch { /* ignore */ }
          }
          // After ~30s of failing, redirect to setup wizard for full restore
          if (waitingCount.current >= 8) {
            router.push("/");
          }
        }
      } catch {
        setConnected(false);
        waitingCount.current++;
        if (waitingCount.current >= 3 && !reloadTriggered.current) {
          reloadTriggered.current = true;
          try { await startRuntime(); } catch { /* ignore */ }
        }
        if (waitingCount.current >= 15) {
          router.push("/");
        }
      }
    }

    async function poll() {
      await check();
      const connected = useChatStore.getState().backendConnected;
      const interval = connected ? 10_000 : 2_000;
      timerRef.current = setTimeout(poll, interval);
    }

    poll();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [setAgents, setConnected, router]);
}
