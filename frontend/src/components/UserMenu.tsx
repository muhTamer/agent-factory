"use client";

import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/hooks/useAuth";
import { useSetupStore } from "@/store/setupStore";
import { resetSession } from "@/lib/concierge-api";
import { LogOut, RotateCcw, User, Loader2 } from "lucide-react";

export function UserMenu() {
  const { session, logout } = useAuth();
  const reset = useSetupStore((s) => s.reset);
  const [open, setOpen] = useState(false);
  const [resetting, setResetting] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on click outside
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const user = session?.user;
  if (!user) return null;

  const initials = (user.name ?? user.email ?? "?")
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  async function handleReset() {
    if (!confirm("This will delete all your agents, chats, and uploaded data. Start over?")) return;
    setResetting(true);
    try {
      await resetSession();
      reset();
      setOpen(false);
    } catch (err) {
      console.error("[Reset] failed:", err);
      alert("Reset failed. Please try again.");
    } finally {
      setResetting(false);
    }
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-100 text-sm font-medium text-blue-700 transition-colors hover:bg-blue-200"
        title={user.name ?? user.email ?? "Account"}
      >
        {initials}
      </button>

      {open && (
        <div className="absolute right-0 top-10 z-50 w-64 rounded-lg border bg-white py-2 shadow-lg">
          <div className="border-b px-4 py-2">
            <p className="text-sm font-medium text-slate-800 truncate">
              {user.name}
            </p>
            <p className="text-xs text-slate-400 truncate">{user.email}</p>
          </div>
          <button
            onClick={handleReset}
            disabled={resetting}
            className="flex w-full items-center gap-2 px-4 py-2 text-sm text-slate-600 transition-colors hover:bg-slate-50 disabled:opacity-50"
          >
            {resetting ? <Loader2 size={14} className="animate-spin" /> : <RotateCcw size={14} />}
            Start Over
          </button>
          <button
            onClick={() => {
              setOpen(false);
              logout();
            }}
            className="flex w-full items-center gap-2 px-4 py-2 text-sm text-slate-600 transition-colors hover:bg-slate-50"
          >
            <LogOut size={14} />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
