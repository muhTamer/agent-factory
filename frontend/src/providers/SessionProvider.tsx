"use client";

import { SessionProvider as NextAuthSessionProvider } from "next-auth/react";

export default function SessionProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  // When no auth providers are configured (local dev), the NextAuth session
  // endpoint returns the dev stub — SessionProvider still works fine.
  return <NextAuthSessionProvider>{children}</NextAuthSessionProvider>;
}
