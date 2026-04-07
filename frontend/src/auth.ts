import NextAuth from "next-auth";
import Google from "next-auth/providers/google";
import Facebook from "next-auth/providers/facebook";
import Apple from "next-auth/providers/apple";
import MicrosoftEntraId from "next-auth/providers/microsoft-entra-id";
import { createHash } from "crypto";
import type { Provider } from "next-auth/providers";

function tenantIdFromEmail(email: string): string {
  return createHash("sha256")
    .update(email.toLowerCase().trim())
    .digest("hex")
    .slice(0, 16);
}

/** Only include providers whose credentials are actually set. */
function buildProviders(): Provider[] {
  const providers: Provider[] = [];

  console.log("[AUTH] Building providers...");
  console.log("[AUTH] AUTH_SECRET set:", !!process.env.AUTH_SECRET);
  console.log("[AUTH] GOOGLE_CLIENT_ID set:", !!process.env.GOOGLE_CLIENT_ID);
  console.log("[AUTH] MICROSOFT_CLIENT_ID set:", !!process.env.MICROSOFT_CLIENT_ID);

  if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
    providers.push(
      Google({
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      })
    );
  }

  if (process.env.MICROSOFT_CLIENT_ID && process.env.MICROSOFT_CLIENT_SECRET) {
    console.log("[AUTH] Configuring Microsoft Entra ID provider");
    providers.push(
      MicrosoftEntraId({
        clientId: process.env.MICROSOFT_CLIENT_ID,
        clientSecret: process.env.MICROSOFT_CLIENT_SECRET,
        authorization: {
          params: {
            scope: "openid profile email User.Read",
          },
        },
      })
    );
  }

  if (
    process.env.FACEBOOK_CLIENT_ID &&
    process.env.FACEBOOK_CLIENT_SECRET &&
    process.env.FACEBOOK_CLIENT_ID !== "placeholder"
  ) {
    providers.push(
      Facebook({
        clientId: process.env.FACEBOOK_CLIENT_ID,
        clientSecret: process.env.FACEBOOK_CLIENT_SECRET,
      })
    );
  }

  if (
    process.env.APPLE_CLIENT_ID &&
    process.env.APPLE_CLIENT_SECRET &&
    process.env.APPLE_CLIENT_ID !== "placeholder"
  ) {
    providers.push(
      Apple({
        clientId: process.env.APPLE_CLIENT_ID,
        clientSecret: process.env.APPLE_CLIENT_SECRET,
      })
    );
  }

  return providers;
}

// Store last auth error for the debug endpoint
export let lastAuthError: { time: string; error: string; details: unknown } | null = null;

export const { handlers, signIn, signOut, auth } = NextAuth({
  debug: true,
  providers: buildProviders(),
  pages: {
    signIn: "/login",
    error: "/login",
  },
  session: {
    strategy: "jwt",
  },
  logger: {
    error(error) {
      const details = error instanceof Error
        ? { name: error.name, message: error.message, cause: String((error as unknown as Record<string, unknown>).cause ?? "none"), stack: error.stack?.split("\n").slice(0, 5).join("\n") }
        : error;
      lastAuthError = {
        time: new Date().toISOString(),
        error: error instanceof Error ? error.message : String(error),
        details,
      };
      console.error("[AUTH ERROR]", JSON.stringify(lastAuthError));
    },
    warn(code) {
      console.warn("[AUTH WARN]", code);
    },
    debug(message, metadata) {
      console.debug("[AUTH DEBUG]", message, metadata);
    },
  },
  callbacks: {
    async jwt({ token, user, account }) {
      if (user) {
        token.provider = account?.provider ?? "";
        token.tenant_id = tenantIdFromEmail(user.email ?? "");
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        (session.user as unknown as Record<string, unknown>).tenant_id =
          token.tenant_id as string;
        (session.user as unknown as Record<string, unknown>).provider =
          token.provider as string;
      }
      return session;
    },
  },
});
