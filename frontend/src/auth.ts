import type { NextRequest } from "next/server";

const IS_DEV_NO_AUTH = !process.env.AUTH_SECRET;

// ── Dev-mode stub (no AUTH_SECRET configured) ──────────────────────────
// Returns a fake session so the app works without OAuth providers.

const DEV_USER = {
  email: "dev@localhost",
  name: "Dev User",
  image: null,
  tenant_id: "dev-tenant",
  provider: "dev",
};

const DEV_SESSION = { user: DEV_USER, expires: "9999-12-31T23:59:59.999Z" };

function devHandlers() {
  const handler = async () =>
    new Response(JSON.stringify(DEV_SESSION), {
      headers: { "content-type": "application/json" },
    });
  return { GET: handler, POST: handler };
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function devAuth(_req?: NextRequest) {
  return DEV_SESSION;
}

// ── Production auth (AUTH_SECRET is set) ───────────────────────────────

/** Edge-compatible SHA-256 hash using Web Crypto API. */
async function tenantIdFromEmail(email: string): Promise<string> {
  const data = new TextEncoder().encode(email.toLowerCase().trim());
  const buf = await globalThis.crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
}

async function buildProdAuth() {
  const NextAuth = (await import("next-auth")).default;
  const Google = (await import("next-auth/providers/google")).default;
  const Facebook = (await import("next-auth/providers/facebook")).default;
  const Apple = (await import("next-auth/providers/apple")).default;
  const MicrosoftEntraId = (
    await import("next-auth/providers/microsoft-entra-id")
  ).default;

  type Provider = import("next-auth/providers").Provider;

  function buildProviders(): Provider[] {
    const providers: Provider[] = [];

    if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
      providers.push(
        Google({
          clientId: process.env.GOOGLE_CLIENT_ID,
          clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        })
      );
    }

    if (
      process.env.MICROSOFT_CLIENT_ID &&
      process.env.MICROSOFT_CLIENT_SECRET
    ) {
      providers.push(
        MicrosoftEntraId({
          clientId: process.env.MICROSOFT_CLIENT_ID,
          clientSecret: process.env.MICROSOFT_CLIENT_SECRET,
          issuer: `https://login.microsoftonline.com/consumers/v2.0`,
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

  return NextAuth({
    providers: buildProviders(),
    pages: { signIn: "/login" },
    session: { strategy: "jwt" },
    callbacks: {
      async jwt({ token, user, account }) {
        if (user) {
          token.provider = account?.provider ?? "";
          token.tenant_id = await tenantIdFromEmail(user.email ?? "");
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
}

// ── Exports ────────────────────────────────────────────────────────────

let _prod: Awaited<ReturnType<typeof buildProdAuth>> | null = null;
async function getProd() {
  if (!_prod) _prod = await buildProdAuth();
  return _prod;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export const handlers: { GET: any; POST: any } = IS_DEV_NO_AUTH
  ? devHandlers()
  : {
      GET: async (req: any) =>
        (await getProd()).handlers.GET(req),
      POST: async (req: any) =>
        (await getProd()).handlers.POST(req),
    };

export const auth: any = IS_DEV_NO_AUTH
  ? devAuth
  : async (req?: any) => (await getProd()).auth(req);

export const signIn: any = IS_DEV_NO_AUTH
  ? async () => {}
  : async (provider?: any, options?: any) =>
      (await getProd()).signIn(provider, options);

export const signOut: any = IS_DEV_NO_AUTH
  ? async () => {}
  : async (options?: any) => (await getProd()).signOut(options);
/* eslint-enable @typescript-eslint/no-explicit-any */
