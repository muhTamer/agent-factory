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

  if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
    providers.push(
      Google({
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      })
    );
  }

  if (process.env.MICROSOFT_CLIENT_ID && process.env.MICROSOFT_CLIENT_SECRET) {
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

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: buildProviders(),
  pages: {
    signIn: "/login",
  },
  session: {
    strategy: "jwt",
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
