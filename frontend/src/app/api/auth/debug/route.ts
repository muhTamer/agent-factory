import { NextResponse } from "next/server";

/**
 * Debug endpoint — shows which auth env vars are set (not their values).
 * Remove this in production once auth is working.
 */
export async function GET() {
  // Test Microsoft OIDC discovery
  let msDiscovery = "not tested";
  try {
    const res = await fetch(
      "https://login.microsoftonline.com/common/v2.0/.well-known/openid-configuration",
      { signal: AbortSignal.timeout(5000) }
    );
    if (res.ok) {
      const data = await res.json();
      msDiscovery = `ok (issuer=${data.issuer})`;
    } else {
      msDiscovery = `http ${res.status}`;
    }
  } catch (e) {
    msDiscovery = `error: ${e instanceof Error ? e.message : String(e)}`;
  }

  return NextResponse.json({
    AUTH_SECRET: !!process.env.AUTH_SECRET,
    AUTH_SECRET_LENGTH: process.env.AUTH_SECRET?.length ?? 0,
    AUTH_TRUST_HOST: process.env.AUTH_TRUST_HOST ?? "not set",
    NEXTAUTH_URL: process.env.NEXTAUTH_URL ?? "not set",
    GOOGLE_CLIENT_ID: !!process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: !!process.env.GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID: !!process.env.MICROSOFT_CLIENT_ID,
    MICROSOFT_CLIENT_SECRET: !!process.env.MICROSOFT_CLIENT_SECRET,
    MICROSOFT_CLIENT_SECRET_LENGTH:
      process.env.MICROSOFT_CLIENT_SECRET?.length ?? 0,
    MICROSOFT_CLIENT_ID_PREFIX:
      process.env.MICROSOFT_CLIENT_ID?.slice(0, 8) ?? "not set",
    FACEBOOK_CLIENT_ID: process.env.FACEBOOK_CLIENT_ID ?? "not set",
    APPLE_CLIENT_ID: process.env.APPLE_CLIENT_ID ?? "not set",
    NODE_ENV: process.env.NODE_ENV,
    ms_oidc_discovery: msDiscovery,
  });
}
