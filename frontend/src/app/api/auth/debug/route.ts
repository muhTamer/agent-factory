import { NextResponse } from "next/server";

/**
 * Debug endpoint — shows which auth env vars are set (not their values).
 * Remove this in production once auth is working.
 */
export async function GET() {
  return NextResponse.json({
    AUTH_SECRET: !!process.env.AUTH_SECRET,
    AUTH_SECRET_LENGTH: process.env.AUTH_SECRET?.length ?? 0,
    AUTH_TRUST_HOST: process.env.AUTH_TRUST_HOST ?? "not set",
    NEXTAUTH_URL: process.env.NEXTAUTH_URL ?? "not set",
    GOOGLE_CLIENT_ID: !!process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: !!process.env.GOOGLE_CLIENT_SECRET,
    MICROSOFT_CLIENT_ID: !!process.env.MICROSOFT_CLIENT_ID,
    MICROSOFT_CLIENT_SECRET: !!process.env.MICROSOFT_CLIENT_SECRET,
    FACEBOOK_CLIENT_ID: process.env.FACEBOOK_CLIENT_ID ?? "not set",
    APPLE_CLIENT_ID: process.env.APPLE_CLIENT_ID ?? "not set",
    NODE_ENV: process.env.NODE_ENV,
  });
}
