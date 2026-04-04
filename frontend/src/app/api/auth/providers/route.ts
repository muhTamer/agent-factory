import { NextResponse } from "next/server";

/**
 * Returns which OAuth providers are configured (have real credentials).
 * The login page uses this to only show buttons for available providers.
 */
export async function GET() {
  const available: string[] = [];

  if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
    available.push("google");
  }
  if (process.env.MICROSOFT_CLIENT_ID && process.env.MICROSOFT_CLIENT_SECRET) {
    available.push("microsoft-entra-id");
  }
  if (
    process.env.FACEBOOK_CLIENT_ID &&
    process.env.FACEBOOK_CLIENT_SECRET &&
    process.env.FACEBOOK_CLIENT_ID !== "placeholder"
  ) {
    available.push("facebook");
  }
  if (
    process.env.APPLE_CLIENT_ID &&
    process.env.APPLE_CLIENT_SECRET &&
    process.env.APPLE_CLIENT_ID !== "placeholder"
  ) {
    available.push("apple");
  }

  return NextResponse.json({ providers: available });
}
