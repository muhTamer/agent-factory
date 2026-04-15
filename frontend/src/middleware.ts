import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

/**
 * Next.js middleware — protects all routes behind auth.
 * When AUTH_SECRET is not set (local dev), skip auth entirely.
 */
export default async function middleware(req: NextRequest) {
  // No AUTH_SECRET = local dev, allow everything
  if (!process.env.AUTH_SECRET) return NextResponse.next();

  const { pathname } = req.nextUrl;
  const isOnLogin = pathname.startsWith("/login");
  const isAuthRoute = pathname.startsWith("/api/auth");
  const isProxyRoute =
    pathname.startsWith("/api/concierge") ||
    pathname.startsWith("/api/runtime");

  // Public routes — no auth check needed
  if (isAuthRoute || isOnLogin || isProxyRoute) return NextResponse.next();

  // Check for next-auth session cookie
  const sessionCookie =
    req.cookies.get("__Secure-authjs.session-token") ??
    req.cookies.get("authjs.session-token");

  if (!sessionCookie?.value) {
    return NextResponse.redirect(new URL("/login", req.nextUrl.origin));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|api/concierge|api/runtime|api/diagnostics).*)",
  ],
};
