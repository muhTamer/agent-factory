import { auth } from "@/auth";

/**
 * Next.js middleware — protects all routes behind auth.
 * When AUTH_SECRET is not set (local dev), skip auth entirely.
 */
export default auth((req) => {
  // If no AUTH_SECRET configured, allow all requests (local dev)
  if (!process.env.AUTH_SECRET) return;

  const { pathname } = req.nextUrl;
  const isLoggedIn = !!req.auth?.user;
  const isOnLogin = pathname.startsWith("/login");
  const isAuthRoute = pathname.startsWith("/api/auth");
  // Proxy routes handle their own auth via Authorization header
  const isProxyRoute =
    pathname.startsWith("/api/concierge") ||
    pathname.startsWith("/api/runtime");

  if (isAuthRoute || isOnLogin || isProxyRoute) return;

  if (!isLoggedIn) {
    return Response.redirect(new URL("/login", req.nextUrl.origin));
  }
});

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|api/concierge|api/runtime|api/diagnostics).*)",
  ],
};
