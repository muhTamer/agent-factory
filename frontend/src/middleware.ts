import { auth } from "@/auth";

/**
 * Next.js middleware — protects all routes behind auth.
 * When AUTH_SECRET is not set (local dev), skip auth entirely.
 */
export default auth((req) => {
  // If no AUTH_SECRET configured, allow all requests (local dev)
  if (!process.env.AUTH_SECRET) return;

  const isLoggedIn = !!req.auth?.user;
  const isOnLogin = req.nextUrl.pathname.startsWith("/login");
  const isAuthRoute = req.nextUrl.pathname.startsWith("/api/auth");

  if (isAuthRoute || isOnLogin) return;

  if (!isLoggedIn) {
    return Response.redirect(new URL("/login", req.nextUrl.origin));
  }
});

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico).*)",
  ],
};
