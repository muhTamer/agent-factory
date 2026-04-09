import { auth } from "@/auth";
import { SignJWT } from "jose";
import { NextResponse } from "next/server";

const IS_DEV = !process.env.AUTH_SECRET;
const SECRET = new TextEncoder().encode(
  process.env.AUTH_SECRET || "dev-secret-not-for-production"
);

export async function GET() {
  let session;
  try {
    session = await auth();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error("[TOKEN] auth() threw:", msg);
    return NextResponse.json(
      { error: "Auth error", detail: msg },
      { status: 500 }
    );
  }

  // In dev mode without auth, return a dev token immediately
  if (IS_DEV && session?.user?.email) {
    const user = session.user as Record<string, unknown>;
    const token = await new SignJWT({
      sub: user.email as string,
      email: user.email as string,
      name: (user.name as string) ?? "",
      tenant_id: (user.tenant_id as string) ?? "dev-tenant",
      provider: (user.provider as string) ?? "dev",
    })
      .setProtectedHeader({ alg: "HS256" })
      .setIssuedAt()
      .setExpirationTime("24h")
      .sign(SECRET);

    return NextResponse.json({ token });
  }

  if (!process.env.AUTH_SECRET) {
    return NextResponse.json(
      { error: "AUTH_SECRET not configured" },
      { status: 500 }
    );
  }

  if (!session?.user?.email) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const user = session.user as Record<string, unknown>;

  const token = await new SignJWT({
    sub: user.email as string,
    email: user.email as string,
    name: (user.name as string) ?? "",
    tenant_id: (user.tenant_id as string) ?? "",
    provider: (user.provider as string) ?? "",
  })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime("24h")
    .sign(SECRET);

  return NextResponse.json({ token });
}
