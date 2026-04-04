import { auth } from "@/auth";
import { SignJWT } from "jose";
import { NextResponse } from "next/server";

const SECRET = new TextEncoder().encode(process.env.AUTH_SECRET ?? "");

export async function GET() {
  const session = await auth();

  if (!session?.user?.email) {
    return NextResponse.json({ error: "Not authenticated" }, { status: 401 });
  }

  const user = session.user as Record<string, unknown>;

  // Create a simple JWT signed with AUTH_SECRET that the Python backend can validate
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
