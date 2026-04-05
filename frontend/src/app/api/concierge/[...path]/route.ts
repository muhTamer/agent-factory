import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const CONCIERGE_URL =
  process.env.NEXT_PUBLIC_CONCIERGE_API || "http://127.0.0.1:8001";

/**
 * Server-side proxy: /api/concierge/* → concierge backend /concierge/*
 * Eliminates CORS — browser only talks to the frontend origin.
 */
async function proxy(req: NextRequest) {
  const path = req.nextUrl.pathname.replace(/^\/api\/concierge/, "/concierge");
  const url = `${CONCIERGE_URL}${path}${req.nextUrl.search}`;

  const headers = new Headers();
  // Forward content-type and authorization
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  const auth = req.headers.get("authorization");
  if (auth) headers.set("authorization", auth);

  try {
    const body =
      req.method !== "GET" && req.method !== "HEAD"
        ? await req.arrayBuffer()
        : undefined;

    const res = await fetch(url, {
      method: req.method,
      headers,
      body,
    });

    const data = await res.arrayBuffer();
    return new NextResponse(data, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") || "application/json",
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`[PROXY] ${req.method} ${url} -> ERROR: ${msg}`);
    return NextResponse.json(
      { detail: `Proxy error: ${msg}`, target: url },
      { status: 502 }
    );
  }
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
