import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 120; // allow up to 2 minutes for LLM calls

const CONCIERGE_URL =
  process.env.NEXT_PUBLIC_CONCIERGE_API || "http://127.0.0.1:8001";

/**
 * Server-side proxy: /api/concierge/* → concierge backend /concierge/*
 * Eliminates CORS — browser only talks to the frontend origin.
 */
async function proxy(req: NextRequest) {
  const path = req.nextUrl.pathname.replace(/^\/api\/concierge/, "/concierge");
  const url = `${CONCIERGE_URL}${path}${req.nextUrl.search}`;

  console.log(`[PROXY] ${req.method} ${url}`);

  const headers = new Headers();
  // Forward content-type and authorization
  const ct = req.headers.get("content-type");
  if (ct) headers.set("content-type", ct);
  const auth = req.headers.get("authorization");
  if (auth) headers.set("authorization", auth);

  try {
    let body: ArrayBuffer | undefined;
    if (req.method !== "GET" && req.method !== "HEAD") {
      body = await req.arrayBuffer();
      console.log(`[PROXY] ${req.method} ${url} body=${body.byteLength} bytes`);
    }

    const res = await fetch(url, {
      method: req.method,
      headers,
      body,
      signal: AbortSignal.timeout(120000), // 2 min timeout
    });

    const data = await res.arrayBuffer();
    console.log(
      `[PROXY] ${req.method} ${url} -> ${res.status} (${data.byteLength} bytes)`
    );

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
