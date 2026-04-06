import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const CONCIERGE_URL =
  process.env.NEXT_PUBLIC_CONCIERGE_API || "http://127.0.0.1:8001";
const RUNTIME_URL =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:808";

/**
 * Diagnostic endpoint that tests connectivity to all backends
 * and returns results in one JSON response.
 * Access via: /api/diagnostics
 */
export async function GET() {
  const results: Record<string, unknown> = {
    timestamp: new Date().toISOString(),
    env: {
      concierge_url: CONCIERGE_URL,
      runtime_url: RUNTIME_URL,
      auth_secret_set: !!process.env.AUTH_SECRET,
      node_env: process.env.NODE_ENV,
    },
    tests: {},
  };

  // Test 1: GET concierge debug
  try {
    const t0 = Date.now();
    const res = await fetch(`${CONCIERGE_URL}/concierge/debug`, {
      signal: AbortSignal.timeout(10000),
    });
    const body = await res.json();
    results.tests = {
      ...results.tests as object,
      concierge_get: {
        ok: true,
        status: res.status,
        elapsed_ms: Date.now() - t0,
        data: body,
      },
    };
  } catch (err) {
    results.tests = {
      ...results.tests as object,
      concierge_get: {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      },
    };
  }

  // Test 2: POST concierge cors-test
  try {
    const t0 = Date.now();
    const res = await fetch(`${CONCIERGE_URL}/concierge/cors-test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ test: true }),
      signal: AbortSignal.timeout(10000),
    });
    const body = await res.json();
    results.tests = {
      ...results.tests as object,
      concierge_post: {
        ok: true,
        status: res.status,
        elapsed_ms: Date.now() - t0,
        data: body,
      },
    };
  } catch (err) {
    results.tests = {
      ...results.tests as object,
      concierge_post: {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      },
    };
  }

  // Test 3: GET runtime health
  try {
    const t0 = Date.now();
    const res = await fetch(`${RUNTIME_URL}/health`, {
      signal: AbortSignal.timeout(10000),
    });
    const body = await res.json();
    results.tests = {
      ...results.tests as object,
      runtime_health: {
        ok: true,
        status: res.status,
        elapsed_ms: Date.now() - t0,
        data: body,
      },
    };
  } catch (err) {
    results.tests = {
      ...results.tests as object,
      runtime_health: {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      },
    };
  }

  // Test 4: GET runtime debug
  try {
    const t0 = Date.now();
    const res = await fetch(`${RUNTIME_URL}/debug`, {
      signal: AbortSignal.timeout(10000),
    });
    const body = await res.json();
    results.tests = {
      ...results.tests as object,
      runtime_debug: {
        ok: true,
        status: res.status,
        elapsed_ms: Date.now() - t0,
        data: body,
      },
    };
  } catch (err) {
    results.tests = {
      ...results.tests as object,
      runtime_debug: {
        ok: false,
        error: err instanceof Error ? err.message : String(err),
      },
    };
  }

  return NextResponse.json(results, { status: 200 });
}
