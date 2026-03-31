# app/runtime/rate_limiter.py
"""
Rate limiting and bot protection middleware for public deployments.

Three layers of protection:
1. IP-based rate limiting — prevents brute-force / DDoS
2. Session-based LLM usage caps — prevents individual cost overruns
3. Bot fingerprinting — blocks automated abuse
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# ── Environment mode ─────────────────────────────────────────────
# Set AF_ENV=development to disable all rate limiting & usage caps locally.
AF_ENV = os.getenv("AF_ENV", "production").lower()
_LIMITS_DISABLED = AF_ENV in ("development", "dev", "test")

# ── Configuration (env-overridable) ─────────────────────────────

# IP rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))  # per window
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Session LLM usage caps
SESSION_MAX_LLM_CALLS = int(os.getenv("SESSION_MAX_LLM_CALLS", "50"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour

# Daily global LLM budget
DAILY_MAX_LLM_CALLS = int(os.getenv("DAILY_MAX_LLM_CALLS", "5000"))
DAILY_WARN_THRESHOLD = float(os.getenv("DAILY_WARN_THRESHOLD", "0.8"))  # 80%

# Bot protection
BOT_BLOCK_ENABLED = os.getenv("BOT_BLOCK_ENABLED", "true").lower() == "true"

# Known bot User-Agent substrings
_BOT_SIGNATURES = [
    "bot", "crawler", "spider", "scrapy", "curl", "wget", "httpx",
    "python-requests", "go-http-client", "java/", "libwww",
    "headlesschrome", "phantomjs", "selenium",
]

# Paths that skip rate limiting (health checks, etc.)
_EXEMPT_PATHS = {"/health", "/version", "/docs", "/openapi.json", "/redoc"}


# ── Data structures ─────────────────────────────────────────────

@dataclass
class _RateBucket:
    """Sliding-window counter for a single IP."""
    timestamps: list[float] = field(default_factory=list)

    def count_in_window(self, now: float, window: float) -> int:
        cutoff = now - window
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        return len(self.timestamps)

    def record(self, now: float) -> None:
        self.timestamps.append(now)


@dataclass
class _SessionUsage:
    """Tracks LLM call count and last-active time for a session."""
    llm_calls: int = 0
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.last_active) > ttl

    def touch(self) -> None:
        self.last_active = time.time()


@dataclass
class _DailyCounter:
    """Global daily LLM call counter."""
    count: int = 0
    date: str = ""

    def today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def reset_if_new_day(self) -> None:
        today = self.today()
        if self.date != today:
            self.count = 0
            self.date = today

    def increment(self) -> int:
        self.reset_if_new_day()
        self.count += 1
        return self.count


# ── Singletons ──────────────────────────────────────────────────

_ip_buckets: dict[str, _RateBucket] = defaultdict(_RateBucket)
_sessions: dict[str, _SessionUsage] = {}
_daily = _DailyCounter()


# ── Helper functions ────────────────────────────────────────────

def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _get_session_id(request: Request) -> str:
    """
    Derive a session identifier from thread_id (in body) or a
    fingerprint of IP + User-Agent. This does NOT require cookies.
    """
    # If the request has a thread_id in query or was already parsed, use it
    thread_id = request.headers.get("x-thread-id")
    if thread_id:
        return f"thread:{thread_id}"

    # Fallback: fingerprint from IP + User-Agent
    ip = _get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    raw = f"{ip}:{ua}"
    return f"fp:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _is_bot(request: Request) -> bool:
    """Simple bot detection based on User-Agent heuristics."""
    if not BOT_BLOCK_ENABLED:
        return False
    ua = (request.headers.get("user-agent") or "").lower()
    if not ua:
        return True  # No User-Agent = suspicious
    return any(sig in ua for sig in _BOT_SIGNATURES)


# ── Public API (used by middleware and session tracker) ──────────

def record_llm_call(session_id: str) -> dict:
    """
    Called after each LLM invocation to track usage.
    Returns a status dict with remaining quota info.
    Raises HTTPException if limits exceeded.
    """
    if _LIMITS_DISABLED:
        return {
            "session_llm_calls": 0,
            "session_llm_limit": SESSION_MAX_LLM_CALLS,
            "session_remaining": SESSION_MAX_LLM_CALLS,
            "daily_llm_calls": 0,
            "daily_llm_limit": DAILY_MAX_LLM_CALLS,
            "daily_warning": False,
        }

    # Session limit
    session = _sessions.get(session_id)
    if session is None:
        session = _SessionUsage()
        _sessions[session_id] = session

    # Expire stale sessions
    if session.is_expired(SESSION_TTL_SECONDS):
        session = _SessionUsage()
        _sessions[session_id] = session

    session.llm_calls += 1
    session.touch()

    if session.llm_calls > SESSION_MAX_LLM_CALLS:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "session_limit_exceeded",
                "message": f"Session LLM limit reached ({SESSION_MAX_LLM_CALLS} calls). "
                           "Please start a new session.",
                "limit": SESSION_MAX_LLM_CALLS,
                "used": session.llm_calls,
            },
        )

    # Daily global limit
    daily_count = _daily.increment()
    if daily_count > DAILY_MAX_LLM_CALLS:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily_limit_exceeded",
                "message": "System daily LLM budget exhausted. Please try again tomorrow.",
                "limit": DAILY_MAX_LLM_CALLS,
            },
        )

    return {
        "session_llm_calls": session.llm_calls,
        "session_llm_limit": SESSION_MAX_LLM_CALLS,
        "session_remaining": SESSION_MAX_LLM_CALLS - session.llm_calls,
        "daily_llm_calls": daily_count,
        "daily_llm_limit": DAILY_MAX_LLM_CALLS,
        "daily_warning": daily_count >= int(DAILY_MAX_LLM_CALLS * DAILY_WARN_THRESHOLD),
    }


def get_session_usage(session_id: str) -> dict:
    """Return current usage stats for a session."""
    session = _sessions.get(session_id)
    _daily.reset_if_new_day()
    return {
        "session_llm_calls": session.llm_calls if session else 0,
        "session_llm_limit": SESSION_MAX_LLM_CALLS,
        "session_remaining": (SESSION_MAX_LLM_CALLS - session.llm_calls) if session else SESSION_MAX_LLM_CALLS,
        "daily_llm_calls": _daily.count,
        "daily_llm_limit": DAILY_MAX_LLM_CALLS,
    }


def get_daily_usage() -> dict:
    """Return global daily usage stats (for monitoring endpoints)."""
    _daily.reset_if_new_day()
    return {
        "date": _daily.date,
        "daily_llm_calls": _daily.count,
        "daily_llm_limit": DAILY_MAX_LLM_CALLS,
        "utilization_pct": round((_daily.count / DAILY_MAX_LLM_CALLS) * 100, 1) if DAILY_MAX_LLM_CALLS > 0 else 0,
        "warning": _daily.count >= int(DAILY_MAX_LLM_CALLS * DAILY_WARN_THRESHOLD),
        "active_sessions": len([s for s in _sessions.values() if not s.is_expired(SESSION_TTL_SECONDS)]),
    }


# ── FastAPI Middleware ──────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces:
    1. Bot blocking (User-Agent fingerprinting)
    2. IP-based rate limiting (sliding window)
    3. Adds rate-limit headers to every response
    """

    async def dispatch(self, request: Request, call_next):
        # In development/test mode, skip all protection
        if _LIMITS_DISABLED:
            return await call_next(request)

        path = request.url.path

        # Skip rate limiting for health/meta endpoints
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        # ── Bot detection ───────────────────────────────
        if _is_bot(request):
            return JSONResponse(
                status_code=403,
                content={
                    "error": "bot_detected",
                    "message": "Automated access is not permitted.",
                },
            )

        # ── IP rate limiting ────────────────────────────
        ip = _get_client_ip(request)
        now = time.time()
        bucket = _ip_buckets[ip]
        count = bucket.count_in_window(now, RATE_LIMIT_WINDOW_SECONDS)

        if count >= RATE_LIMIT_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Limit: {RATE_LIMIT_REQUESTS} "
                               f"per {RATE_LIMIT_WINDOW_SECONDS}s.",
                    "retry_after": RATE_LIMIT_WINDOW_SECONDS,
                },
                headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
            )

        bucket.record(now)

        # ── Process the request ─────────────────────────
        response = await call_next(request)

        # Add rate-limit headers
        remaining = RATE_LIMIT_REQUESTS - bucket.count_in_window(time.time(), RATE_LIMIT_WINDOW_SECONDS)
        response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Window"] = str(RATE_LIMIT_WINDOW_SECONDS)

        return response
