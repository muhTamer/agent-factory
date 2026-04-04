# app/auth.py
"""
JWT authentication for multi-tenant API access.

The frontend (NextAuth.js) handles OAuth login and issues a signed JWT
containing user_id, email, and tenant_id. The backend validates this
JWT on every request using the shared AUTH_SECRET.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

AUTH_SECRET = os.getenv("AUTH_SECRET", "")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthUser:
    """Authenticated user extracted from JWT."""

    user_id: str
    email: str
    name: str
    tenant_id: str
    provider: str


def tenant_id_from_email(email: str) -> str:
    """Derive a stable tenant_id from a user's email."""
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:16]


def decode_token(token: str) -> AuthUser:
    """Decode and validate a JWT token."""
    if not AUTH_SECRET:
        raise HTTPException(
            status_code=500,
            detail="AUTH_SECRET not configured on server.",
        )
    try:
        payload = jwt.decode(token, AUTH_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    email = payload.get("email", "")
    return AuthUser(
        user_id=payload.get("sub", ""),
        email=email,
        name=payload.get("name", ""),
        tenant_id=payload.get("tenant_id") or tenant_id_from_email(email),
        provider=payload.get("provider", ""),
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser:
    """
    FastAPI dependency that extracts the authenticated user from the request.
    When AUTH_ENABLED=false (local dev), returns a default dev user.
    """
    if not AUTH_ENABLED:
        return AuthUser(
            user_id="dev",
            email="dev@localhost",
            name="Developer",
            tenant_id="dev",
            provider="local",
        )

    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in.",
        )

    return decode_token(credentials.credentials)


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthUser | None:
    """
    Same as get_current_user but returns None instead of raising 401.
    Use for endpoints that work both authenticated and unauthenticated.
    """
    if not AUTH_ENABLED:
        return AuthUser(
            user_id="dev",
            email="dev@localhost",
            name="Developer",
            tenant_id="dev",
            provider="local",
        )

    if credentials is None:
        return None

    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None
