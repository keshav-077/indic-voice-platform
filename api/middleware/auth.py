"""
Authentication Middleware — JWT Bearer tokens + API Keys.
Both schemes are supported. API keys are for service-to-service calls.
JWT tokens are issued via /auth/token endpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import settings

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

# Paths that skip authentication
PUBLIC_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json", "/metrics"}


class AuthMiddleware:
    """Starlette-compatible middleware to reject unauthenticated requests early."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            await self.app(scope, receive, send)
            return

        await self.app(scope, receive, send)


# FastAPI dependency ──────────────────────────────────────────────────────────


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)] = None,
) -> dict:
    """
    Validates JWT Bearer token or X-API-Key header.
    Returns user context dict on success; raises 401 on failure.
    """
    # Check X-API-Key header first (service-to-service)
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return _validate_api_key(api_key)

    # Check Bearer token
    if credentials and credentials.scheme.lower() == "bearer":
        return _validate_jwt(credentials.credentials)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "authentication_required",
            "message": ("Provide a Bearer token (Authorization: Bearer <token>) or API key (X-API-Key: <key>)."),
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


def _validate_jwt(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.VAI_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "token_expired", "message": "JWT token has expired."},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_token", "message": str(e)},
        )


def _validate_api_key(key: str) -> dict:
    key_hash = _hash_key(key)

    if not settings.static_api_key_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "no_api_keys_configured",
                "message": "No API keys configured on server.",
            },
        )

    # Constant-time comparison against all configured keys
    matched = any(hmac.compare_digest(_hash_key(k).encode(), key_hash.encode()) for k in settings.static_api_key_list)
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_api_key", "message": "API key not recognized."},
        )

    return {"sub": "api_key_user", "key_hash": key_hash[:8], "scope": "api"}


def _hash_key(key: str) -> str:
    return hashlib.sha256(f"{key}{settings.VAI_API_KEY_SALT}".encode()).hexdigest()


# Token issuance (used by /auth/token endpoint) ───────────────────────────────


def create_access_token(user_id: str, scopes: list[str] | None = None) -> str:
    payload = {
        "sub": user_id,
        "scopes": scopes or ["api"],
        "iat": int(time.time()),
        "exp": int(time.time()) + (settings.JWT_EXPIRY_HOURS * 3600),
    }
    return jwt.encode(payload, settings.VAI_JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


# Keep APIKeyMiddleware alias for compatibility with main.py import
APIKeyMiddleware = AuthMiddleware
