"""
Rate Limiting Middleware — Redis-backed sliding window algorithm.
Limits are per API key/user, per endpoint category.
Falls back to in-memory limiting if Redis is unavailable.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import settings

logger = logging.getLogger(__name__)

# Endpoint → RPM limit mapping
ENDPOINT_LIMITS = {
    "/v1/asr": settings.RATE_LIMIT_ASR_RPM,
    "/v1/asr/upload": settings.RATE_LIMIT_ASR_RPM,
    "/v1/tts": settings.RATE_LIMIT_TTS_RPM,
    "/v1/translate": settings.RATE_LIMIT_MT_RPM,
    "/v1/translate/batch": settings.RATE_LIMIT_MT_RPM // 2,
}
DEFAULT_RPM = settings.RATE_LIMIT_DEFAULT_RPM


def _get_limit(path: str) -> int:
    for prefix, limit in ENDPOINT_LIMITS.items():
        if path.startswith(prefix):
            return limit
    return DEFAULT_RPM


class InMemoryRateLimiter:
    """Fallback in-process sliding window limiter (not suitable for multi-replica)."""

    def __init__(self) -> None:
        self._windows: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_sec: int = 60) -> tuple[bool, int]:
        now = time.monotonic()
        window = self._windows[key]

        # Evict old entries
        while window and window[0] < now - window_sec:
            window.popleft()

        remaining = limit - len(window)
        if remaining <= 0:
            return False, 0

        window.append(now)
        return True, remaining - 1


_in_memory_limiter = InMemoryRateLimiter()
_redis_client = None


async def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(settings.VAI_REDIS_URL, decode_responses=True)
            await _redis_client.ping()
        except Exception as e:
            logger.warning("Redis unavailable (%s); using in-memory rate limiter.", e)
            _redis_client = "unavailable"
    return _redis_client if _redis_client != "unavailable" else None


async def _redis_sliding_window(redis, key: str, limit: int, window_sec: int = 60) -> tuple[bool, int]:
    """Redis sorted-set sliding window — works correctly across replicas."""
    now = time.time()
    pipe = redis.pipeline()
    pipe.zremrangebyscore(key, 0, now - window_sec)
    pipe.zcard(key)
    pipe.zadd(key, {str(now): now})
    pipe.expire(key, window_sec + 1)
    results = await pipe.execute()
    count = results[1]  # count BEFORE adding current request
    remaining = limit - count - 1
    if count >= limit:
        return False, 0
    return True, max(0, remaining)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-key rate limits. Skips public paths."""

    SKIP_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.SKIP_PATHS:
            return await call_next(request)

        # Identify caller — prefer API key, fall back to IP
        api_key = request.headers.get("X-API-Key", "")
        ip = request.client.host if request.client else "unknown"
        caller_id = f"key:{api_key[:12]}" if api_key else f"ip:{ip}"
        rate_key = f"vai:rl:{caller_id}:{path.split('/')[2] if len(path.split('/')) > 2 else 'misc'}"

        limit = _get_limit(path)
        redis = await _get_redis()

        if redis:
            try:
                allowed, remaining = await _redis_sliding_window(redis, rate_key, limit)
            except Exception:
                # Redis failed mid-request — fall back to in-memory
                allowed, remaining = _in_memory_limiter.is_allowed(caller_id + path[:20], limit)
        else:
            allowed, remaining = _in_memory_limiter.is_allowed(caller_id + path[:20], limit)

        # Always add rate-limit headers
        async def add_headers(response):
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Window"] = "60"
            return response

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit of {limit} requests/minute exceeded. Please slow down.",
                    "retry_after_sec": 60,
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)
        return await add_headers(response)
