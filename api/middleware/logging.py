"""Request logging middleware — DPDP-compliant: no PII logged by default."""

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from core.config import settings

logger = logging.getLogger("vai.access")

# PII patterns to mask in logs
PII_PATTERNS = [
    (re.compile(r"\b\d{10}\b"), "***PHONE***"),  # phone
    (re.compile(r"\b\d{12}\b"), "***AADHAAR***"),  # Aadhaar
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "***EMAIL***"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "***CARD***"),
]


def mask_pii(text: str) -> str:
    if not settings.VAI_PII_MASKING:
        return text
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/health", "/health/live", "/health/ready", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        start = time.perf_counter()

        response = await call_next(request)

        elapsed = round((time.perf_counter() - start) * 1000, 1)
        caller = request.headers.get("X-API-Key", "anon")[:8]

        logger.info(
            "%s %s %s %dms req=%s caller=%s",
            request.method,
            mask_pii(str(request.url.path)),
            response.status_code,
            elapsed,
            request_id,
            caller,
        )
        response.headers["X-Request-ID"] = request_id
        return response
