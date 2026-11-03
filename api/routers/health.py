"""
Health Router — /health, /health/ready, /health/live, /metrics.
Provides Kubernetes-compatible liveness and readiness probes.
"""

import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.schemas.models import HealthResponse, ServiceStatus
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Full health check — validates all service backends.
    Returns 200 if OK, 207 if degraded, 503 if critical services are down.
    """
    services = await _check_all_services()
    any_down = any(s.status == "down" for s in services.values())
    any_degraded = any(s.status == "degraded" for s in services.values())

    overall = "down" if any_down else ("degraded" if any_degraded else "ok")

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        services=services,
        uptime_sec=round(time.time() - _start_time, 1),
    )


@router.get("/health/live", tags=["Health"])
async def liveness():
    """Kubernetes liveness probe — just confirms the process is alive."""
    return {"status": "alive"}


@router.get("/health/ready", tags=["Health"])
async def readiness():
    """Kubernetes readiness probe — confirms the app can serve traffic."""
    # Quick Redis check
    redis_ok = await _check_redis()
    if not redis_ok:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Redis unavailable"},
        )
    return {"status": "ready"}


@router.get("/metrics", response_class=PlainTextResponse, tags=["Health"])
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.PROMETHEUS_ENABLED:
        return PlainTextResponse("# metrics disabled\n")
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return PlainTextResponse("# prometheus_client not installed\n")


async def _check_all_services() -> dict[str, ServiceStatus]:
    checks = await asyncio.gather(
        _check_redis_service(),
        _check_db_service(),
        _check_asr_service(),
        _check_mt_service(),
        _check_tts_service(),
        return_exceptions=True,
    )
    keys = ["redis", "database", "asr", "mt", "tts"]
    services = {}
    for key, result in zip(keys, checks, strict=False):
        if isinstance(result, Exception):
            services[key] = ServiceStatus(status="down", backend="unknown", error=str(result))
        else:
            services[key] = result
    return services


async def _check_redis() -> bool:
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.VAI_REDIS_URL)
        await asyncio.wait_for(r.ping(), timeout=2)
        return True
    except Exception:
        return False


async def _check_redis_service() -> ServiceStatus:
    start = time.perf_counter()
    ok = await _check_redis()
    ms = round((time.perf_counter() - start) * 1000, 1)
    return ServiceStatus(status="ok" if ok else "down", backend="redis", latency_ms=ms)


async def _check_db_service() -> ServiceStatus:
    start = time.perf_counter()
    from core.db import is_db_available

    if not is_db_available():
        return ServiceStatus(status="ok", backend="none (running without DB)", latency_ms=0)
    try:
        from sqlalchemy import text

        from core.db import get_db_session

        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceStatus(status="ok", backend="postgresql", latency_ms=ms)
    except Exception as e:
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceStatus(status="down", backend="postgresql", latency_ms=ms, error=str(e))


async def _check_asr_service() -> ServiceStatus:
    start = time.perf_counter()
    try:
        # Reuse the singleton from the router instead of creating a new dispatcher
        from api.routers.asr import get_dispatcher

        d = get_dispatcher()
        backends = list(d._backends.keys())
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceStatus(
            status="ok" if backends else "down",
            backend=",".join(backends) or "none",
            latency_ms=ms,
        )
    except Exception as e:
        return ServiceStatus(status="degraded", backend=settings.VAI_ASR_BACKEND, error=str(e))


async def _check_mt_service() -> ServiceStatus:
    start = time.perf_counter()
    try:
        from api.routers.mt import get_dispatcher

        d = get_dispatcher()
        backends = list(d._backends.keys())
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceStatus(
            status="ok" if backends else "down",
            backend=",".join(backends) or "none",
            latency_ms=ms,
        )
    except Exception as e:
        return ServiceStatus(status="degraded", backend=settings.VAI_MT_BACKEND, error=str(e))


async def _check_tts_service() -> ServiceStatus:
    start = time.perf_counter()
    try:
        from api.routers.tts import get_dispatcher

        d = get_dispatcher()
        backends = list(d._backends.keys())
        ms = round((time.perf_counter() - start) * 1000, 1)
        return ServiceStatus(
            status="ok" if backends else "down",
            backend=",".join(backends) or "none",
            latency_ms=ms,
        )
    except Exception as e:
        return ServiceStatus(status="degraded", backend=settings.VAI_TTS_BACKEND, error=str(e))
