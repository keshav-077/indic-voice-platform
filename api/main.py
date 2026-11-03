"""FastAPI app serving ASR, MT, TTS, LID, and NLU for 22 Indian languages."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth import APIKeyMiddleware
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routers import admin, asr, health, lid, mt, nlu, tts
from core.config import settings
from core.db import init_db
from core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    setup_logging(settings.LOG_LEVEL)
    logger.info("BharatVoiceAI starting up, version %s", settings.APP_VERSION)

    # Initialize database connection pool (optional, used for rate limiting state)
    if settings.VAI_DB_URL:
        try:
            await init_db()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(
                "Database unavailable (%s) — running without DB. Rate limiting will use in-memory storage.",
                e,
            )
    else:
        logger.info("No database configured (VAI_DB_URL empty) — running without DB")

    # Warm up model backends (optional, reduces first-request latency)
    if settings.WARMUP_MODELS:
        from core.model_warmup import warmup_all

        await warmup_all()
        logger.info("Models warmed up")

    yield  # Application runs here

    logger.info("BharatVoiceAI shutting down")


app = FastAPI(
    title="BharatVoiceAI",
    description=(
        "Open-source AI services for 22 Indian languages: ASR, Machine Translation, TTS, Language ID, and NLU."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    contact={
        "name": "BharatVoiceAI Community",
        "url": "https://github.com/samirdutta/bharatvoiceai",
    },
)

# Middleware (order matters: outermost runs first on request)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ORIGINS != ["*"],  # credentials not allowed with wildcard origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)


# Request timing header


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response


# Global exception handler


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": request.headers.get("X-Request-ID"),
        },
    )


# Routers

API_V1 = "/v1"

app.include_router(health.router, tags=["Health"])
app.include_router(asr.router, prefix=API_V1, tags=["ASR — Speech to Text"])
app.include_router(mt.router, prefix=API_V1, tags=["MT — Machine Translation"])
app.include_router(tts.router, prefix=API_V1, tags=["TTS — Text to Speech"])
app.include_router(lid.router, prefix=API_V1, tags=["LID — Language Identification"])
app.include_router(nlu.router, prefix=API_V1, tags=["NLU — Natural Language Understanding"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
