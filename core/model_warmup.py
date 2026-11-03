"""
Model warmup — pre-loads models at startup to reduce first-request latency.
Only runs if WARMUP_MODELS=true (disabled in dev by default to save RAM).
"""

import asyncio
import logging

from core.config import settings

logger = logging.getLogger(__name__)


async def warmup_all() -> None:
    """Warm up all configured model backends concurrently."""
    tasks = []

    if settings.VAI_ASR_BACKEND in ("indicwhisper", "wav2vec2"):
        tasks.append(_warmup_asr())

    if settings.VAI_MT_BACKEND == "indictrans2":
        tasks.append(_warmup_mt())

    if settings.VAI_TTS_BACKEND == "indic_parler":
        tasks.append(_warmup_tts())

    if settings.VAI_LID_BACKEND == "indiclid":
        tasks.append(_warmup_lid())

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.warning("Warmup error (non-fatal): %s", result)


async def _warmup_asr() -> None:
    try:
        from services.asr.dispatcher import ASRDispatcher

        d = ASRDispatcher()
        backend = d._backends.get(settings.VAI_ASR_BACKEND)
        if backend and hasattr(backend, "_ensure_loaded"):
            await backend._ensure_loaded()
            logger.info("ASR warmed up: %s", settings.VAI_ASR_BACKEND)
    except Exception as e:
        logger.warning("ASR warmup failed: %s", e)


async def _warmup_mt() -> None:
    try:
        from services.mt.dispatcher import MTDispatcher

        d = MTDispatcher()
        backend = d._backends.get("indictrans2")
        if backend:
            await backend._ensure_loaded("en_indic")
            logger.info("MT (en_indic) warmed up")
    except Exception as e:
        logger.warning("MT warmup failed: %s", e)


async def _warmup_tts() -> None:
    try:
        from services.tts.dispatcher import TTSDispatcher

        d = TTSDispatcher()
        backend = d._backends.get("indic_parler")
        if backend and hasattr(backend, "_ensure_loaded"):
            await backend._ensure_loaded()
            logger.info("TTS warmed up")
    except Exception as e:
        logger.warning("TTS warmup failed: %s", e)


async def _warmup_lid() -> None:
    try:
        from services.lid.dispatcher import LIDDispatcher

        d = LIDDispatcher()
        backend = d._backends.get("indiclid")
        if backend and hasattr(backend, "_ensure_loaded"):
            await backend._ensure_loaded()
            logger.info("LID warmed up")
    except Exception as e:
        logger.warning("LID warmup failed: %s", e)
