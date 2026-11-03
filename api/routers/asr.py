"""
ASR Router — Speech-to-Text endpoints.
Supports multiple backends: IndicWhisper, BHASHINI, Wav2Vec2.
"""

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.middleware.auth import get_current_user
from api.schemas.models import APIError, ASRRequest, ASRResponse
from core.config import settings
from core.metrics import record_api_call
from services.asr.dispatcher import ASRDispatcher

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared dispatcher instance (initialized at startup)
_dispatcher: ASRDispatcher | None = None


def get_dispatcher() -> ASRDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ASRDispatcher(default_backend=settings.VAI_ASR_BACKEND)
    return _dispatcher


@router.post(
    "/asr",
    response_model=ASRResponse,
    responses={
        400: {
            "model": APIError,
            "description": "Bad request (invalid audio, unsupported language)",
        },
        401: {"model": APIError, "description": "Missing or invalid auth token"},
        413: {"model": APIError, "description": "Audio too large (max 25MB)"},
        429: {"model": APIError, "description": "Rate limit exceeded"},
        503: {"model": APIError, "description": "ASR backend unavailable"},
    },
    summary="Transcribe audio to text",
    description=(
        "Accepts audio as a URL or base64-encoded bytes and returns a transcript "
        "with confidence scores and optional word-level timestamps. "
        "Supports 22 Indian languages + English. Auto-detects language if `lang` is omitted."
    ),
)
async def transcribe(
    request: ASRRequest,
    dispatcher: ASRDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> ASRResponse:
    """Transcribe audio in an Indian language to text."""
    # DPDP 2023: consent required for audio processing
    if settings.CONSENT_REQUIRED and not request.consent_given:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "consent_required",
                "message": (
                    "Under India's DPDP Act 2023, explicit user consent is required "
                    "for processing audio. Set consent_given=true to acknowledge."
                ),
            },
        )

    try:
        response = await dispatcher.transcribe(request)
        record_api_call("asr", response.backend_used, success=True, lang=response.lang_detected)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})
    except TimeoutError:
        raise HTTPException(
            status_code=503,
            detail={"error": "backend_timeout", "message": "ASR backend timed out. Try again."},
        )


@router.post(
    "/asr/upload",
    response_model=ASRResponse,
    summary="Transcribe uploaded audio file",
    description="Upload an audio file directly (multipart/form-data). Max 25MB.",
)
async def transcribe_upload(
    file: Annotated[UploadFile, File(description="Audio file (WAV, MP3, OPUS, OGG, FLAC)")],
    lang: Annotated[str | None, Form()] = None,
    return_word_timestamps: Annotated[bool, Form()] = False,
    consent_given: Annotated[bool, Form()] = False,
    backend: Annotated[str, Form()] = "auto",
    dispatcher: ASRDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> ASRResponse:
    # Validate size
    contents = await file.read()
    max_bytes = settings.AUDIO_UPLOAD_MAX_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"Audio file exceeds {settings.AUDIO_UPLOAD_MAX_MB}MB limit.",
            },
        )

    import base64

    audio_b64 = base64.b64encode(contents).decode()
    request = ASRRequest(
        audio_base64=audio_b64,
        lang=lang,
        return_word_timestamps=return_word_timestamps,
        consent_given=consent_given,
        backend=backend,
    )
    return await transcribe(request, dispatcher, user)


@router.websocket("/asr/stream")
async def stream_transcribe(websocket) -> None:
    """
    Streaming ASR via WebSocket.

    Protocol:
    1. Client sends JSON config: {"lang": "hi", "consent_given": true}
    2. Client sends audio chunks as binary frames (16kHz mono PCM WAV)
    3. Server streams partial transcripts as JSON: {"partial": "नमस्ते..."}
    4. On silence/end, server sends final: {"final": "नमस्ते कैसे हैं", "confidence": 0.94}
    5. Client sends "END" to close stream
    """
    await websocket.accept()
    dispatcher = get_dispatcher()

    try:
        # Receive config
        config_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        lang = config_msg.get("lang")
        consent = config_msg.get("consent_given", False)

        if settings.CONSENT_REQUIRED and not consent:
            await websocket.send_json(
                {
                    "error": "consent_required",
                    "message": "Set consent_given=true in config message.",
                }
            )
            await websocket.close(code=1008)
            return

        await websocket.send_json({"status": "ready", "lang": lang or "auto"})

        audio_buffer = bytearray()

        while True:
            msg = await asyncio.wait_for(websocket.receive(), timeout=30)

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg:
                audio_buffer.extend(msg["bytes"])
                # Send partial every ~2 seconds of audio (32000 bytes at 16kHz 16-bit mono)
                if len(audio_buffer) >= 32000:
                    import base64

                    chunk = base64.b64encode(bytes(audio_buffer)).decode()
                    partial = await dispatcher.transcribe_partial(chunk, lang)
                    await websocket.send_json({"partial": partial})
                    audio_buffer = bytearray()

            elif "text" in msg and msg["text"] == "END":
                # Final transcription
                if audio_buffer:
                    import base64

                    remaining = base64.b64encode(bytes(audio_buffer)).decode()
                    req = ASRRequest(
                        audio_base64=remaining,
                        lang=lang,
                        return_word_timestamps=True,
                        consent_given=True,
                        backend="auto",
                    )
                    result = await dispatcher.transcribe(req)
                    await websocket.send_json(
                        {
                            "final": result.transcript,
                            "confidence": result.confidence,
                            "words": [w.model_dump() for w in result.words],
                        }
                    )
                break

    except TimeoutError:
        await websocket.send_json({"error": "timeout", "message": "Connection timed out."})
    except Exception as e:
        logger.exception("Streaming ASR error: %s", e)
        await websocket.send_json({"error": "server_error", "message": str(e)})
    finally:
        await websocket.close()
