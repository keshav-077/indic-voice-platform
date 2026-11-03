"""TTS Router — Text-to-Speech endpoints."""

import base64
import io
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.middleware.auth import get_current_user
from api.schemas.models import APIError, TTSRequest, TTSResponse
from core.config import settings
from core.metrics import record_api_call
from services.tts.dispatcher import TTSDispatcher

logger = logging.getLogger(__name__)
router = APIRouter()
_dispatcher: TTSDispatcher | None = None


def get_dispatcher() -> TTSDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = TTSDispatcher(default_backend=settings.VAI_TTS_BACKEND)
    return _dispatcher


@router.post(
    "/tts",
    response_model=TTSResponse,
    responses={400: {"model": APIError}, 429: {"model": APIError}},
    summary="Convert text to speech",
    description=(
        "Synthesize natural-sounding speech from text in 22 Indian languages. "
        "Uses Indic Parler-TTS (44 speakers) by default. Returns base64 audio for short clips "
        "or a presigned URL for longer outputs."
    ),
)
async def synthesize(
    request: TTSRequest,
    dispatcher: TTSDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> TTSResponse:
    try:
        response = await dispatcher.synthesize(request)
        record_api_call("tts", response.backend_used, success=True, lang=request.lang)
        return response
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail={"error": "backend_unavailable", "message": str(e)})


@router.post(
    "/tts/stream",
    summary="Stream synthesized audio directly",
    description="Returns audio as a streaming binary response (Content-Type: audio/mpeg or audio/wav).",
    response_class=StreamingResponse,
)
async def synthesize_stream(
    request: TTSRequest,
    dispatcher: TTSDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
):
    response = await dispatcher.synthesize(request)

    if response.audio_base64:
        audio_bytes = base64.b64decode(response.audio_base64)
    else:
        raise HTTPException(
            status_code=503,
            detail={"error": "no_audio", "message": "Audio could not be generated."},
        )

    media_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "ogg": "audio/ogg",
    }.get(request.format.value, "audio/mpeg")

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{request.format.value}"},
    )
