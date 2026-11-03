"""
Admin Router — Token issuance, API key management, model info.
Protected by admin-level API key.
"""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.middleware.auth import create_access_token, get_current_user
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class TokenRequest(BaseModel):
    user_id: str
    scopes: list[str] = ["api"]
    note: str = ""


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


@router.post("/token", response_model=TokenResponse, summary="Issue a JWT access token")
async def issue_token(body: TokenRequest, user=Depends(get_current_user)):
    """Issue a JWT token for a user. Requires admin API key."""
    token = create_access_token(user_id=body.user_id, scopes=body.scopes)
    return TokenResponse(
        access_token=token,
        expires_in_seconds=settings.JWT_EXPIRY_HOURS * 3600,
    )


@router.get("/models", summary="List available model backends")
async def list_models(user=Depends(get_current_user)):
    """Returns the currently active backend for each service."""
    return {
        "asr": {
            "default": settings.VAI_ASR_BACKEND,
            "bhashini_configured": settings.is_bhashini_configured,
        },
        "mt": {
            "default": settings.VAI_MT_BACKEND,
            "sarvam_configured": settings.is_sarvam_configured,
        },
        "tts": {"default": settings.VAI_TTS_BACKEND},
        "lid": {"default": settings.VAI_LID_BACKEND},
        "nlu": {"default": settings.VAI_NLU_BACKEND},
    }


@router.get("/config", summary="Show non-sensitive config")
async def show_config(user=Depends(get_current_user)):
    return {
        "version": settings.APP_VERSION,
        "audio_logging": settings.VAI_AUDIO_LOGGING,
        "pii_masking": settings.VAI_PII_MASKING,
        "data_retention_days": settings.VAI_DATA_RETENTION_DAYS,
        "consent_required": settings.CONSENT_REQUIRED,
        "rate_limit_asr_rpm": settings.RATE_LIMIT_ASR_RPM,
        "rate_limit_mt_rpm": settings.RATE_LIMIT_MT_RPM,
        "rate_limit_tts_rpm": settings.RATE_LIMIT_TTS_RPM,
        "max_audio_duration_sec": settings.MAX_AUDIO_DURATION_SEC,
    }
