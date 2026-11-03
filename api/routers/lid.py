"""LID Router — Language Identification."""

import logging

from fastapi import APIRouter, Depends

from api.middleware.auth import get_current_user
from api.schemas.models import APIError, LIDRequest, LIDResponse
from core.config import settings
from services.lid.dispatcher import LIDDispatcher

logger = logging.getLogger(__name__)
router = APIRouter()
_dispatcher: LIDDispatcher | None = None


def get_dispatcher() -> LIDDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = LIDDispatcher(default_backend=settings.VAI_LID_BACKEND)
    return _dispatcher


@router.post(
    "/lid",
    response_model=LIDResponse,
    responses={400: {"model": APIError}},
    summary="Identify language of text or audio",
    description=(
        "Detect the language of input text or audio. Handles 47 languages including "
        "code-mixed varieties like Hinglish. Returns top-k candidates with confidence scores."
    ),
)
async def identify_language(
    request: LIDRequest,
    dispatcher: LIDDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> LIDResponse:
    return await dispatcher.identify(request)
