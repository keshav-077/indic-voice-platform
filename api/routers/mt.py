"""
MT Router — Machine Translation endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.middleware.auth import get_current_user
from api.schemas.models import APIError, MTBatchRequest, MTBatchResponse, MTRequest, MTResponse
from core.config import settings
from core.metrics import record_api_call
from services.mt.dispatcher import MTDispatcher

logger = logging.getLogger(__name__)
router = APIRouter()
_dispatcher: MTDispatcher | None = None


def get_dispatcher() -> MTDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = MTDispatcher(default_backend=settings.VAI_MT_BACKEND)
    return _dispatcher


@router.post(
    "/translate",
    response_model=MTResponse,
    responses={400: {"model": APIError}, 429: {"model": APIError}},
    summary="Translate text between Indian languages",
    description=(
        "Translate text between any of 22 Indic languages or English. "
        "Uses IndicTrans2 (1B parameters) by default. Supports direct Indic↔Indic translation "
        "without English pivot."
    ),
)
async def translate(
    request: MTRequest,
    dispatcher: MTDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> MTResponse:
    try:
        response = await dispatcher.translate(request)
        record_api_call("mt", response.backend_used, success=True)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "invalid_input", "message": str(e)})
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail={"error": "backend_unavailable", "message": str(e)})


@router.post(
    "/translate/batch",
    response_model=MTBatchResponse,
    summary="Batch translate up to 50 sentences",
    description="Efficiently translate multiple sentences in one call. Groups by language pair internally.",
)
async def translate_batch(
    request: MTBatchRequest,
    dispatcher: MTDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> MTBatchResponse:
    return await dispatcher.translate_batch(request)
