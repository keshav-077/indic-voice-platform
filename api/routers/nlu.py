"""NLU Router — Natural Language Understanding / Intent."""

import logging

from fastapi import APIRouter, Depends

from api.middleware.auth import get_current_user
from api.schemas.models import APIError, NLURequest, NLUResponse
from core.config import settings
from services.nlu.dispatcher import NLUDispatcher

logger = logging.getLogger(__name__)
router = APIRouter()
_dispatcher: NLUDispatcher | None = None


def get_dispatcher() -> NLUDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = NLUDispatcher(default_backend=settings.VAI_NLU_BACKEND)
    return _dispatcher


@router.post(
    "/nlu",
    response_model=NLUResponse,
    responses={400: {"model": APIError}},
    summary="Extract intent and entities from text",
    description=(
        "Classify intent and extract entities from text in any Indian language. "
        "Uses a fast rule engine first; escalates to Sarvam-M LLM for unknown intents. "
        "Optionally generates a natural-language response in the same language."
    ),
)
async def process_intent(
    request: NLURequest,
    dispatcher: NLUDispatcher = Depends(get_dispatcher),
    user=Depends(get_current_user),
) -> NLUResponse:
    return await dispatcher.process(request)
