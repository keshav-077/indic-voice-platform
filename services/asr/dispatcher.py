"""
ASR Dispatcher — routes ASR requests to the appropriate backend.
Backends: IndicWhisper (AI4Bharat), BHASHINI, Wav2Vec2-Indic, Local ONNX.

Follows the Strategy pattern: each backend is a class implementing BaseASRBackend.
The dispatcher selects backends based on config, language support, and availability.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from abc import ABC, abstractmethod

import httpx

from api.schemas.models import ASRRequest, ASRResponse, WordTimestamp
from core.config import settings

logger = logging.getLogger(__name__)


# Abstract Backend


class BaseASRBackend(ABC):
    """All ASR backends must implement this interface."""

    name: str
    supported_langs: set[str]

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, lang: str | None, **kwargs) -> dict:
        """
        Returns dict with:
            transcript: str
            confidence: float
            lang_detected: str
            words: list[dict]  (optional)
        """
        ...

    async def transcribe_partial(self, audio_bytes: bytes, lang: str | None) -> str:
        """Optional: streaming partial transcription. Default: full transcription."""
        result = await self.transcribe(audio_bytes, lang)
        # Return first few words as "partial"
        words = result.get("transcript", "").split()
        return " ".join(words[:5]) + ("..." if len(words) > 5 else "")

    def supports(self, lang: str | None) -> bool:
        if lang is None:
            return True  # will attempt LID
        return lang in self.supported_langs


# IndicWhisper Backend


class IndicWhisperBackend(BaseASRBackend):
    """
    AI4Bharat IndicWhisper — fine-tuned Whisper on Vistaar benchmark.
    Achieves lowest WER on 39/59 Vistaar benchmarks across 12 Indian languages.
    Model: ai4bharat/indicwhisper-large-v2
    """

    name = "indicwhisper"
    supported_langs = {"hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"}

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await asyncio.get_event_loop().run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        """Load model (runs in thread pool to avoid blocking event loop)."""
        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            logger.info("Loading IndicWhisper model: %s", settings.INDICWHISPER_MODEL)
            device = _resolve_device()
            self._processor = WhisperProcessor.from_pretrained(
                settings.INDICWHISPER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR,
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                settings.INDICWHISPER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            ).to(device)
            self._model.eval()
            self._loaded = True
            logger.info("IndicWhisper loaded on %s", device)
        except ImportError:
            raise RuntimeError("transformers not installed. Run: pip install transformers torch")
        except Exception as e:
            logger.error("Failed to load IndicWhisper: %s", e)
            raise

    async def transcribe(self, audio_bytes: bytes, lang: str | None = None, **kwargs) -> dict:
        await self._ensure_loaded()

        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_inference, audio_bytes, lang, kwargs.get("return_timestamps", False)
        )

    def _run_inference(self, audio_bytes: bytes, lang: str | None, return_timestamps: bool) -> dict:
        import torch

        audio_array = _bytes_to_audio_array(audio_bytes)
        device = next(self._model.parameters()).device

        inputs = self._processor(
            audio_array,
            sampling_rate=settings.ASR_TARGET_SAMPLE_RATE,
            return_tensors="pt",
        ).to(device)

        # Language forcing for better accuracy
        forced_decoder_ids = None
        if lang and lang in self.supported_langs:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=_lang_to_whisper_lang(lang),
                task="transcribe",
            )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                return_timestamps=return_timestamps,
                max_new_tokens=224,
            )

        result = self._processor.batch_decode(outputs, skip_special_tokens=True)
        transcript = result[0].strip() if result else ""

        return {
            "transcript": transcript,
            "confidence": 0.90,  # Whisper doesn't expose token-level conf; use fixed estimate
            "lang_detected": lang or "hi",
            "words": [],
        }


# BHASHINI Backend


class BhashiniASRBackend(BaseASRBackend):
    """
    BHASHINI ULCA API — Government of India's language AI platform.
    Free tier for PoC; contact BHASHINI team for production/commercial use.
    Registration: https://bhashini.gov.in/ulca/user/register
    """

    name = "bhashini"
    supported_langs = {
        "hi",
        "ta",
        "te",
        "bn",
        "mr",
        "gu",
        "kn",
        "ml",
        "pa",
        "or",
        "as",
        "ur",
        "mai",
        "kok",
        "mni",
        "doi",
        "ks",
        "ne",
        "sa",
    }

    COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    def __init__(self) -> None:
        if not settings.is_bhashini_configured:
            raise RuntimeError(
                "BHASHINI_USER_ID and BHASHINI_API_KEY must be set. "
                "Register at: https://bhashini.gov.in/ulca/user/register"
            )

    async def transcribe(self, audio_bytes: bytes, lang: str | None = None, **kwargs) -> dict:
        if not lang:
            lang = "hi"  # default; ideally run LID first

        audio_b64 = base64.b64encode(audio_bytes).decode()

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {"sourceLanguage": lang},
                        "serviceId": "",  # leave empty to auto-select best model
                        "audioFormat": "wav",
                        "samplingRate": settings.ASR_TARGET_SAMPLE_RATE,
                    },
                }
            ],
            "inputData": {
                "audio": [{"audioContent": audio_b64}],
            },
        }

        headers = {
            "userID": settings.BHASHINI_USER_ID,
            "ulcaApiKey": settings.BHASHINI_API_KEY,
            "Authorization": settings.BHASHINI_API_KEY,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.COMPUTE_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        # Parse BHASHINI response format
        try:
            output = data["pipelineResponse"][0]["output"][0]
            transcript = output.get("source", "")
            confidence = float(output.get("score", 0.85))
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Unexpected BHASHINI response format: %s | Raw: %s", e, data)
            raise ValueError("Unexpected BHASHINI response format.")

        return {
            "transcript": transcript,
            "confidence": confidence,
            "lang_detected": lang,
            "words": [],
        }


# Wav2Vec2 Indic Backend


class Wav2Vec2IndicBackend(BaseASRBackend):
    """
    AI4Bharat IndicWav2Vec — fine-tuned Wav2Vec2 for Indian languages.
    Lighter than Whisper; good for Hindi, Tamil, Telugu on CPU.
    """

    name = "wav2vec2"
    supported_langs = {"hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "or"}

    MODEL_MAP = {
        "hi": "ai4bharat/indicwav2vec-hindi",
        "ta": "ai4bharat/indicwav2vec-tamil",
        "te": "ai4bharat/indicwav2vec-telugu",
    }
    DEFAULT_MODEL = "ai4bharat/indicwav2vec-hindi"

    def __init__(self) -> None:
        self._models: dict = {}
        self._processors: dict = {}

    async def _ensure_loaded(self, lang: str) -> None:
        if lang not in self._models:
            model_name = self.MODEL_MAP.get(lang, self.DEFAULT_MODEL)
            await asyncio.get_event_loop().run_in_executor(None, self._load_model, lang, model_name)

    def _load_model(self, lang: str, model_name: str) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        logger.info("Loading Wav2Vec2 model: %s", model_name)
        device = _resolve_device()
        self._processors[lang] = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=settings.MODEL_CACHE_DIR)
        self._models[lang] = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=settings.MODEL_CACHE_DIR).to(device)

    async def transcribe(self, audio_bytes: bytes, lang: str | None = None, **kwargs) -> dict:
        lang = lang or "hi"
        await self._ensure_loaded(lang)

        return await asyncio.get_event_loop().run_in_executor(None, self._run_inference, lang, audio_bytes)

    def _run_inference(self, lang: str, audio_bytes: bytes) -> dict:
        import torch

        processor = self._processors[lang]
        model = self._models[lang]
        device = next(model.parameters()).device

        audio_array = _bytes_to_audio_array(audio_bytes)
        inputs = processor(
            audio_array,
            sampling_rate=settings.ASR_TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = processor.batch_decode(predicted_ids)[0]

        return {
            "transcript": transcript,
            "confidence": 0.85,
            "lang_detected": lang,
            "words": [],
        }


# Dispatcher


class ASRDispatcher:
    """
    Selects and calls the appropriate ASR backend.
    Falls back gracefully: primary → bhashini → error.
    """

    def __init__(self, default_backend: str = "indicwhisper") -> None:
        self._default = default_backend
        self._backends: dict[str, BaseASRBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        # Always register IndicWhisper (no external credentials needed)
        try:
            self._backends["indicwhisper"] = IndicWhisperBackend()
        except Exception as e:
            logger.warning("IndicWhisper backend unavailable: %s", e)

        # Register Wav2Vec2
        try:
            self._backends["wav2vec2"] = Wav2Vec2IndicBackend()
        except Exception as e:
            logger.warning("Wav2Vec2 backend unavailable: %s", e)

        # Register BHASHINI if configured
        if settings.is_bhashini_configured:
            try:
                self._backends["bhashini"] = BhashiniASRBackend()
            except Exception as e:
                logger.warning("BHASHINI ASR backend unavailable: %s", e)
        else:
            logger.info(
                "BHASHINI not configured. Set BHASHINI_USER_ID and BHASHINI_API_KEY "
                "to enable. Free registration: https://bhashini.gov.in/ulca/user/register"
            )

    def _select_backend(self, request: ASRRequest) -> BaseASRBackend:
        name = request.backend if request.backend != "auto" else self._default

        if name in self._backends:
            backend = self._backends[name]
            if backend.supports(request.lang):
                return backend
            # Requested backend doesn't support this language — fallback
            logger.warning("Backend %s does not support lang=%s. Falling back.", name, request.lang)

        # Fallback chain
        for fallback_name in ["indicwhisper", "bhashini", "wav2vec2"]:
            if fallback_name in self._backends:
                fb = self._backends[fallback_name]
                if fb.supports(request.lang):
                    logger.info("Using fallback backend: %s", fallback_name)
                    return fb

        raise RuntimeError("No available ASR backend supports the requested language.")

    async def transcribe(self, request: ASRRequest) -> ASRResponse:
        start = time.perf_counter()
        backend = self._select_backend(request)

        # Resolve audio bytes
        audio_bytes = await _resolve_audio_bytes(request)

        result = await backend.transcribe(
            audio_bytes,
            lang=request.lang,
            return_timestamps=request.return_word_timestamps,
        )

        words = [WordTimestamp(**w) for w in result.get("words", [])] if request.return_word_timestamps else []

        elapsed_ms = round((time.perf_counter() - start) * 1000)

        return ASRResponse(
            transcript=result["transcript"],
            confidence=result["confidence"],
            lang_detected=result["lang_detected"],
            words=words,
            backend_used=backend.name,
            processing_ms=elapsed_ms,
        )

    async def transcribe_partial(self, audio_b64: str, lang: str | None) -> str:
        backend = self._backends.get(self._default)
        if not backend:
            return ""
        audio_bytes = base64.b64decode(audio_b64)
        return await backend.transcribe_partial(audio_bytes, lang)


# Utilities


def _resolve_device() -> str:
    import torch

    if settings.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.DEVICE


def _bytes_to_audio_array(audio_bytes: bytes):
    """Convert raw audio bytes to a 16kHz mono float32 numpy array."""
    import io

    import soundfile as sf

    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
        audio = f.read(dtype="float32")
        sr = f.samplerate

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != settings.ASR_TARGET_SAMPLE_RATE:
        # Resample
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.ASR_TARGET_SAMPLE_RATE)

    return audio


async def _resolve_audio_bytes(request: ASRRequest) -> bytes:
    if request.audio_base64:
        return base64.b64decode(request.audio_base64)

    if request.audio_url:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(str(request.audio_url))
            resp.raise_for_status()
            return resp.content

    raise ValueError("No audio source provided.")


def _lang_to_whisper_lang(lang_code: str) -> str:
    """Map ISO-639-1 to Whisper language names."""
    mapping = {
        "hi": "hindi",
        "ta": "tamil",
        "te": "telugu",
        "bn": "bengali",
        "mr": "marathi",
        "gu": "gujarati",
        "kn": "kannada",
        "ml": "malayalam",
        "pa": "punjabi",
        "or": "odia",
        "as": "assamese",
        "ur": "urdu",
        "en": "english",
    }
    return mapping.get(lang_code, "hindi")
