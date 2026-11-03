"""
TTS Dispatcher — Text-to-Speech backends.
Primary: Indic Parler-TTS (AI4Bharat) — 22 languages, 44 speakers.
Fallback: BHASHINI TTS, Local synthesis.

Indic Parler-TTS: https://huggingface.co/ai4bharat/indic-parler-tts
Open-source Indian language TTS server wrapping ai4bharat/indic-parler-tts.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from abc import ABC, abstractmethod

import httpx

from api.schemas.models import AudioFormat, TTSRequest, TTSResponse, TTSVoice
from core.config import settings

logger = logging.getLogger(__name__)

# Voice descriptions for Parler-TTS (used in prompt conditioning)
VOICE_PROMPTS = {
    "hi": {
        TTSVoice.FEMALE_1: "Aditi speaks in a clear and friendly Hindi tone, with natural pauses and warm expression.",
        TTSVoice.FEMALE_2: "Priya delivers text in a professional and calm Hindi voice, suitable for announcements.",
        TTSVoice.MALE_1: "Rajesh speaks in a deep, authoritative Hindi voice with clear diction.",
        TTSVoice.MALE_2: "Arjun uses a conversational and approachable Hindi speaking style.",
        TTSVoice.NEUTRAL: "A neutral, clear Hindi voice with standard pronunciation and moderate pace.",
    },
    "ta": {
        TTSVoice.FEMALE_1: "Kavitha speaks in a clear and expressive Tamil tone.",
        TTSVoice.MALE_1: "Karthik delivers text in a warm and natural Tamil voice.",
        TTSVoice.NEUTRAL: "A neutral, clear Tamil voice with standard pronunciation.",
    },
    # Add more languages as community contributes voice prompts
}

DEFAULT_VOICE_PROMPT = "A clear, natural voice with standard pronunciation and moderate pace."


class BaseTTSBackend(ABC):
    name: str
    supported_langs: set[str]

    @abstractmethod
    async def synthesize(self, text: str, lang: str, voice: str, **kwargs) -> bytes:
        """Returns audio bytes."""
        ...


class IndicParlerTTSBackend(BaseTTSBackend):
    """
    AI4Bharat Indic Parler-TTS — 22 Indian languages, 44 speakers.
    Uses Parler-TTS architecture with voice conditioning via text prompts.
    Model: ai4bharat/indic-parler-tts
    """

    name = "indic_parler"
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
        "sd",
        "brx",
        "sat",
    }

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await asyncio.get_event_loop().run_in_executor(None, self._load)

    def _load(self) -> None:
        try:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            logger.info("Loading Indic Parler-TTS: %s", settings.INDIC_PARLER_MODEL)
            device = _resolve_device()

            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.INDIC_PARLER_MODEL,
                cache_dir=settings.MODEL_CACHE_DIR,
            )
            self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                settings.INDIC_PARLER_MODEL,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                cache_dir=settings.MODEL_CACHE_DIR,
            ).to(device)
            self._loaded = True
            logger.info("Indic Parler-TTS loaded on %s", device)
        except ImportError:
            raise RuntimeError(
                "parler_tts not installed. Run: pip install git+https://github.com/huggingface/parler-tts"
            )

    async def synthesize(self, text: str, lang: str, voice: str = "female_1", **kwargs) -> bytes:
        await self._ensure_loaded()

        voice_enum = TTSVoice(voice) if voice in TTSVoice._value2member_map_ else TTSVoice.NEUTRAL
        lang_prompts = VOICE_PROMPTS.get(lang, {})
        voice_prompt = lang_prompts.get(voice_enum, DEFAULT_VOICE_PROMPT)

        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_inference, text, voice_prompt, lang, kwargs.get("speed", 1.0)
        )

    def _run_inference(self, text: str, voice_prompt: str, lang: str, speed: float) -> bytes:
        import soundfile as sf
        import torch

        device = next(self._model.parameters()).device

        # Tokenize description (voice conditioning) and input text
        desc_input = self._tokenizer(voice_prompt, return_tensors="pt").to(device)
        text_input = self._tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_input.input_ids,
                prompt_input_ids=text_input.input_ids,
                attention_mask=desc_input.attention_mask,
                prompt_attention_mask=text_input.attention_mask,
            )

        audio = generation.cpu().numpy().squeeze()
        sr = self._model.config.sampling_rate

        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()


class BhashiniTTSBackend(BaseTTSBackend):
    """
    BHASHINI TTS — HiFi-GAN neural vocoder for Indian languages.
    Supports emotion/style tags: Neutral, Happy, Sad, News, Conversational, etc.
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
    }
    COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    async def synthesize(self, text: str, lang: str, voice: str = "female_1", **kwargs) -> bytes:
        if not settings.is_bhashini_configured:
            raise RuntimeError("BHASHINI credentials not configured.")

        gender = "female" if "female" in voice else "male"

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {"sourceLanguage": lang},
                        "gender": gender,
                        "samplingRate": 22050,
                    },
                }
            ],
            "inputData": {"input": [{"source": text}]},
        }

        headers = {
            "userID": settings.BHASHINI_USER_ID,
            "ulcaApiKey": settings.BHASHINI_API_KEY,
            "Authorization": settings.BHASHINI_API_KEY,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.COMPUTE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        audio_b64 = data["pipelineResponse"][0]["audio"][0]["audioContent"]
        return base64.b64decode(audio_b64)


class GTTSBackend(BaseTTSBackend):
    """Lightweight Google TTS fallback. No GPU needed. Supports many Indian languages."""

    name = "gtts"
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
        "ur",
        "en",
    }

    async def synthesize(self, text: str, lang: str, voice: str = "female_1", **kwargs) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(None, self._generate, text, lang)

    def _generate(self, text: str, lang: str) -> bytes:
        from gtts import gTTS

        buf = io.BytesIO()
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()  # returns MP3 bytes


class TTSDispatcher:
    """Route TTS requests to the best available backend."""

    def __init__(self, default_backend: str = "indic_parler") -> None:
        self._default = default_backend
        self._backends: dict[str, BaseTTSBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            self._backends["indic_parler"] = IndicParlerTTSBackend()
        except Exception as e:
            logger.warning("Indic Parler-TTS unavailable: %s", e)

        if settings.is_bhashini_configured:
            self._backends["bhashini"] = BhashiniTTSBackend()

        try:
            self._backends["gtts"] = GTTSBackend()
        except Exception as e:
            logger.warning("gTTS unavailable: %s", e)

    async def synthesize(self, request: TTSRequest) -> TTSResponse:
        start = time.perf_counter()
        backend_name = request.backend if request.backend != "auto" else self._default
        backend = self._backends.get(backend_name)

        if not backend or request.lang not in backend.supported_langs:
            # Fallback
            for _fb_name, fb in self._backends.items():
                if request.lang in fb.supported_langs:
                    backend = fb
                    break
            else:
                raise RuntimeError(f"No TTS backend supports language: {request.lang}")

        try:
            audio_bytes = await backend.synthesize(
                text=request.text,
                lang=request.lang,
                voice=request.voice.value,
                speed=request.speed,
            )
        except Exception as e:
            logger.warning("TTS backend %s failed: %s. Trying fallback.", backend.name, e)
            # Try gtts fallback
            fallback = self._backends.get("gtts")
            if fallback and fallback is not backend:
                audio_bytes = await fallback.synthesize(
                    text=request.text,
                    lang=request.lang,
                    voice=request.voice.value,
                    speed=request.speed,
                )
                backend = fallback
            else:
                raise

        # Convert format if needed (gtts returns MP3 already)
        if request.format != AudioFormat.WAV and backend.name != "gtts":
            audio_bytes = await _convert_audio(audio_bytes, request.format)

        duration_sec = _estimate_duration(audio_bytes, request.format)
        elapsed_ms = round((time.perf_counter() - start) * 1000)

        # Small outputs inline, large as URL
        audio_b64 = None
        audio_url = None
        if len(audio_bytes) < 500 * 1024:  # < 500KB
            audio_b64 = base64.b64encode(audio_bytes).decode()
        else:
            audio_url = await _upload_audio(audio_bytes, request.format)

        return TTSResponse(
            audio_url=audio_url,
            audio_base64=audio_b64,
            duration_sec=duration_sec,
            format=request.format.value,
            backend_used=backend.name,
            processing_ms=elapsed_ms,
        )


async def _convert_audio(wav_bytes: bytes, fmt: AudioFormat) -> bytes:
    """Convert WAV to MP3/OPUS/OGG via pydub."""
    if fmt == AudioFormat.WAV:
        return wav_bytes

    def _convert():
        try:
            from pydub import AudioSegment
        except ImportError:
            logger.warning("pydub not installed; returning WAV instead of %s", fmt)
            return wav_bytes

        audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        buf = io.BytesIO()
        audio.export(buf, format=fmt.value)
        return buf.getvalue()

    return await asyncio.get_event_loop().run_in_executor(None, _convert)


def _estimate_duration(audio_bytes: bytes, fmt: AudioFormat) -> float:
    try:
        if fmt == AudioFormat.WAV:
            import wave

            with wave.open(io.BytesIO(audio_bytes)) as wf:
                return wf.getnframes() / wf.getframerate()
    except Exception:
        pass
    # Rough estimate: ~16KB/sec for typical Indic TTS output
    return len(audio_bytes) / 16384


async def _upload_audio(audio_bytes: bytes, fmt: AudioFormat) -> str | None:
    """Upload to object storage and return presigned URL."""
    if not settings.OBJECT_STORAGE_URL:
        return None
    # TODO: implement S3-compatible upload
    return None


def _resolve_device() -> str:
    try:
        import torch

        if settings.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return "cpu"
