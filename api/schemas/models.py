"""
API Schemas — Pydantic v2 models for all request/response types.
These form the contract of the BharatVoiceAI surface.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

# Language codes

SUPPORTED_LANGS = {
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
    "sa",
    "mai",
    "kok",
    "mni",
    "bho",
    "doi",
    "ks",
    "ne",
    "sd",
    "sat",
    "en",  # English included for translate
}

LanguageCode = Annotated[
    str,
    Field(
        min_length=2,
        max_length=5,
        pattern=r"^[a-z]{2,3}(_[A-Z]{4})?$",
        examples=["hi", "ta", "en", "bn"],
    ),
]


class AudioFormat(StrEnum):
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"
    OGG = "ogg"
    FLAC = "flac"


class TTSVoice(StrEnum):
    # Sarvam / Indic Parler voices
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    MALE_1 = "male_1"
    MALE_2 = "male_2"
    NEUTRAL = "neutral"


class NLUDomain(StrEnum):
    GENERAL = "general"
    BILLING = "billing"
    BANKING = "banking"
    HEALTHCARE = "healthcare"
    AGRICULTURE = "agriculture"
    ECOMMERCE = "ecommerce"
    EDUCATION = "education"
    GOVERNMENT = "government"


# Common


class APIError(BaseModel):
    error: str = Field(examples=["invalid_language"])
    message: str = Field(examples=["Language code 'xyz' is not supported."])
    request_id: str | None = None


class WordTimestamp(BaseModel):
    text: str
    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    conf: float = Field(ge=0, le=1, description="Confidence score 0-1")


# ASR


class ASRRequest(BaseModel):
    """Speech-to-Text request."""

    audio_url: HttpUrl | None = Field(
        default=None,
        description="Publicly accessible URL to an audio file (WAV/MP3/OPUS). Mutually exclusive with audio_base64.",
        examples=["https://example.com/audio.wav"],
    )
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio bytes. Max 25MB. Mutually exclusive with audio_url.",
    )
    lang: LanguageCode | None = Field(
        default=None,
        description="BCP-47 language code. If omitted, Language ID is auto-run.",
        examples=["hi", "ta"],
    )
    backend: Literal["indicwhisper", "bhashini", "wav2vec2", "auto"] = "auto"
    return_word_timestamps: bool = Field(
        default=False,
        description="If true, return per-word start/end timestamps.",
    )
    audio_format: AudioFormat = AudioFormat.WAV
    consent_given: bool = Field(
        default=False,
        description="Explicit user consent for processing this audio (required by DPDP 2023).",
    )

    @field_validator("audio_url", "audio_base64")
    @classmethod
    def check_audio_source(cls, v, info):
        # Called per field; cross-field check in model_validator
        return v

    def model_post_init(self, __context) -> None:
        if not self.audio_url and not self.audio_base64:
            raise ValueError("Either audio_url or audio_base64 must be provided.")
        if self.audio_url and self.audio_base64:
            raise ValueError("Provide either audio_url or audio_base64, not both.")


class ASRResponse(BaseModel):
    transcript: str = Field(examples=["नमस्ते, आप कैसे हैं?"])
    confidence: float = Field(ge=0, le=1, examples=[0.94])
    lang_detected: str = Field(examples=["hi"])
    words: list[WordTimestamp] = Field(default_factory=list)
    backend_used: str = Field(examples=["indicwhisper"])
    processing_ms: int = Field(examples=[210])


# MT


class MTRequest(BaseModel):
    """Machine Translation request."""

    text: str = Field(
        min_length=1,
        max_length=5000,
        examples=["Hello, how are you?"],
    )
    src_lang: LanguageCode = Field(examples=["en"])
    tgt_lang: LanguageCode = Field(examples=["hi"])
    backend: Literal["indictrans2", "bhashini", "sarvam", "auto"] = "auto"
    domain: NLUDomain = NLUDomain.GENERAL

    @field_validator("tgt_lang")
    @classmethod
    def src_tgt_differ(cls, v, info):
        if info.data.get("src_lang") == v:
            raise ValueError("src_lang and tgt_lang must differ.")
        return v


class MTResponse(BaseModel):
    translation: str = Field(examples=["नमस्ते, आप कैसे हैं?"])
    src_lang: str
    tgt_lang: str
    confidence: float = Field(ge=0, le=1, examples=[0.91])
    backend_used: str = Field(examples=["indictrans2"])
    processing_ms: int


class MTBatchRequest(BaseModel):
    """Batch translation — up to 50 sentences."""

    items: list[MTRequest] = Field(min_length=1, max_length=50)


class MTBatchResponse(BaseModel):
    results: list[MTResponse]
    total_processing_ms: int


# TTS


class TTSRequest(BaseModel):
    """Text-to-Speech request."""

    text: str = Field(min_length=1, max_length=2000, examples=["आपका स्वागत है"])
    lang: LanguageCode = Field(examples=["hi"])
    voice: TTSVoice = TTSVoice.FEMALE_1
    format: AudioFormat = AudioFormat.MP3
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech rate multiplier")
    backend: Literal["indic_parler", "bhashini", "auto"] = "auto"
    # SSML subset (only speak, break, prosody supported)
    use_ssml: bool = False


class TTSResponse(BaseModel):
    audio_url: str | None = Field(
        default=None,
        description="URL to download synthesized audio (valid for 1 hour).",
    )
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded audio for small outputs (<500KB).",
    )
    duration_sec: float
    format: str
    backend_used: str
    processing_ms: int


# LID


class LIDRequest(BaseModel):
    """Language Identification request. Supports text or audio."""

    text: str | None = Field(
        default=None,
        max_length=1000,
        examples=["mera naam Rahul hai"],
    )
    audio_url: HttpUrl | None = None
    audio_base64: str | None = None
    top_k: int = Field(default=1, ge=1, le=5, description="Return top-k language guesses")

    def model_post_init(self, __context) -> None:
        if not self.text and not self.audio_url and not self.audio_base64:
            raise ValueError("Provide text, audio_url, or audio_base64.")


class LIDCandidate(BaseModel):
    lang: str
    confidence: float = Field(ge=0, le=1)
    script: str | None = None


class LIDResponse(BaseModel):
    lang: str = Field(description="Best detected language code", examples=["hi"])
    script: str | None = Field(default=None, examples=["Devanagari"])
    confidence: float = Field(ge=0, le=1)
    is_code_mixed: bool = Field(default=False)
    code_mix_langs: list[str] = Field(default_factory=list)
    candidates: list[LIDCandidate] = Field(default_factory=list)
    backend_used: str
    processing_ms: int


# NLU / Intent


class NLUEntity(BaseModel):
    type: str = Field(examples=["account_id", "date", "amount"])
    value: str
    confidence: float = Field(ge=0, le=1)


class NLURequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000, examples=["मेरा बिल दिखाओ"])
    lang: LanguageCode = Field(examples=["hi"])
    domain: NLUDomain = NLUDomain.GENERAL
    context: dict | None = Field(
        default=None,
        description="Optional session context (user_id, previous_intent, etc.)",
    )
    generate_response: bool = Field(
        default=False,
        description="If true, generate a natural-language response text (uses LLM).",
    )


class NLUResponse(BaseModel):
    intent: str = Field(examples=["view_bill"])
    confidence: float = Field(ge=0, le=1)
    entities: list[NLUEntity] = Field(default_factory=list)
    response_text: str | None = Field(
        default=None,
        description="LLM-generated response, if generate_response=true.",
    )
    backend_used: str
    processing_ms: int


# Transliteration


class TranslitRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, examples=["namaste"])
    src_script: Literal[
        "Latin",
        "Devanagari",
        "Tamil",
        "Telugu",
        "Bengali",
        "Gujarati",
        "Kannada",
        "Malayalam",
        "Gurmukhi",
        "Odia",
    ] = "Latin"
    tgt_script: Literal[
        "Latin",
        "Devanagari",
        "Tamil",
        "Telugu",
        "Bengali",
        "Gujarati",
        "Kannada",
        "Malayalam",
        "Gurmukhi",
        "Odia",
    ] = "Devanagari"
    lang: LanguageCode = Field(examples=["hi"])


class TranslitResponse(BaseModel):
    transliteration: str = Field(examples=["नमस्ते"])
    src_script: str
    tgt_script: str
    processing_ms: int


# Health


class ServiceStatus(BaseModel):
    status: Literal["ok", "degraded", "down"]
    backend: str
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    services: dict[str, ServiceStatus]
    uptime_sec: float
