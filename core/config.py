"""
Core configuration — loaded from environment variables.
Uses pydantic-settings for type-safe, validated config.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "BharatVoiceAI"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    WARMUP_MODELS: bool = True

    # API Security
    VAI_JWT_SECRET: str = Field(default="CHANGE_ME_IN_PRODUCTION", min_length=32)
    VAI_API_KEY_SALT: str = Field(default="CHANGE_ME_IN_PRODUCTION", min_length=16)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    # Comma-separated list of static API keys for admin/testing
    STATIC_API_KEYS: str = ""

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    # Rate Limiting
    RATE_LIMIT_DEFAULT_RPM: int = 60  # requests per minute per API key
    RATE_LIMIT_ASR_RPM: int = 30  # ASR is compute-heavy
    RATE_LIMIT_TTS_RPM: int = 60
    RATE_LIMIT_MT_RPM: int = 120
    RATE_LIMIT_BURST_MULTIPLIER: float = 1.5

    # Database (leave empty to run without a database)
    VAI_DB_URL: str = ""
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Redis (rate limiting & caching)
    VAI_REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour for TTS / MT cache

    # BHASHINI
    BHASHINI_USER_ID: str = ""
    BHASHINI_API_KEY: str = ""
    BHASHINI_PIPELINE_ID: str = ""
    BHASHINI_BASE_URL: str = "https://dhruva-api.bhashini.gov.in"
    BHASHINI_SEARCH_URL: str = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"

    # Sarvam AI
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"

    # Model Backends
    VAI_ASR_BACKEND: Literal["indicwhisper", "bhashini", "wav2vec2", "local"] = "indicwhisper"
    VAI_MT_BACKEND: Literal["indictrans2", "bhashini", "marian", "sarvam"] = "indictrans2"
    VAI_TTS_BACKEND: Literal["indic_parler", "bhashini", "local"] = "indic_parler"
    VAI_LID_BACKEND: Literal["smart", "indiclid", "bhashini", "local"] = "smart"
    VAI_NLU_BACKEND: Literal["rule", "llm", "rule+llm"] = "rule+llm"

    # Model Paths (for local/self-hosted models)
    MODEL_CACHE_DIR: str = "/tmp/vai_models"
    INDICTRANS2_MODEL: str = "ai4bharat/indictrans2-en-indic-1B"
    INDICWHISPER_MODEL: str = "openai/whisper-large-v3"
    INDIC_PARLER_MODEL: str = "ai4bharat/indic-parler-tts"
    INDICLID_MODEL: str = "ai4bharat/IndicLID"
    SARVAM_LLM_MODEL: str = "sarvam-m"

    # Inference
    DEVICE: str = "auto"  # auto | cpu | cuda | cuda:0
    BATCH_SIZE: int = 8
    MAX_AUDIO_DURATION_SEC: int = 120  # max audio clip for ASR
    MAX_TEXT_CHARS: int = 5000  # max text for MT/TTS/NLU
    ASR_TARGET_SAMPLE_RATE: int = 16000

    # Privacy & DPDP 2023
    VAI_AUDIO_LOGGING: bool = False  # NEVER log raw audio by default
    VAI_PII_MASKING: bool = True  # mask phone/aadhaar/email in logs
    VAI_DATA_RETENTION_DAYS: int = 0  # 0 = delete immediately post-processing
    CONSENT_REQUIRED: bool = True  # require explicit consent header for audio

    # Storage
    AUDIO_UPLOAD_MAX_MB: int = 25
    OBJECT_STORAGE_URL: str = ""  # S3-compatible URL for audio uploads
    OBJECT_STORAGE_BUCKET: str = "vai-audio"

    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    SENTRY_DSN: str = ""
    OTEL_ENDPOINT: str = ""  # OpenTelemetry collector endpoint

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @property
    def static_api_key_list(self) -> list[str]:
        return [k.strip() for k in self.STATIC_API_KEYS.split(",") if k.strip()]

    @property
    def is_bhashini_configured(self) -> bool:
        return bool(self.BHASHINI_USER_ID and self.BHASHINI_API_KEY)

    @property
    def is_sarvam_configured(self) -> bool:
        return bool(self.SARVAM_API_KEY)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
