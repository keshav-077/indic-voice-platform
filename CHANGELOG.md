# Changelog

All notable changes to BharatVoiceAI will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [Unreleased]

### Planned
- GGUF model support for Sarvam 30B/105B (blocked on upstream vLLM integration)
- Streaming MT endpoint (Server-Sent Events)
- OCR endpoint for Indic scripts (Phase 2)
- WhatsApp Business API integration example

---

## [0.1.0] — 2026-04-04

### Added
- **ASR endpoint** (`POST /v1/asr`, `POST /v1/asr/upload`, `WS /v1/asr/stream`)
  - Backends: IndicWhisper (AI4Bharat), BHASHINI, Wav2Vec2-Indic
  - 12 languages, word-level timestamps, streaming WebSocket
- **Machine Translation** (`POST /v1/translate`, `POST /v1/translate/batch`)
  - Backends: IndicTrans2-1B, BHASHINI NMT, Sarvam Translate API
  - All 22 Indic languages, direct Indic↔Indic translation (no English pivot)
  - Batch endpoint (up to 50 sentences, grouped by language pair)
- **Text-to-Speech** (`POST /v1/tts`, `POST /v1/tts/stream`)
  - Backends: Indic Parler-TTS (44 voices), BHASHINI TTS
  - 22 languages, audio streaming, format conversion (WAV/MP3/OPUS)
- **Language Identification** (`POST /v1/lid`)
  - Backends: IndicLID, heuristic Unicode-block, BHASHINI
  - 47 languages, code-mix detection (Hinglish etc.), audio + text input
- **NLU / Intent** (`POST /v1/nlu`)
  - Hybrid rule engine + Sarvam-M LLM fallback
  - 7 built-in intents (view_bill, check_balance, greet, etc.)
  - Extensible via YAML intent files
  - Entity extraction (amount, phone, date, account_id)
- **Auth**: JWT Bearer tokens + X-API-Key header
- **Rate limiting**: Redis sliding-window, per-endpoint limits
- **DPDP 2023 compliance**:
  - `consent_given` required for audio processing
  - `VAI_AUDIO_LOGGING=false` by default
  - PII masking in all logs (Aadhaar, phone, email)
  - `VAI_DATA_RETENTION_DAYS=0` by default
- **Health checks**: `/health`, `/health/live`, `/health/ready`, `/metrics`
- **Docker Compose** stack (API + PostgreSQL + Redis + Prometheus + Grafana + Nginx)
- **Kubernetes manifests** with HPA, PDB, NetworkPolicy, Ingress
- **CI/CD**: GitHub Actions (lint, unit tests, Docker build, GHCR publish)
- **Evaluation scripts**: `eval_asr.py` (WER on Vistaar/FLEURS), `eval_mt.py` (BLEU on FLORES-200/IN22)
- **Documentation**: BHASHINI setup guide, DPDP compliance guide, model cards
- **Contributing guide** with language/backend/intent addition walkthroughs

### Technical Stack
- FastAPI 0.110+ with async throughout
- Pydantic v2 for all schemas
- Redis sliding-window rate limiting
- PostgreSQL + asyncpg connection pool
- Prometheus + Grafana monitoring
- Multi-stage Docker build (non-root user)

---

[Unreleased]: https://github.com/samirdutta/bharatvoiceai/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/samirdutta/bharatvoiceai/releases/tag/v0.1.0
