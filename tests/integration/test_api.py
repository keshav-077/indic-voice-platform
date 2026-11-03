"""
Integration Tests — BharatVoiceAI
============================
Tests full request/response cycles with mocked external backends.
Run: pytest tests/integration/ -v

These tests mock BHASHINI/Sarvam API calls using respx,
so they work in CI without real credentials.
"""

import base64
import struct

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

# Fixtures


@pytest.fixture(scope="module")
def app():
    import os

    os.environ.update(
        {
            "VAI_JWT_SECRET": "integration_test_secret_32_chars_long",
            "VAI_API_KEY_SALT": "integration_salt16",
            "STATIC_API_KEYS": "integration-test-key",
            "WARMUP_MODELS": "false",
            "CONSENT_REQUIRED": "false",
            "VAI_ASR_BACKEND": "bhashini",
            "VAI_MT_BACKEND": "bhashini",
            "VAI_TTS_BACKEND": "bhashini",
            "VAI_LID_BACKEND": "indiclid",
            "VAI_NLU_BACKEND": "rule",
            "BHASHINI_USER_ID": "test_user_id",
            "BHASHINI_API_KEY": "test_api_key",
            "VAI_DB_URL": "sqlite+aiosqlite:///./integration_test.db",
        }
    )
    from api.main import app

    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth():
    return {"X-API-Key": "integration-test-key"}


@pytest.fixture
def minimal_wav_b64():
    """Minimal valid WAV file."""
    num_samples = 16000  # 1 second at 16kHz
    sample_rate = 16000
    bits, channels = 16, 1
    data_size = num_samples * channels * (bits // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * (bits // 8),
        channels * (bits // 8),
        bits,
        b"data",
        data_size,
    )
    wav_bytes = header + (b"\x00" * data_size)
    return base64.b64encode(wav_bytes).decode()


# BHASHINI Mock Responses

BHASHINI_ASR_RESPONSE = {
    "pipelineResponse": [
        {
            "output": [
                {
                    "source": "नमस्ते आप कैसे हैं",
                    "score": 0.92,
                }
            ]
        }
    ]
}

BHASHINI_MT_RESPONSE = {
    "pipelineResponse": [
        {
            "output": [
                {
                    "target": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
                }
            ]
        }
    ]
}

BHASHINI_TTS_RESPONSE = {
    "pipelineResponse": [
        {
            "audio": [
                {
                    "audioContent": base64.b64encode(b"\x00" * 1000).decode(),
                }
            ]
        }
    ]
}

BHASHINI_LID_RESPONSE = {
    "pipelineResponse": [
        {
            "output": [
                {
                    "langCode": "hi",
                }
            ]
        }
    ]
}

BHASHINI_COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"


# ASR Integration Tests


class TestASRIntegration:
    """Full ASR request cycle with mocked BHASHINI."""

    @respx.mock
    def test_asr_json_request(self, client, auth, minimal_wav_b64):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_ASR_RESPONSE))

        resp = client.post(
            "/v1/asr",
            headers=auth,
            json={
                "audio_base64": minimal_wav_b64,
                "lang": "hi",
                "consent_given": True,
                "backend": "bhashini",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "transcript" in data
        assert data["transcript"] == "नमस्ते आप कैसे हैं"
        assert "confidence" in data
        assert data["confidence"] > 0
        assert "processing_ms" in data
        assert "backend_used" in data

    @respx.mock
    def test_asr_missing_consent_rejected(self, client, auth, minimal_wav_b64):
        import os

        os.environ["CONSENT_REQUIRED"] = "true"

        resp = client.post(
            "/v1/asr",
            headers=auth,
            json={
                "audio_base64": minimal_wav_b64,
                "lang": "hi",
                "consent_given": False,
            },
        )
        # Restore
        os.environ["CONSENT_REQUIRED"] = "false"
        assert resp.status_code == 400
        assert "consent" in resp.json()["detail"]["error"]

    def test_asr_no_audio_rejected(self, client, auth):
        resp = client.post(
            "/v1/asr",
            headers=auth,
            json={"lang": "hi"},
        )
        assert resp.status_code == 422  # Validation error

    @respx.mock
    def test_asr_word_timestamps(self, client, auth, minimal_wav_b64):
        # BHASHINI doesn't return timestamps; should still work with empty words list
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_ASR_RESPONSE))
        resp = client.post(
            "/v1/asr",
            headers=auth,
            json={
                "audio_base64": minimal_wav_b64,
                "lang": "hi",
                "return_word_timestamps": True,
                "consent_given": True,
                "backend": "bhashini",
            },
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["words"], list)

    @respx.mock
    def test_asr_backend_error_handled(self, client, auth, minimal_wav_b64):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(503, json={"error": "Service Unavailable"}))
        resp = client.post(
            "/v1/asr",
            headers=auth,
            json={
                "audio_base64": minimal_wav_b64,
                "lang": "hi",
                "consent_given": True,
                "backend": "bhashini",
            },
        )
        # Should return 503 or fall back
        assert resp.status_code in (503, 200)


# MT Integration Tests


class TestMTIntegration:
    @respx.mock
    def test_translate_hi_to_ta(self, client, auth):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_MT_RESPONSE))

        resp = client.post(
            "/v1/translate",
            headers=auth,
            json={"text": "नमस्ते", "src_lang": "hi", "tgt_lang": "ta", "backend": "bhashini"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "translation" in data
        assert data["src_lang"] == "hi"
        assert data["tgt_lang"] == "ta"
        assert "confidence" in data
        assert "processing_ms" in data

    @respx.mock
    def test_batch_translate(self, client, auth):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_MT_RESPONSE))

        resp = client.post(
            "/v1/translate/batch",
            headers=auth,
            json={
                "items": [
                    {"text": "नमस्ते", "src_lang": "hi", "tgt_lang": "ta", "backend": "bhashini"},
                    {"text": "धन्यवाद", "src_lang": "hi", "tgt_lang": "ta", "backend": "bhashini"},
                ]
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert "total_processing_ms" in data

    def test_same_language_rejected(self, client, auth):
        resp = client.post(
            "/v1/translate",
            headers=auth,
            json={"text": "hello", "src_lang": "hi", "tgt_lang": "hi"},
        )
        assert resp.status_code == 422

    def test_text_too_long_rejected(self, client, auth):
        resp = client.post(
            "/v1/translate",
            headers=auth,
            json={"text": "x" * 6000, "src_lang": "en", "tgt_lang": "hi"},
        )
        assert resp.status_code == 422

    @respx.mock
    def test_batch_limit_enforced(self, client, auth):
        resp = client.post(
            "/v1/translate/batch",
            headers=auth,
            json={
                "items": [{"text": "test", "src_lang": "en", "tgt_lang": "hi"}] * 51  # 51 > max 50
            },
        )
        assert resp.status_code == 422


# TTS Integration Tests


class TestTTSIntegration:
    @respx.mock
    def test_tts_hindi(self, client, auth):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_TTS_RESPONSE))

        resp = client.post(
            "/v1/tts",
            headers=auth,
            json={
                "text": "नमस्ते",
                "lang": "hi",
                "voice": "female_1",
                "format": "mp3",
                "backend": "bhashini",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "duration_sec" in data
        assert "format" in data
        assert "backend_used" in data
        # Should have either audio_base64 or audio_url
        assert data.get("audio_base64") or data.get("audio_url")

    @respx.mock
    def test_tts_stream_endpoint(self, client, auth):
        respx.post(BHASHINI_COMPUTE_URL).mock(return_value=httpx.Response(200, json=BHASHINI_TTS_RESPONSE))

        resp = client.post(
            "/v1/tts/stream",
            headers=auth,
            json={"text": "hello", "lang": "hi", "backend": "bhashini"},
        )
        # Should return audio bytes or 503 if backend unavailable
        assert resp.status_code in (200, 503)

    def test_tts_text_too_long(self, client, auth):
        resp = client.post(
            "/v1/tts",
            headers=auth,
            json={"text": "x" * 2001, "lang": "hi"},
        )
        assert resp.status_code == 422

    def test_tts_speed_bounds(self, client, auth):
        resp = client.post(
            "/v1/tts",
            headers=auth,
            json={"text": "test", "lang": "hi", "speed": 3.0},  # > 2.0 max
        )
        assert resp.status_code == 422


# LID Integration Tests


class TestLIDIntegration:
    """LID tests use heuristic backend (no external calls needed)."""

    def test_lid_devanagari(self, client, auth):
        resp = client.post(
            "/v1/lid",
            headers=auth,
            json={"text": "नमस्ते भारत", "top_k": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["lang"] == "hi"
        assert data["confidence"] > 0.5
        assert "script" in data
        assert "is_code_mixed" in data

    def test_lid_tamil(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={"text": "வணக்கம்"})
        assert resp.status_code == 200
        assert resp.json()["lang"] == "ta"

    def test_lid_telugu(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={"text": "నమస్కారం"})
        assert resp.status_code == 200
        assert resp.json()["lang"] == "te"

    def test_lid_hinglish_code_mix(self, client, auth):
        resp = client.post(
            "/v1/lid",
            headers=auth,
            json={"text": "mera naam Rahul hai aur main Delhi mein rehta hoon"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Might detect as Hindi or English depending on backend
        assert data["lang"] in ("hi", "en")
        # Code-mix should be detected
        assert data["is_code_mixed"] is True

    def test_lid_top_k(self, client, auth):
        resp = client.post(
            "/v1/lid",
            headers=auth,
            json={"text": "नमस्ते hello world", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "candidates" in data

    def test_lid_no_input_rejected(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={})
        assert resp.status_code == 422


# NLU Integration Tests


class TestNLUIntegration:
    """NLU tests use rule backend (no external LLM calls)."""

    def test_nlu_view_bill_hindi(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "mera bill dikhao", "lang": "hi", "domain": "billing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "view_bill"
        assert data["confidence"] >= 0.8

    def test_nlu_greet_english(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "Hello, how are you?", "lang": "en"},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "greet"

    def test_nlu_with_response_generation(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={
                "text": "namaste",
                "lang": "hi",
                "generate_response": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "greet"
        assert data["response_text"] is not None
        assert len(data["response_text"]) > 0

    def test_nlu_without_response_generation(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "namaste", "lang": "hi", "generate_response": False},
        )
        assert resp.status_code == 200
        assert resp.json()["response_text"] is None

    def test_nlu_entity_extraction(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "₹500 ka bill pay karo", "lang": "hi", "domain": "billing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "pay_bill"
        entities = {e["type"]: e["value"] for e in data["entities"]}
        assert "amount" in entities
        assert "500" in entities["amount"]

    def test_nlu_unknown_intent(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "quantum entanglement ki theory batao", "lang": "hi"},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "unknown"

    def test_nlu_agriculture_domain(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={
                "text": "gehu ka mandi bhav batao",
                "lang": "hi",
                "domain": "agriculture",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "check_crop_price"

    def test_nlu_healthcare_emergency(self, client, auth):
        resp = client.post(
            "/v1/nlu",
            headers=auth,
            json={"text": "ambulance chahiye", "lang": "hi", "domain": "healthcare"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "emergency_services"
        # Emergency response should mention 108
        if data.get("response_text"):
            assert "108" in data["response_text"]


# Admin Endpoints


class TestAdminEndpoints:
    def test_issue_token(self, client, auth):
        resp = client.post(
            "/admin/token",
            headers=auth,
            json={"user_id": "test-user", "scopes": ["api"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in_seconds"] > 0

    def test_list_models(self, client, auth):
        resp = client.get("/admin/models", headers=auth)
        assert resp.status_code == 200
        data = resp.json()
        assert "asr" in data
        assert "mt" in data
        assert "tts" in data
        assert "lid" in data
        assert "nlu" in data

    def test_config_endpoint(self, client, auth):
        resp = client.get("/admin/config", headers=auth)
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_logging"] is False  # DPDP default
        assert data["pii_masking"] is True
        assert data["data_retention_days"] == 0


# Rate Limiting Integration


class TestRateLimitIntegration:
    def test_rate_limit_headers_on_lid(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={"text": "test"})
        assert resp.status_code == 200
        assert "X-RateLimit-Limit" in resp.headers
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining >= 0

    def test_process_time_header(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={"text": "test"})
        assert "X-Process-Time-Ms" in resp.headers
        ms = float(resp.headers["X-Process-Time-Ms"])
        assert ms > 0

    def test_request_id_header(self, client, auth):
        resp = client.post("/v1/lid", headers=auth, json={"text": "test"})
        assert "X-Request-ID" in resp.headers


# JWT Token Flow


class TestJWTFlow:
    def test_jwt_token_works(self, client, auth):
        # Issue a JWT
        token_resp = client.post(
            "/admin/token",
            headers=auth,
            json={"user_id": "jwt-test-user"},
        )
        assert token_resp.status_code == 200
        jwt_token = token_resp.json()["access_token"]

        # Use JWT to access endpoint
        resp = client.post(
            "/v1/lid",
            headers={"Authorization": f"Bearer {jwt_token}"},
            json={"text": "नमस्ते"},
        )
        assert resp.status_code == 200

    def test_invalid_jwt_rejected(self, client):
        resp = client.post(
            "/v1/lid",
            headers={"Authorization": "Bearer invalid.jwt.token"},
            json={"text": "test"},
        )
        assert resp.status_code == 401

    def test_no_auth_rejected(self, client):
        resp = client.post("/v1/lid", json={"text": "test"})
        assert resp.status_code == 401
