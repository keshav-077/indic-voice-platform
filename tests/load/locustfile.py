"""
Load Testing — Locust
=====================
Simulates realistic traffic patterns for BharatVoiceAI.

Usage:
    # Interactive web UI at http://localhost:8089
    locust -f tests/load/locustfile.py

    # Headless (CI)
    locust -f tests/load/locustfile.py \
        --headless -u 50 -r 5 --run-time 60s \
        --host http://localhost:8000

    # Target specific endpoints only
    locust -f tests/load/locustfile.py --tags mt lid

Scenarios:
    - ASRUser: Simulates voice input workflows
    - MTUser: Simulates translation-heavy app (content localization)
    - LIDUser: Simulates auto-detect + route flows
    - MixedUser: Realistic mixed workload
"""

import base64
import random
import struct

from locust import HttpUser, between, tag, task

# Test fixtures


def make_wav_b64(duration_sec: float = 1.0) -> str:
    """Generate a minimal WAV file for ASR testing."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
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


# Pre-generate test audio to avoid CPU overhead during load test
SHORT_AUDIO = make_wav_b64(0.5)
MEDIUM_AUDIO = make_wav_b64(2.0)

SAMPLE_TEXTS = {
    "hi": [
        "नमस्ते, आप कैसे हैं?",
        "मेरा बिल दिखाओ",
        "मौसम कैसा है आज?",
        "धन्यवाद आपकी मदद के लिए",
        "मुझे डॉक्टर से मिलना है",
    ],
    "ta": [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "என் பில் காட்டுங்கள்",
        "இன்று வானிலை எப்படி?",
    ],
    "te": [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "నా బిల్లు చూపించు",
    ],
    "en": [
        "Hello, how are you?",
        "Show me my bill",
        "What is the weather today?",
        "Thank you for your help",
    ],
}

LANG_PAIRS = [
    ("en", "hi"),
    ("en", "ta"),
    ("en", "te"),
    ("en", "bn"),
    ("hi", "en"),
    ("hi", "ta"),
    ("ta", "en"),
]


class BaseVAIUser(HttpUser):
    """Base class with auth headers."""

    abstract = True

    def on_start(self):
        self.headers = {"X-API-Key": "test-load-key"}

    def _random_text(self, lang: str = "hi") -> str:
        texts = SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS["en"])
        return random.choice(texts)


class ASRUser(BaseVAIUser):
    """Simulates ASR-heavy workflows (voice bots, IVR systems)."""

    wait_time = between(2, 5)  # realistic IVR pacing
    weight = 2  # 20% of users

    @task(3)
    @tag("asr")
    def transcribe_short(self):
        self.client.post(
            "/v1/asr",
            json={
                "audio_base64": SHORT_AUDIO,
                "lang": random.choice(["hi", "ta", "te", "bn"]),
                "consent_given": True,
                "backend": "bhashini",
            },
            headers=self.headers,
            name="/v1/asr [short]",
        )

    @task(1)
    @tag("asr")
    def transcribe_medium(self):
        self.client.post(
            "/v1/asr",
            json={
                "audio_base64": MEDIUM_AUDIO,
                "lang": "hi",
                "return_word_timestamps": True,
                "consent_given": True,
                "backend": "bhashini",
            },
            headers=self.headers,
            name="/v1/asr [medium]",
        )


class MTUser(BaseVAIUser):
    """Simulates translation-heavy workflows (content localization, e-commerce)."""

    wait_time = between(0.5, 2)
    weight = 4  # 40% of users

    @task(5)
    @tag("mt")
    def translate_single(self):
        src, tgt = random.choice(LANG_PAIRS)
        self.client.post(
            "/v1/translate",
            json={
                "text": self._random_text(src),
                "src_lang": src,
                "tgt_lang": tgt,
                "backend": "bhashini",
            },
            headers=self.headers,
            name="/v1/translate",
        )

    @task(2)
    @tag("mt")
    def translate_batch(self):
        src, tgt = random.choice(LANG_PAIRS)
        items = [
            {
                "text": self._random_text(src),
                "src_lang": src,
                "tgt_lang": tgt,
                "backend": "bhashini",
            }
            for _ in range(random.randint(3, 10))
        ]
        self.client.post(
            "/v1/translate/batch",
            json={"items": items},
            headers=self.headers,
            name="/v1/translate/batch",
        )


class LIDUser(BaseVAIUser):
    """Simulates LID-heavy workflows (auto-routing, content classification)."""

    wait_time = between(0.2, 1)
    weight = 3  # 30% of users

    @task(5)
    @tag("lid")
    def identify_text(self):
        lang = random.choice(list(SAMPLE_TEXTS.keys()))
        text = self._random_text(lang)
        self.client.post(
            "/v1/lid",
            json={"text": text, "top_k": 1},
            headers=self.headers,
            name="/v1/lid",
        )

    @task(1)
    @tag("lid")
    def identify_top_k(self):
        self.client.post(
            "/v1/lid",
            json={"text": "mera naam Rahul hai", "top_k": 3},
            headers=self.headers,
            name="/v1/lid [top-k]",
        )


class NLUUser(BaseVAIUser):
    """Simulates NLU/intent workflows (chatbots, voice assistants)."""

    wait_time = between(1, 3)
    weight = 1  # 10% of users

    QUERIES = [
        ("mera bill dikhao", "hi", "billing"),
        ("balance check karo", "hi", "banking"),
        ("namaste", "hi", "general"),
        ("gehu ka bhav batao", "hi", "agriculture"),
        ("ambulance chahiye", "hi", "healthcare"),
        ("Hello", "en", "general"),
    ]

    @task(4)
    @tag("nlu")
    def process_intent(self):
        text, lang, domain = random.choice(self.QUERIES)
        self.client.post(
            "/v1/nlu",
            json={
                "text": text,
                "lang": lang,
                "domain": domain,
                "generate_response": random.choice([True, False]),
            },
            headers=self.headers,
            name="/v1/nlu",
        )


class MixedUser(BaseVAIUser):
    """
    Realistic mixed workload — simulates a full voice-bot session:
    1. LID (detect language)
    2. ASR (transcribe)
    3. NLU (extract intent)
    4. MT (translate response if needed)
    5. TTS (synthesize response audio)
    """

    wait_time = between(1, 4)
    weight = 0  # Disabled by default (sequential multi-step is slow)

    @task
    @tag("mixed", "e2e")
    def full_voice_session(self):
        lang = random.choice(["hi", "ta"])

        # Step 1: LID
        lid_resp = self.client.post(
            "/v1/lid",
            json={"text": self._random_text(lang)},
            headers=self.headers,
            name="/v1/lid [session]",
        )
        if lid_resp.status_code != 200:
            return
        detected_lang = lid_resp.json().get("lang", lang)

        # Step 2: ASR
        self.client.post(
            "/v1/asr",
            json={
                "audio_base64": SHORT_AUDIO,
                "lang": detected_lang,
                "consent_given": True,
                "backend": "bhashini",
            },
            headers=self.headers,
            name="/v1/asr [session]",
        )

        # Step 3: NLU
        self.client.post(
            "/v1/nlu",
            json={
                "text": self._random_text(detected_lang),
                "lang": detected_lang,
                "generate_response": True,
            },
            headers=self.headers,
            name="/v1/nlu [session]",
        )
