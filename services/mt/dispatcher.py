"""
MT Dispatcher — Machine Translation backends.
Primary: IndicTrans2 (AI4Bharat) — SOTA on all 22 Indic languages.
Fallback: BHASHINI NMT, Sarvam Translate API.

IndicTrans2 reference: https://github.com/AI4Bharat/IndicTrans2
 - En-Indic: ai4bharat/indictrans2-en-indic-1B
 - Indic-En: ai4bharat/indictrans2-indic-en-1B
 - Indic-Indic: ai4bharat/indictrans2-indic-indic-1B
 - Distilled (faster, smaller): ai4bharat/indictrans2-en-indic-dist-200M
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod

import httpx

from api.schemas.models import MTBatchRequest, MTBatchResponse, MTRequest, MTResponse
from core.config import settings

logger = logging.getLogger(__name__)

# IndicTrans2 language token mapping
LANG_TO_FLORES = {
    "as": "asm_Beng",
    "bn": "ben_Beng",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "en": "eng_Latn",
    "gu": "guj_Gujr",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "ks": "kas_Arab",
    "kok": "kok_Deva",
    "mai": "mai_Deva",
    "ml": "mal_Mlym",
    "mni": "mni_Mtei",
    "mr": "mar_Deva",
    "ne": "npi_Deva",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "sa": "san_Deva",
    "sat": "sat_Olck",
    "sd": "snd_Arab",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ur": "urd_Arab",
}


class BaseMTBackend(ABC):
    name: str
    supported_langs: set[str]

    @abstractmethod
    async def translate(self, text: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
        """Returns dict: {translation: str, confidence: float}"""
        ...

    async def translate_batch(self, items: list[dict]) -> list[dict]:
        """Default: sequential. Override for true batching."""
        return [await self.translate(**item) for item in items]


class IndicTrans2Backend(BaseMTBackend):
    """
    AI4Bharat IndicTrans2 — SOTA multilingual NMT for 22 Indian languages.
    January 2025 update: RoPE-based long-context variants (up to 2048 tokens).

    Uses three models internally:
    - en_indic: English → any Indic
    - indic_en: any Indic → English
    - indic_indic: Indic → Indic (direct, no pivot)
    """

    name = "indictrans2"
    supported_langs = set(LANG_TO_FLORES.keys())

    # Model IDs (swappable with dist-200M for speed vs accuracy trade-off)
    MODELS = {
        "en_indic": "ai4bharat/indictrans2-en-indic-1B",
        "indic_en": "ai4bharat/indictrans2-indic-en-1B",
        "indic_indic": "ai4bharat/indictrans2-indic-indic-1B",
    }

    def __init__(self) -> None:
        self._models: dict = {}
        self._tokenizers: dict = {}
        self._ip = None  # IndicProcessor
        self._loaded: set[str] = set()

    def _get_model_key(self, src: str, tgt: str) -> str:
        if src == "en":
            return "en_indic"
        if tgt == "en":
            return "indic_en"
        return "indic_indic"

    async def _ensure_loaded(self, model_key: str) -> None:
        if model_key not in self._loaded:
            await asyncio.get_event_loop().run_in_executor(None, self._load_model, model_key)

    def _load_model(self, model_key: str) -> None:
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_name = self.MODELS[model_key]
        logger.info("Loading IndicTrans2 model: %s", model_name)
        device = _resolve_device()

        self._tokenizers[model_key] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=settings.MODEL_CACHE_DIR,
        )
        self._models[model_key] = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            cache_dir=settings.MODEL_CACHE_DIR,
        ).to(device)
        self._models[model_key].eval()

        if self._ip is None:
            self._ip = IndicProcessor(inference=True)

        self._loaded.add(model_key)
        logger.info("IndicTrans2 [%s] loaded on %s", model_key, device)

    async def translate(self, text: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
        model_key = self._get_model_key(src_lang, tgt_lang)
        await self._ensure_loaded(model_key)

        return await asyncio.get_event_loop().run_in_executor(
            None, self._run_inference, model_key, text, src_lang, tgt_lang
        )

    def _run_inference(self, model_key: str, text: str, src_lang: str, tgt_lang: str) -> dict:
        import torch

        src_flores = LANG_TO_FLORES[src_lang]
        tgt_flores = LANG_TO_FLORES[tgt_lang]

        tokenizer = self._tokenizers[model_key]
        model = self._models[model_key]
        device = next(model.parameters()).device

        batch = self._ip.preprocess_batch([text], src_lang=src_flores, tgt_lang=tgt_flores)
        inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=5,
                num_return_sequences=1,
                max_length=512,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations = self._ip.postprocess_batch(decoded, lang=tgt_flores)

        return {
            "translation": translations[0] if translations else "",
            "confidence": 0.88,  # IndicTrans2 doesn't expose sequence-level scores directly
        }

    async def translate_batch(self, items: list[dict]) -> list[dict]:
        """Efficient batched translation — groups by model_key."""
        from collections import defaultdict

        groups: dict[str, list] = defaultdict(list)
        for i, item in enumerate(items):
            key = self._get_model_key(item["src_lang"], item["tgt_lang"])
            groups[key].append((i, item))

        results = [None] * len(items)
        for model_key, group in groups.items():
            await self._ensure_loaded(model_key)
            indices = [g[0] for g in group]
            batch_items = [g[1] for g in group]

            batch_results = await asyncio.get_event_loop().run_in_executor(
                None, self._run_batch_inference, model_key, batch_items
            )
            for idx, res in zip(indices, batch_results, strict=False):
                results[idx] = res

        return results

    def _run_batch_inference(self, model_key: str, items: list[dict]) -> list[dict]:
        import torch

        tokenizer = self._tokenizers[model_key]
        model = self._models[model_key]
        device = next(model.parameters()).device

        # Each item may have different src/tgt lang — group further by lang pair
        from itertools import groupby

        sorted_items = sorted(enumerate(items), key=lambda x: (x[1]["src_lang"], x[1]["tgt_lang"]))

        results = [None] * len(items)
        for (src, tgt), group in groupby(sorted_items, key=lambda x: (x[1]["src_lang"], x[1]["tgt_lang"])):
            group = list(group)
            original_indices = [g[0] for g in group]
            texts = [g[1]["text"] for g in group]

            src_flores = LANG_TO_FLORES[src]
            tgt_flores = LANG_TO_FLORES[tgt]

            batch = self._ip.preprocess_batch(texts, src_lang=src_flores, tgt_lang=tgt_flores)
            inputs = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(**inputs, num_beams=4, max_length=512)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translations = self._ip.postprocess_batch(decoded, lang=tgt_flores)

            for orig_idx, translation in zip(original_indices, translations, strict=False):
                results[orig_idx] = {"translation": translation, "confidence": 0.88}

        return results


class BhashiniMTBackend(BaseMTBackend):
    """
    BHASHINI NMT — direct language-to-language models (no Hindi pivot).
    Good fallback when IndicTrans2 is not available.
    """

    name = "bhashini"
    supported_langs = set(LANG_TO_FLORES.keys())
    COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

    async def translate(self, text: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
        if not settings.is_bhashini_configured:
            raise RuntimeError("BHASHINI credentials not configured.")

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": src_lang,
                            "targetLanguage": tgt_lang,
                        },
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

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self.COMPUTE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        translation = data["pipelineResponse"][0]["output"][0].get("target", "")
        return {"translation": translation, "confidence": 0.82}


class SarvamMTBackend(BaseMTBackend):
    """
    Sarvam AI Translate API — open-weight translation model for 22 Indian languages.
    Supports structured long-form text. Requires SARVAM_API_KEY.
    """

    name = "sarvam"
    supported_langs = set(LANG_TO_FLORES.keys())
    BASE_URL = "https://api.sarvam.ai/translate"

    SARVAM_LANG_MAP = {
        "hi": "hi-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "bn": "bn-IN",
        "mr": "mr-IN",
        "gu": "gu-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "pa": "pa-IN",
        "or": "od-IN",
        "en": "en-IN",
    }

    async def translate(self, text: str, src_lang: str, tgt_lang: str, **kwargs) -> dict:
        if not settings.is_sarvam_configured:
            raise RuntimeError("SARVAM_API_KEY not configured.")

        src = self.SARVAM_LANG_MAP.get(src_lang, src_lang)
        tgt = self.SARVAM_LANG_MAP.get(tgt_lang, tgt_lang)

        payload = {
            "input": text,
            "source_language_code": src,
            "target_language_code": tgt,
            "speaker_gender": "Male",
            "mode": "formal",
            "model": "mayura:v1",
        }
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(self.BASE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return {"translation": data.get("translated_text", ""), "confidence": 0.87}


class MTDispatcher:
    """Route MT requests to the best available backend."""

    def __init__(self, default_backend: str = "indictrans2") -> None:
        self._default = default_backend
        self._backends: dict[str, BaseMTBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            self._backends["indictrans2"] = IndicTrans2Backend()
        except Exception as e:
            logger.warning("IndicTrans2 unavailable: %s", e)

        if settings.is_bhashini_configured:
            self._backends["bhashini"] = BhashiniMTBackend()

        if settings.is_sarvam_configured:
            self._backends["sarvam"] = SarvamMTBackend()

    async def translate(self, request: MTRequest) -> MTResponse:
        start = time.perf_counter()
        backend = self._select(request.backend, request.src_lang, request.tgt_lang)

        result = await backend.translate(
            text=request.text,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
        )

        return MTResponse(
            translation=result["translation"],
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            confidence=result["confidence"],
            backend_used=backend.name,
            processing_ms=round((time.perf_counter() - start) * 1000),
        )

    async def translate_batch(self, request: MTBatchRequest) -> MTBatchResponse:
        start = time.perf_counter()
        tasks = [self.translate(item) for item in request.items]
        results = await asyncio.gather(*tasks)
        return MTBatchResponse(
            results=list(results),
            total_processing_ms=round((time.perf_counter() - start) * 1000),
        )

    def _select(self, backend_pref: str, src: str, tgt: str) -> BaseMTBackend:
        name = backend_pref if backend_pref != "auto" else self._default
        if name in self._backends:
            return self._backends[name]

        # Fallback chain
        for fb in ["indictrans2", "sarvam", "bhashini"]:
            if fb in self._backends:
                return self._backends[fb]

        raise RuntimeError("No MT backend available.")


def _resolve_device() -> str:
    try:
        import torch

        if settings.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return "cpu"
