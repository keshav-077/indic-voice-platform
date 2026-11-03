"""
NLU Dispatcher — Natural Language Understanding.
Supports intent classification + entity extraction + response generation.

Strategy:
  1. Rule-based engine (fast, zero-latency, high precision for known intents)
  2. LLM fallback (Sarvam-M / Sarvam-1 via API) for open-domain queries
  3. Hybrid: rule engine tries first, LLM takes over for unknown intents

Domain-specific intent maps are loaded from YAML files (extensible by community).
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx
import yaml

from api.schemas.models import NLUEntity, NLURequest, NLUResponse
from core.config import settings

logger = logging.getLogger(__name__)

INTENT_DATA_DIR = Path(__file__).parent / "intents"

# Intent definitions
# Each domain YAML has: intents: {name: {patterns: [...], entities: [...], responses: {...}}}
# Community can add new domains by creating domain.yml files.

DEFAULT_INTENTS: dict[str, dict] = {
    # billing domain — Hindi + English patterns
    "view_bill": {
        "patterns": [
            r"(bill?|billed|invoice|statement)\s*(dekhao|dikhao|show|check|view|dekh)",
            r"(mera?|meri|apna|my)\s*(bill?|invoice|statement)",
            r"bijli\s*bill",
            r"paani\s*bill",
            r"electric(ity)?\s*bill",
            r"bil\s*dikh",  # transliterated Devanagari: बिल दिखाओ → bil dikhao
        ],
        "entities": [],
        "domain": "billing",
        "responses": {
            "hi": "आपका बिल लाया जा रहा है...",
            "en": "Fetching your bill...",
            "ta": "உங்கள் பில் கொண்டு வருகிறேன்...",
        },
    },
    "pay_bill": {
        "patterns": [
            r"(bill?|invoice)\s*(pay|bhar|bharo|jama|submit|pe\b)",
            r"(pay|bhugtan|bhugtaan)\s*(kar|karo|karna|karan)",
        ],
        "entities": ["amount", "account_id"],
        "domain": "billing",
        "responses": {
            "hi": "भुगतान शुरू किया जा रहा है...",
            "en": "Initiating payment...",
        },
    },
    "check_balance": {
        "patterns": [
            r"(balance|bailems|bakaya|baaki|remaining)\s*(check|chek|dekh|kitna|kya|how\s*much)",
            r"(account|khata)\s*(balance|mein\s*kitna)",
            r"kitna\s*(paisa|paise|rupees|rupaya)\s*(bacha|bache|hai|hain)",
        ],
        "entities": ["account_id"],
        "domain": "banking",
        "responses": {
            "hi": "आपका बैलेंस चेक किया जा रहा है...",
            "en": "Checking your balance...",
        },
    },
    "book_appointment": {
        "patterns": [
            r"(appointment|milna|milne\s*ka\s*waqt|time)\s*(book|le|lena|chahiye|fix)",
            r"(doctor|daaktar|da.ktar)\s*(se\s*mil|appointment)",
        ],
        "entities": ["date", "time", "doctor_name"],
        "domain": "healthcare",
        "responses": {
            "hi": "अपॉइंटमेंट बुक हो रही है...",
            "en": "Booking your appointment...",
        },
    },
    "check_weather": {
        "patterns": [
            r"(mausam|weather|baarish|barish|rain|garmi|sardi)\s*(kaisa|kya|how|aaj|kal|today|tomorrow|kab|hog)?",
            r"(aaj|kal|today|tomorrow)\s*(ka\s*)?(mausam|weather)",
        ],
        "entities": ["location", "date"],
        "domain": "general",
        "responses": {
            "hi": "मौसम की जानकारी ले रहा हूं...",
            "en": "Getting weather information...",
        },
    },
    "greet": {
        "patterns": [
            r"^(namaste|namaskar|hello|hi|hey|hola|vanakkam|pranam|sat\s*sri\s*akal|kem\s*cho)\b",
            r"^(good\s*morning|good\s*evening|good\s*afternoon)",
        ],
        "entities": [],
        "domain": "general",
        "responses": {
            "hi": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?",
            "en": "Hello! How can I help you today?",
            "ta": "வணக்கம்! நான் உங்களுக்கு எவ்வாறு உதவலாம்?",
        },
    },
    "thanks": {
        "patterns": [
            r"(dhanyavaad|shukriya|thank\s*you|thanks|theek\s*hai|ok|okay|acha|accha)\b",
        ],
        "entities": [],
        "domain": "general",
        "responses": {
            "hi": "आपका स्वागत है! क्या और कुछ चाहिए?",
            "en": "You're welcome! Is there anything else I can help with?",
        },
    },
}


class BaseNLUBackend(ABC):
    name: str

    @abstractmethod
    async def process(self, text: str, lang: str, domain: str, context: dict | None) -> dict:
        """
        Returns:
            intent: str
            confidence: float
            entities: list[dict]
            response_text: str | None
        """
        ...


class RuleNLUBackend(BaseNLUBackend):
    """
    Fast regex-based intent classifier.
    Loads from intents/ YAML files + default patterns above.
    Latency: ~1ms. No GPU/API required.
    """

    name = "rule"

    def __init__(self) -> None:
        self._intents = dict(DEFAULT_INTENTS)
        self._load_yaml_intents()
        # Compile patterns for performance
        self._compiled: dict[str, list] = {
            name: [re.compile(p, re.IGNORECASE | re.UNICODE) for p in data["patterns"]]
            for name, data in self._intents.items()
        }

    def _load_yaml_intents(self) -> None:
        """Load community YAML intent files from intents/ directory."""
        if not INTENT_DATA_DIR.exists():
            return
        for path in INTENT_DATA_DIR.glob("*.yml"):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                intents = data.get("intents", {})
                self._intents.update(intents)
                logger.info("Loaded intent file: %s (%d intents)", path.name, len(intents))
            except Exception as e:
                logger.warning("Failed to load intent file %s: %s", path, e)

    async def process(self, text: str, lang: str, domain: str, context: dict | None) -> dict:
        text_lower = text.lower().strip()

        # Transliterate Indic script to romanized for pattern matching
        # This allows Devanagari input to match romanized regex patterns
        romanized = _transliterate_to_roman(text_lower)
        # Match against original, romanized, and romanized-with-trailing-a (ITRANS convention)
        match_texts = [text_lower]
        if romanized != text_lower:
            match_texts.append(romanized)
            # Also add version with trailing 'a' restored (for patterns like "gussa", "baarish")
            words_with_a = " ".join(
                w + "a" if len(w) > 2 and not w.endswith(("a", "e", "i", "o", "u")) else w for w in romanized.split()
            )
            if words_with_a != romanized:
                match_texts.append(words_with_a)

        best_intent = "unknown"
        best_confidence = 0.0
        matched_intent_data = None

        for intent_name, patterns in self._compiled.items():
            intent_data = self._intents[intent_name]
            # Domain filter
            if domain != "general" and intent_data.get("domain", "general") not in (
                domain,
                "general",
            ):
                continue

            for pattern in patterns:
                if any(pattern.search(t) for t in match_texts):
                    confidence = 0.90  # rule match = high confidence
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent_name
                        matched_intent_data = intent_data
                    break

        # Extract entities (simple regex-based extraction)
        entities = []
        if matched_intent_data:
            entities = _extract_entities(text, matched_intent_data.get("entities", []))

        # Get response in the right language
        response_text = None
        if matched_intent_data:
            responses = matched_intent_data.get("responses", {})
            response_text = responses.get(lang) or responses.get("en") or responses.get("hi")

        return {
            "intent": best_intent,
            "confidence": best_confidence if best_intent != "unknown" else 0.0,
            "entities": entities,
            "response_text": response_text,
        }


class SarvamLLMBackend(BaseNLUBackend):
    """
    Sarvam AI LLM backend for open-domain NLU.
    Uses Sarvam-M (24B, Mistral-based, Apache 2.0) via Sarvam API.
    Falls back to Sarvam-1 (2B) for speed.

    Sarvam-M achieves +20% on Indian language benchmarks vs base Mistral Small.
    """

    name = "llm"

    SYSTEM_PROMPT = """You are an intelligent assistant that understands Indian languages.
Given a user query, respond with JSON only (no other text):
{
  "intent": "<intent_name>",
  "confidence": <0.0-1.0>,
  "entities": [{"type": "<entity_type>", "value": "<value>", "confidence": <0.0-1.0>}],
  "response_text": "<natural language response in the same language as the query>"
}

Common intents: view_bill, pay_bill, check_balance, book_appointment, check_weather,
get_info, file_complaint, track_order, greet, thanks, unknown.

Be concise. Respond in the same language as the user's query."""

    async def process(self, text: str, lang: str, domain: str, context: dict | None) -> dict:
        if not settings.is_sarvam_configured:
            raise RuntimeError("SARVAM_API_KEY not configured.")

        messages = []
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT + f"\n\nDomain context: {domain}",
                }
            )
        else:
            messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        messages.append({"role": "user", "content": text})

        payload = {
            "model": settings.SARVAM_LLM_MODEL,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.1,  # low temperature for structured output
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{settings.SARVAM_BASE_URL}/chat/completions",
                json=payload,
                headers={"api-subscription-key": settings.SARVAM_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"]

        import json

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from messy output
            import re

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}

        return {
            "intent": parsed.get("intent", "unknown"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "entities": parsed.get("entities", []),
            "response_text": parsed.get("response_text"),
        }


class HybridNLUBackend(BaseNLUBackend):
    """
    Rule engine first → LLM fallback for low-confidence / unknown intents.
    Best of both: speed + coverage.
    Threshold: if rule confidence < 0.5, escalate to LLM.
    """

    name = "rule+llm"
    ESCALATION_THRESHOLD = 0.5

    def __init__(self) -> None:
        self._rule = RuleNLUBackend()
        self._llm: SarvamLLMBackend | None = None
        if settings.is_sarvam_configured:
            self._llm = SarvamLLMBackend()
        else:
            logger.info(
                "Sarvam not configured — NLU will use rule-only mode. "
                "Set SARVAM_API_KEY for LLM-powered open-domain NLU."
            )

    async def process(self, text: str, lang: str, domain: str, context: dict | None) -> dict:
        # Try rules first (always fast)
        rule_result = await self._rule.process(text, lang, domain, context)

        if rule_result["confidence"] >= self.ESCALATION_THRESHOLD:
            rule_result["_backend"] = "rule"
            return rule_result

        # Escalate to LLM
        if self._llm:
            try:
                llm_result = await self._llm.process(text, lang, domain, context)
                llm_result["_backend"] = "llm"
                # Merge: LLM response, but keep rule entities if LLM found none
                if not llm_result.get("entities") and rule_result.get("entities"):
                    llm_result["entities"] = rule_result["entities"]
                return llm_result
            except Exception as e:
                logger.warning("LLM NLU failed (%s), returning rule result.", e)

        return rule_result


class NLUDispatcher:
    def __init__(self, default_backend: str = "rule+llm") -> None:
        self._default = default_backend
        self._backends: dict[str, BaseNLUBackend] = {}
        self._init_backends()

    def _init_backends(self) -> None:
        self._backends["rule"] = RuleNLUBackend()
        try:
            self._backends["rule+llm"] = HybridNLUBackend()
        except Exception as e:
            logger.warning("Hybrid NLU init failed: %s", e)
        if settings.is_sarvam_configured:
            try:
                self._backends["llm"] = SarvamLLMBackend()
            except Exception as e:
                logger.warning("LLM NLU init failed: %s", e)

    async def process(self, request: NLURequest) -> NLUResponse:
        start = time.perf_counter()
        backend_name = self._default
        backend = self._backends.get(backend_name, self._backends["rule"])

        result = await backend.process(
            text=request.text,
            lang=request.lang,
            domain=request.domain.value,
            context=request.context,
        )

        # If generate_response is False, clear response_text
        response_text = result.get("response_text") if request.generate_response else None

        entities = [
            NLUEntity(
                type=e.get("type", "unknown"),
                value=str(e.get("value", "")),
                confidence=float(e.get("confidence", 0.8)),
            )
            for e in result.get("entities", [])
        ]

        backend_used = result.get("_backend", backend.name)

        return NLUResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            entities=entities,
            response_text=response_text,
            backend_used=backend_used,
            processing_ms=round((time.perf_counter() - start) * 1000),
        )


# Entity extractors


def _transliterate_to_roman(text: str) -> str:
    """
    Convert Indic script text to romanized form for pattern matching.
    Uses indic-transliteration library with ITRANS scheme.
    Post-processes to strip trailing 'a' from words (Sanskrit virama convention)
    so that बिल→"bil" matches "bill" patterns and मदद→"madad" not "madada".
    """
    # Quick check: if all ASCII, no transliteration needed
    if all(ord(c) < 0x0900 for c in text):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        # Detect script and transliterate
        script_map = {
            (0x0900, 0x097F): sanscript.DEVANAGARI,
            (0x0980, 0x09FF): sanscript.BENGALI,
            (0x0B80, 0x0BFF): sanscript.TAMIL,
            (0x0C00, 0x0C7F): sanscript.TELUGU,
            (0x0C80, 0x0CFF): sanscript.KANNADA,
            (0x0D00, 0x0D7F): sanscript.MALAYALAM,
            (0x0A80, 0x0AFF): sanscript.GUJARATI,
            (0x0A00, 0x0A7F): sanscript.GURMUKHI,
        }

        src_script = None
        for ch in text:
            cp = ord(ch)
            for (lo, hi), script in script_map.items():
                if lo <= cp <= hi:
                    src_script = script
                    break
            if src_script:
                break

        if not src_script:
            return text

        result = transliterate(text, src_script, sanscript.ITRANS).lower()

        # Post-process: strip trailing 'a' from each word (ITRANS adds inherent 'a')
        # बिल → "bila" → "bil", मदद → "madada" → "madad", करना → "karana" → "karan"
        # But preserve short words and common endings
        words = result.split()
        cleaned = []
        for w in words:
            # Strip trailing 'a' if word is long enough and ends with 'a'
            # but not if the word IS just 'a' or common words like 'na', 'ka'
            if len(w) > 2 and w.endswith("a") and not w.endswith("aa"):
                cleaned.append(w[:-1])
            else:
                cleaned.append(w)
        result_stripped = " ".join(cleaned)

        # Return both forms joined — so patterns can match either
        # "bila dikhao" OR "bil dikhao" will be searched
        return result_stripped

    except ImportError:
        logger.debug("indic-transliteration not installed, skipping transliteration")
        return text
    except Exception:
        return text


ENTITY_PATTERNS = {
    "amount": [
        r"(?:rs\.?|rupees?|₹)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)",
        r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs\.?|rupees?|₹)",
    ],
    "phone": [r"(?:\+?91[-\s]?)?([6-9]\d{9})"],
    "date": [
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        r"\b(aaj|kal|parso|today|tomorrow|yesterday)\b",
        r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*)\b",
    ],
    "account_id": [r"\b(\d{9,18})\b"],
    "location": [],  # Needs NER model — placeholder
}


def _extract_entities(text: str, entity_types: list[str]) -> list[dict]:
    found = []
    for etype in entity_types:
        patterns = ENTITY_PATTERNS.get(etype, [])
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match if isinstance(match, str) else match[0]
                if value:
                    found.append({"type": etype, "value": value, "confidence": 0.85})
                    break  # one match per entity type
    return found
