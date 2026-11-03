# Contributing to BharatVoiceAI

First off — **شुक्रिया / ধন্যবাদ / ధన్యవాదాలు / நன்றி / ನಮಸ್ಕಾರ** — thank you for contributing!

This project exists to make AI accessible in every Indian language. Every contribution — code, data, translations, documentation, or bug reports — matters.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Adding a New Language](#adding-a-new-language)
- [Adding a New Model Backend](#adding-a-new-model-backend)
- [Adding New NLU Intents](#adding-new-nlu-intents)
- [Pull Request Process](#pull-request-process)
- [Commit Convention](#commit-convention)
- [Testing Requirements](#testing-requirements)
- [DPDP Compliance for Contributors](#dpdp-compliance-for-contributors)

---

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). By participating, you agree to uphold it. Report issues to maintainers via email.

---

## Ways to Contribute

| Type | What you can do |
|---|---|
| **Bug fix** | Fix incorrect transcription, wrong API response, crashes |
| **New language** | Add a new Indic language (model + test data + LID label) |
| **New backend** | Integrate a new ASR/MT/TTS model or API |
| **New intents** | Add NLU intent YAML for a new domain (healthcare, agriculture...) |
| **Docs** | Improve README, architecture docs, BHASHINI setup guide |
| **Tests** | Add unit/integration tests for edge cases |
| **Benchmarks** | Run evaluations and contribute benchmark results |
| **Data** | Contribute voice recordings or translations to BHASHINI/Bhashadaan |

---

## Development Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Git

### Quick Start

```bash
# 1. Fork & clone
git clone https://github.com/YOUR_USERNAME/bharatvoiceai.git
cd bharatvoiceai

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy and configure environment
cp .env.example .env
# Edit .env — at minimum set BHASHINI credentials (free registration)
# or set VAI_NLU_BACKEND=rule for zero-config start

# 6. Start dependencies
docker compose up -d postgres redis

# 7. Run the API
uvicorn api.main:app --reload --port 8000

# 8. Run tests
pytest tests/unit/ -v
```

### Running the Full Stack

```bash
# Start everything (API + DB + Redis + Prometheus + Grafana)
docker compose up -d

# View logs
docker compose logs -f api

# Stop
docker compose down
```

---

## Adding a New Language

Adding a language that isn't yet supported involves several steps.

### 1. Add language code to schemas

Edit `api/schemas/models.py`:
```python
SUPPORTED_LANGS = {
    ...,
    "new_lang_code",  # e.g. "bho" for Bhojpuri
}
```

### 2. Add LID mapping

Edit `services/lid/dispatcher.py`:
```python
LANG_SCRIPT_MAP = {
    ...,
    "bho": "Devanagari",  # Bhojpuri uses Devanagari script
}
```

### 3. Add FLORES mapping (for MT evaluation)

Edit `services/mt/dispatcher.py`:
```python
LANG_TO_FLORES = {
    ...,
    "bho": "bho_Deva",
}
```

### 4. Add TTS voice prompts (optional but recommended)

Edit `services/tts/dispatcher.py`:
```python
VOICE_PROMPTS = {
    ...,
    "bho": {
        TTSVoice.FEMALE_1: "Describe the voice characteristics for this language...",
    }
}
```

### 5. Add test cases

Add tests in `tests/unit/test_all.py` for the new language:
```python
async def test_bhojpuri_lid(self):
    results = await backend.identify_text("राम के जाता बा", top_k=1)
    assert results[0]["lang"] in ("hi", "bho")  # Bhojpuri close to Hindi
```

### 6. Submit a PR

Open a PR titled: `feat(lang): add Bhojpuri (bho) support`

---

## Adding a New Model Backend

Each service (ASR, MT, TTS, LID, NLU) uses a dispatcher pattern with pluggable backends.

### Example: Adding a new ASR backend

Create your backend class in `services/asr/dispatcher.py`:

```python
class MyNewASRBackend(BaseASRBackend):
    name = "my_backend"
    supported_langs = {"hi", "ta", "en"}

    async def transcribe(self, audio_bytes: bytes, lang: str | None = None, **kwargs) -> dict:
        # Your implementation here
        # Must return: {"transcript": str, "confidence": float, "lang_detected": str, "words": list}
        ...
```

Register it in `ASRDispatcher._init_backends()`:
```python
def _init_backends(self):
    ...
    try:
        self._backends["my_backend"] = MyNewASRBackend()
    except Exception as e:
        logger.warning("MyBackend unavailable: %s", e)
```

Add it to the config choices in `core/config.py`:
```python
VAI_ASR_BACKEND: Literal["indicwhisper", "bhashini", "wav2vec2", "my_backend", "local"] = "indicwhisper"
```

Add tests and update `README.md` benchmarks section.

---

## Adding New NLU Intents

NLU intents are defined in YAML files. To add a new domain (e.g., agricultural advisory):

Create `services/nlu/intents/agriculture.yml`:

```yaml
# Agriculture domain intents
# Add patterns in Hindi, English, and other major Indic languages
intents:
  check_crop_price:
    patterns:
      - "(fasal|crop|gehu|chawal|dhan)\\s*(ka\\s*)?(bhav|price|rate|dam)"
      - "(mandi|market)\\s*(mein\\s*)?(kya\\s*)?(rate|bhav|price)"
    entities:
      - crop_name
      - location
      - date
    domain: agriculture
    responses:
      hi: "फसल की कीमत जाँच रहे हैं..."
      en: "Checking crop prices..."
      pa: "ਫ਼ਸਲ ਦੀ ਕੀਮਤ ਜਾਂਚ ਰਹੇ ਹਾਂ..."

  weather_for_farming:
    patterns:
      - "(baarish|rain|barsat)\\s*(kab|when|hogi|aayegi)"
      - "(kheti|farming|fasal)\\s*(ke\\s*liye\\s*)?(mausam|weather)"
    entities:
      - location
      - date
    domain: agriculture
    responses:
      hi: "कृषि मौसम की जानकारी ले रहे हैं..."
      en: "Getting agricultural weather forecast..."
```

The YAML is auto-loaded at startup — no code changes required.

---

## Pull Request Process

1. **Branch naming**: `feat/`, `fix/`, `docs/`, `test/`, `chore/` + short description
   - e.g. `feat/add-bhojpuri-support`, `fix/asr-timeout-handling`

2. **Commit messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/)
   ```
   feat(asr): add IndicWhisper streaming support
   fix(mt): handle empty translation response from BHASHINI
   docs: update BHASHINI setup guide with 2025 API changes
   test(nlu): add Marathi intent classification tests
   ```

3. **DCO sign-off**: Add to every commit:
   ```bash
   git commit -s -m "feat: my change"
   # This adds: Signed-off-by: Your Name <your@email.com>
   ```

4. **PR checklist**:
   - [ ] Tests pass (`pytest tests/unit/ -v`)
   - [ ] Linting passes (`ruff check .`)
   - [ ] Added tests for new code (coverage ≥ 70%)
   - [ ] Updated `CHANGELOG.md`
   - [ ] Updated `README.md` if API surface changed
   - [ ] DPDP-compliant (no audio logged, PII masked)

---

## Commit Convention

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
Signed-off-by: Name <email>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`

Scopes: `asr`, `mt`, `tts`, `lid`, `nlu`, `api`, `infra`, `ci`, `docs`, `lang`

---

## Testing Requirements

- **New features**: must have unit tests
- **Bug fixes**: must have a regression test (test that would have caught the bug)
- **New backends**: must mock external API calls (use `respx` for httpx mocking)
- **Coverage**: project-wide coverage must stay ≥ 70%

```bash
# Run tests with coverage report
pytest tests/unit/ -v --cov --cov-report=term-missing

# Run only fast tests
pytest tests/unit/ -v -m "not slow"

# Run specific test
pytest tests/unit/test_all.py::TestRuleNLU::test_view_bill_hindi -v
```

---

## DPDP Compliance for Contributors

All contributions must respect India's **Digital Personal Data Protection Act 2023**:

1. **Never log raw audio** — `VAI_AUDIO_LOGGING=false` must remain the default
2. **Mask PII in all log statements** — use `mask_pii()` from `api.middleware.logging`
3. **No storing audio longer than needed** — `VAI_DATA_RETENTION_DAYS=0` by default
4. **Test data must not contain real PII** — use synthetic/anonymized data in tests
5. **New endpoints collecting audio must include `consent_given` field**

If you're unsure whether a change is DPDP-compliant, open an issue to discuss first.

---

## Questions?

- **Bug?** → [Open an issue](https://github.com/samirdutta/bharatvoiceai/issues/new?template=bug_report.md)
- **Idea?** → [Start a discussion](https://github.com/samirdutta/bharatvoiceai/discussions)
- **Chat** → Join our community on Discord (link in README)

---

*आपके योगदान के लिए धन्यवाद! — Thank you for your contribution!*
