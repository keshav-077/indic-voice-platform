# BHASHINI API Setup Guide

BHASHINI (BHASha INterface for India) is the Government of India's national language AI platform,
built by MeitY/Digital India. It provides free PoC access to ASR, MT, TTS, and OCR models for
22 constitutional Indian languages.

**Important licensing note**: The free tier is explicitly for Proof-of-Concept use only.
If you are building a commercial product or charging end-users, contact the BHASHINI team
for a paid plan: `digitalindiabhashinidivision@gmail.com`

---

## Step 1: Register on BHASHINI

1. Visit: **https://bhashini.gov.in/ulca/user/register**
2. Fill in your details (name, email, organization)
3. Check your email for verification link (also check spam)
4. Log in at: https://bhashini.gov.in/ulca/user/login

---

## Step 2: Get Your API Keys

1. After logging in, go to **My Profile** (top right)
2. Click **"Create API Key"**
3. Give it a name (e.g., `bharatvoiceai-dev`)
4. Copy:
   - **User ID** → set as `BHASHINI_USER_ID`
   - **API Key** → set as `BHASHINI_API_KEY`

---

## Step 3: Configure BharatVoiceAI

Edit your `.env`:
```bash
BHASHINI_USER_ID=your_user_id_here
BHASHINI_API_KEY=your_api_key_here

# Switch backends to use BHASHINI (easiest starting point)
VAI_ASR_BACKEND=bhashini
VAI_MT_BACKEND=bhashini
VAI_TTS_BACKEND=bhashini
```

---

## Step 4: Test the Connection

```bash
# Start the API
uvicorn api.main:app --reload

# Test health
curl http://localhost:8000/health

# Test translation (Hindi → Tamil)
curl -X POST http://localhost:8000/v1/translate \
  -H "X-API-Key: your-static-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते", "src_lang": "hi", "tgt_lang": "ta"}'
```

---

## BHASHINI API Architecture

BHASHINI uses a **3-call pipeline pattern**:

```
1. Search Call  → Find available pipelines for your task
2. Config Call  → Get model config for your specific pipeline
3. Compute Call → Run inference (actual ASR/MT/TTS)
```

BharatVoiceAI handles all three calls automatically. The `BHASHINI_PIPELINE_ID` is optional —
if not set, the dispatcher will auto-select the best available pipeline.

### Supported Tasks

| Task | API Code | Languages |
|---|---|---|
| ASR (Speech → Text) | `asr` | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, + more |
| Machine Translation | `translation` | All 22 constitutional languages, direct pairs |
| Text to Speech | `tts` | Major 12+ languages, male/female voices |
| Language Identification | `langDetection` | 22+ languages |
| OCR | `ocr` | Devanagari, Tamil, Telugu, Bengali scripts |
| Transliteration | `transliteration` | All scripts ↔ Latin |

### Rate Limits (Free Tier)

BHASHINI's free tier has undocumented limits. In practice:
- ~100 requests/day per API key for PoC
- Response time: 500ms–3000ms depending on load

For production, contact BHASHINI for SLA-backed plans.

---

## Available Service IDs

BHASHINI models are identified by `serviceId`. The full list is at:
https://dibd-bhashini.gitbook.io/bhashini-apis/available-models-for-usage

Key service IDs (as of February 2026):
- **ASR Hindi**: Use empty string to auto-select best model
- **MT any pair**: Pipeline-based, auto-routed
- **TTS Hindi Female**: Specified via `gender: "female"` in config

---

## Troubleshooting

### "401 Unauthorized"
- Check that `BHASHINI_USER_ID` and `BHASHINI_API_KEY` are set correctly
- Keys are case-sensitive
- Ensure email verification was completed

### "503 / Gateway Timeout"
- BHASHINI free tier can be slow during peak hours (Indian daytime)
- Retry with exponential backoff (BharatVoiceAI does this automatically)
- Consider switching to local models (`indicwhisper`, `indictrans2`) for reliability

### "No pipeline found"
- Some language pairs may not have pipelines on BHASHINI
- Check: https://bhashini.gov.in/ulca (try the web interface first)
- Fall back to IndicTrans2 for MT: set `VAI_MT_BACKEND=indictrans2`

### Getting a production plan
Email: `digitalindiabhashinidivision@gmail.com`
Subject: "BharatVoiceAI integration — requesting Bhashini App Mitra / Udyat Mitra status"

---

## Alternative: Run Models Locally

If BHASHINI is unavailable or too slow, run models locally:

```bash
# Install model extras
pip install -e ".[models]"

# Switch to local models
VAI_ASR_BACKEND=indicwhisper    # Downloads ~3GB model on first use
VAI_MT_BACKEND=indictrans2      # Downloads ~4GB model on first use
VAI_TTS_BACKEND=indic_parler    # Downloads ~8GB model on first use

# Requires: 16GB RAM minimum, GPU recommended
```

Models are downloaded from HuggingFace Hub and cached in `MODEL_CACHE_DIR`.
