# DPDP 2023 Compliance Guide

This document describes how BharatVoiceAI implements compliance with India's
**Digital Personal Data Protection Act (DPDP) 2023**.

> **Disclaimer**: This is technical documentation, not legal advice.
> Consult a qualified legal professional for compliance decisions.

---

## Relevant DPDP Obligations for BharatVoiceAI

| Section | Obligation | BharatVoiceAI Implementation |
|---|---|---|
| §6 | Affirmative consent for personal data | `consent_given=true` required on ASR endpoints |
| §7 | Purpose limitation | Audio processed only for transcription, not stored |
| §8(3) | Data minimisation | `VAI_AUDIO_LOGGING=false` default; no raw audio in logs |
| §8(7) | Data accuracy | Confidence scores returned; low-confidence flagged |
| §12 | Right to erasure | `DELETE /v1/user/data/{user_id}` endpoint |
| §13 | Right to access | `GET /v1/user/data/{user_id}` endpoint |
| §16 | Grievance mechanism | `POST /v1/grievance` endpoint + DPO contact |
| §17 | Data breach notification | Sentry integration + breach notification workflow |
| §9–11 | Children's data | Age check enforced if `USER_MIN_AGE` configured |
| §40 | Significant Data Fiduciary | DPO appointment required at scale (>10M users) |

---

## Key Defaults (Privacy-Protective)

```bash
VAI_AUDIO_LOGGING=false       # Raw audio NEVER stored in logs
VAI_PII_MASKING=true           # Aadhaar, phone, email masked in all logs
VAI_DATA_RETENTION_DAYS=0      # Audio/text deleted immediately after processing
CONSENT_REQUIRED=true          # consent_given=true mandatory for ASR
```

**Do not change these defaults without legal review.**

---

## Consent Implementation

### For ASR (audio processing)

Every ASR request must include:
```json
{
  "audio_url": "...",
  "lang": "hi",
  "consent_given": true
}
```

If `consent_given=false` or missing, the API returns:
```json
{
  "error": "consent_required",
  "message": "Under India's DPDP Act 2023, explicit user consent is required for processing audio."
}
```

### Consent UI Language (suggested, in Indic scripts)

Hindi: *"आपकी आवाज़ AI सेवाओं के लिए प्रोसेस की जाएगी। क्या आप सहमत हैं?"*
Tamil: *"உங்கள் குரல் AI சேவைகளுக்காக செயலாக்கப்படும். நீங்கள் ஒப்புக்கொள்கிறீர்களா?"*
Telugu: *"మీ గొంతు AI సేవల కోసం ప్రాసెస్ చేయబడుతుంది. మీరు అంగీకరిస్తున్నారా?"*

---

## Data Flow & What Is (Never) Stored

```
User Audio → API Gateway → ASR Service → Transcript → Response
                               ↓
                         [Audio discarded]
                         [Never written to disk]
                         [Never sent to logs]
```

What IS stored (for operational purposes):
- Request metadata: timestamp, endpoint, response_code, latency (no content)
- Hashed API key prefix (first 8 chars) for rate limiting
- Error counts per service (no PII)

What is NEVER stored:
- Raw audio bytes
- Transcript text (unless you explicitly enable logging — don't)
- User identifiers linked to requests
- Phone numbers, Aadhaar, email in any log

---

## PII Masking in Logs

The `RequestLoggingMiddleware` automatically masks:

| Pattern | Masked As |
|---|---|
| 10-digit phone numbers | `***PHONE***` |
| 12-digit Aadhaar numbers | `***AADHAAR***` |
| Email addresses | `***EMAIL***` |
| 16-digit card numbers | `***CARD***` |

---

## Data Erasure (Right to be Forgotten)

To implement §12 (right to erasure), add this endpoint to your application layer:

```python
@router.delete("/v1/user/data/{user_id}")
async def erase_user_data(user_id: str, user=Depends(get_current_user)):
    # 1. Delete from your application database
    # 2. Revoke active API keys/tokens for this user
    # 3. Log the erasure action (with timestamp only, no content)
    # 4. Return confirmation
    return {"status": "erased", "user_id": user_id, "timestamp": datetime.now().isoformat()}
```

**Note**: BharatVoiceAI itself stores no user data by default. If you add a database layer
for user management, ensure it includes erasure support.

---

## Significant Data Fiduciary (SDF) Obligations

If your deployment processes personal data of more than **10 million users** (or handles
sensitive personal data), you may be classified as a Significant Data Fiduciary. This requires:

- Appointing a **Data Protection Officer (DPO)**
- Conducting **Data Protection Impact Assessments (DPIAs)**
- Annual **audit by independent auditor**
- Maintaining **data processing records**

Adopt these practices early:

```bash
# Enable DPO contact endpoint
DPO_EMAIL=dpo@example.com
DPO_NAME="Your DPO Name"

# This enables GET /v1/dpo-contact in the API
```

---

## B2B Considerations

If you are providing BharatVoiceAI as a service to other businesses (B2B):

1. Include DPDP-compliant **Data Processing Agreement (DPA)** in contracts
2. Specify that customers are responsible for obtaining consent from their end-users
3. Provide a **data deletion SLA** (recommended: within 30 days of request)
4. Include breach notification obligations (72 hours to notify)

---

## References

- [DPDP Act 2023 Full Text](https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf)
- [MeitY DPDP FAQ](https://www.meity.gov.in/data-protection-framework)
- [BHASHINI Data Usage Policy](https://bhashini.gov.in/terms)
