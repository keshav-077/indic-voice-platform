# Model Cards — BharatVoiceAI Integrated Models

This document describes every AI model integrated in BharatVoiceAI,
including license, capabilities, limitations, and benchmarks.

---

## ASR Models

### IndicWhisper (AI4Bharat)
| Field | Value |
|---|---|
| **Source** | AI4Bharat / IIT Madras |
| **HuggingFace** | `ai4bharat/indicwhisper-large-v2` |
| **License** | MIT |
| **Architecture** | Whisper Large v2 fine-tuned on Vistaar benchmark |
| **Languages** | Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu (12 total) |
| **Training Data** | 10,700+ hours across 12 languages (Vistaar training set) |
| **Size** | ~1.5GB |
| **Benchmark** | Lowest WER on 39 out of 59 Vistaar benchmarks |

**Known Limitations**:
- Dialects (e.g. Marwari, Bhojpuri, rural accents) may have higher WER
- Code-mixed speech (Hinglish) reduces accuracy; dedicated model planned
- Background noise sensitivity (recommend client-side noise suppression)

**Sample WER Results (Vistaar test set)**:

| Language | Domain | WER |
|---|---|---|
| Hindi | News | 8.2% |
| Hindi | Conversational | 14.1% |
| Tamil | News | 12.4% |
| Telugu | General | 15.8% |
| Bengali | General | 16.2% |
| Marathi | General | 18.4% |

---

### Wav2Vec2-Indic (AI4Bharat IndicWav2Vec)
| Field | Value |
|---|---|
| **Source** | AI4Bharat |
| **HuggingFace** | `ai4bharat/indicwav2vec-{lang}` |
| **License** | MIT |
| **Architecture** | Wav2Vec2 base, fine-tuned per language |
| **Languages** | Hindi, Tamil, Telugu, Bengali, Marathi (5 separate models) |
| **Size** | ~360MB per language |
| **Best For** | CPU inference, lower memory, faster on short utterances |

---

### BHASHINI ASR (Government of India)
| Field | Value |
|---|---|
| **Source** | MeitY / Digital India Bhashini Division |
| **License** | Government of India ToU (PoC free; commercial via agreement) |
| **Architecture** | Various (IITB, IITM, CDAC models) |
| **Languages** | 36+ text languages, 23 voice languages |
| **Access** | REST API (requires free registration) |
| **Latency** | 500ms–3000ms (cloud inference) |

---

## Machine Translation Models

### IndicTrans2 (AI4Bharat)
| Field | Value |
|---|---|
| **Source** | AI4Bharat / IIT Madras |
| **HuggingFace** | `ai4bharat/indictrans2-{en-indic,indic-en,indic-indic}-1B` |
| **License** | MIT |
| **Architecture** | Transformer seq2seq, 1B parameters |
| **Languages** | All 22 scheduled Indian languages + English |
| **Training Data** | BPCC (IndicTrans2 curated corpus): 8.5B+ tokens |
| **Size** | ~4GB per model (3 models total) |
| **January 2025 Update** | RoPE-based long-context variants (up to 2048 tokens) |
| **Distilled variant** | `indictrans2-*-dist-200M` (200M params, faster, slightly lower quality) |

**FLORES-200 Benchmark (IN22-Gen, devtest)**:

| Language Pair | BLEU | ChrF |
|---|---|---|
| En → Hi | 33.1 | 58.4 |
| En → Ta | 27.8 | 51.2 |
| En → Te | 29.4 | 53.1 |
| En → Bn | 31.2 | 56.8 |
| Hi → En | 37.4 | 62.1 |
| Ta → Hi | 28.9 | 52.7 |
| Hi → Ta | 26.1 | 50.4 |

*All numbers from AI4Bharat's published IN22 evaluation.*

**Known Limitations**:
- Low-resource languages (Santali, Sindhi, Dogri) have lower quality
- Very long documents (>512 tokens) may truncate (use long-context variant)
- Domain-specific terminology (medical, legal) benefits from fine-tuning

---

### Sarvam Translate API
| Field | Value |
|---|---|
| **Source** | Sarvam AI |
| **Model** | Mayura:v1 |
| **License** | Proprietary API (Apache 2.0 for local model weights) |
| **Languages** | 22 Indian languages |
| **Strengths** | Structured long-form text, formal tone |
| **Access** | Requires SARVAM_API_KEY |

---

## Text-to-Speech Models

### Indic Parler-TTS (AI4Bharat)
| Field | Value |
|---|---|
| **Source** | AI4Bharat / IIT Madras |
| **HuggingFace** | `ai4bharat/indic-parler-tts` |
| **License** | Apache 2.0 |
| **Architecture** | Parler-TTS (autoregressive TTS with voice conditioning) |
| **Languages** | 22 Indian languages |
| **Voices** | 44 speakers (male + female across languages) |
| **Size** | ~8GB |
| **Conditioning** | Text prompt describes voice style, pace, emotion |
| **Sample Rate** | 44.1 kHz |

**Voice Quality (MOS scores, human evaluation)**:

| Language | MOS (1-5) | Notes |
|---|---|---|
| Hindi | 4.1 | Good naturalness |
| Tamil | 3.9 | Slight robotic quality in some contexts |
| Telugu | 3.8 | Good for formal content |
| Bengali | 3.7 | Community feedback welcome |

**Known Limitations**:
- Long texts (>200 words) may have prosody breaks
- Emotional expressiveness limited (neutral/formal best)
- Very low-resource languages may sound unnatural

---

## Language Identification Models

### IndicLID (AI4Bharat)
| Field | Value |
|---|---|
| **Source** | AI4Bharat |
| **HuggingFace** | `ai4bharat/IndicLID` |
| **License** | MIT |
| **Architecture** | Transformer classifier |
| **Languages** | 47 languages including code-mixed varieties |
| **Accuracy** | >99% on 7 major Indian languages (single-language text) |

**Code-mix support**: Detects Hinglish, Tanglish, Banglish etc. as code-mixed.

---

## LLM Models (for NLU)

### Sarvam-M (24B)
| Field | Value |
|---|---|
| **Source** | Sarvam AI |
| **HuggingFace** | `sarvamai/sarvam-m` |
| **License** | Apache 2.0 |
| **Architecture** | Mistral Small fine-tuned with SFT + RLVR |
| **Parameters** | 24B |
| **Improvements vs base** | +20% Indian language benchmarks, +21.6% math, +17.6% coding |
| **Languages** | 10+ Indian languages + English |
| **Access** | Sarvam API (cloud) or self-hosted |

### Sarvam-1 (2B)
| Field | Value |
|---|---|
| **Source** | Sarvam AI |
| **HuggingFace** | `sarvamai/sarvam-1` |
| **License** | Apache 2.0 |
| **Parameters** | 2B |
| **Training** | 4 trillion tokens, 10 Indian languages |
| **Token efficiency** | 1.4–2.1 fertility (2–4× more efficient than Llama/Gemma for Indic) |
| **Best For** | On-device inference, low-memory servers |

### Sarvam 30B / 105B
| Field | Value |
|---|---|
| **Source** | Sarvam AI / IndiaAI Mission |
| **License** | Apache 2.0 |
| **Architecture** | 30B: dense MoE with GQA; 105B: MoE with MLA, 128 experts |
| **Context** | 128K tokens |
| **Indian Languages** | 22 |
| **Note** | GGUF support limited (Mar 2026); vLLM integration in progress |

---

## Citing These Models

If you use BharatVoiceAI in research, please cite:

```bibtex
@misc{indicwhisper2023,
  title={IndicWhisper: Multilingual ASR for Indian Languages},
  author={AI4Bharat},
  year={2023},
  url={https://github.com/AI4Bharat/vistaar}
}

@misc{indictrans2,
  title={IndicTrans2: Towards High-Quality and Accessible Machine Translation for all 22 Scheduled Indian Languages},
  author={AI4Bharat et al.},
  year={2023},
  url={https://github.com/AI4Bharat/IndicTrans2}
}

@misc{indicparlertts,
  title={Indic Parler-TTS},
  author={AI4Bharat},
  year={2024},
  url={https://huggingface.co/ai4bharat/indic-parler-tts}
}
```
