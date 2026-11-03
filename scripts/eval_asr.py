#!/usr/bin/env python3
"""
ASR Evaluation Script — computes Word Error Rate (WER) on standard benchmarks.
Supports: IITM Hindi, Vistaar, MUCS, FLEURS, CommonVoice.

Usage:
    python scripts/eval_asr.py --backend indicwhisper --lang hi --dataset vistaar
    python scripts/eval_asr.py --backend bhashini --lang ta --max-samples 100

Output:
    - WER per language/domain
    - Comparison table across backends
    - JSON report saved to eval_results/asr_{backend}_{lang}_{timestamp}.json

References:
    - Vistaar: https://github.com/AI4Bharat/vistaar (59 benchmarks, 12 languages)
    - MUCS: https://navana-tech.github.io/MUCS2021/
    - FLEURS: https://huggingface.co/datasets/google/fleurs
"""

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from datasets import load_dataset
    from jiwer import cer, wer
    from tqdm import tqdm
except ImportError:
    print("Install eval extras: pip install 'bharatvoiceai[eval]'")
    raise

# Dataset configs per language
DATASET_CONFIGS = {
    "vistaar": {
        "hi": {"hf_path": "ai4bharat/vistaar", "config": "hindi", "split": "test"},
        "ta": {"hf_path": "ai4bharat/vistaar", "config": "tamil", "split": "test"},
        "te": {"hf_path": "ai4bharat/vistaar", "config": "telugu", "split": "test"},
        "bn": {"hf_path": "ai4bharat/vistaar", "config": "bengali", "split": "test"},
        "mr": {"hf_path": "ai4bharat/vistaar", "config": "marathi", "split": "test"},
        "gu": {"hf_path": "ai4bharat/vistaar", "config": "gujarati", "split": "test"},
        "kn": {"hf_path": "ai4bharat/vistaar", "config": "kannada", "split": "test"},
        "ml": {"hf_path": "ai4bharat/vistaar", "config": "malayalam", "split": "test"},
        "pa": {"hf_path": "ai4bharat/vistaar", "config": "punjabi", "split": "test"},
        "or": {"hf_path": "ai4bharat/vistaar", "config": "odia", "split": "test"},
        "as": {"hf_path": "ai4bharat/vistaar", "config": "assamese", "split": "test"},
        "ur": {"hf_path": "ai4bharat/vistaar", "config": "urdu", "split": "test"},
    },
    "fleurs": {
        "hi": {"hf_path": "google/fleurs", "config": "hi_in", "split": "test"},
        "ta": {"hf_path": "google/fleurs", "config": "ta_in", "split": "test"},
        "te": {"hf_path": "google/fleurs", "config": "te_in", "split": "test"},
        "bn": {"hf_path": "google/fleurs", "config": "bn_in", "split": "test"},
    },
    "commonvoice": {
        "hi": {"hf_path": "mozilla-foundation/common_voice_16_1", "config": "hi", "split": "test"},
        "ta": {"hf_path": "mozilla-foundation/common_voice_16_1", "config": "ta", "split": "test"},
    },
}


async def evaluate_backend(
    backend_name: str,
    lang: str,
    dataset_name: str,
    max_samples: int | None = None,
    output_dir: Path = Path("eval_results"),
) -> dict:
    """Run WER evaluation on a dataset."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Backend: {backend_name} | Lang: {lang} | Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    # Load dataset
    if dataset_name not in DATASET_CONFIGS or lang not in DATASET_CONFIGS[dataset_name]:
        print(f"Dataset {dataset_name}/{lang} not configured. Using synthetic test data.")
        samples = _synthetic_samples(lang)
    else:
        cfg = DATASET_CONFIGS[dataset_name][lang]
        print(f"Loading {cfg['hf_path']} ({cfg['config']})...")
        ds = load_dataset(cfg["hf_path"], cfg["config"], split=cfg["split"], trust_remote_code=True)
        samples = list(ds)
        if max_samples:
            samples = samples[:max_samples]

    print(f"Evaluating {len(samples)} samples...")

    # Initialize backend
    from services.asr.dispatcher import ASRDispatcher

    dispatcher = ASRDispatcher(default_backend=backend_name)

    references = []
    hypotheses = []
    latencies = []
    errors = 0

    for sample in tqdm(samples, desc=f"Transcribing ({lang})"):
        ref_text = _get_reference(sample)
        audio_bytes = _get_audio_bytes(sample)

        if not audio_bytes or not ref_text:
            continue

        import base64

        from api.schemas.models import ASRRequest

        req = ASRRequest(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            lang=lang,
            consent_given=True,
            backend=backend_name,
        )

        try:
            t0 = time.perf_counter()
            result = await dispatcher.transcribe(req)
            latency_ms = (time.perf_counter() - t0) * 1000

            references.append(ref_text)
            hypotheses.append(result.transcript)
            latencies.append(latency_ms)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"Error: {e}")

    if not references:
        print("No samples processed.")
        return {}

    # Compute metrics
    word_err_rate = wer(references, hypotheses)
    char_err_rate = cer(references, hypotheses)
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    result = {
        "backend": backend_name,
        "lang": lang,
        "dataset": dataset_name,
        "num_samples": len(references),
        "errors": errors,
        "wer": round(word_err_rate * 100, 2),
        "cer": round(char_err_rate * 100, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "timestamp": datetime.now().isoformat(),
    }

    print("\nResults:")
    print(f"  WER:           {result['wer']}%  (lower is better, target ≤15%)")
    print(f"  CER:           {result['cer']}%")
    print(f"  Avg latency:   {result['avg_latency_ms']}ms")
    print(f"  P95 latency:   {result['p95_latency_ms']}ms")
    print(f"  Samples:       {result['num_samples']}")
    print(f"  Errors:        {result['errors']}")

    # Save report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"asr_{backend_name}_{lang}_{dataset_name}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {out_path}")

    return result


def _get_reference(sample: dict) -> str:
    for key in ["sentence", "text", "transcription", "normalized_text", "raw_transcription"]:
        if sample.get(key):
            return str(sample[key]).strip()
    return ""


def _get_audio_bytes(sample: dict) -> bytes | None:
    try:
        import io

        import numpy as np
        import soundfile as sf

        if sample.get("audio"):
            audio_data = sample["audio"]
            array = audio_data.get("array", np.zeros(1600, dtype=np.float32))
            sr = audio_data.get("sampling_rate", 16000)
            buf = io.BytesIO()
            sf.write(buf, array, sr, format="WAV")
            return buf.getvalue()
    except Exception:
        pass
    return None


def _synthetic_samples(lang: str) -> list[dict]:
    """Generate minimal synthetic test samples for smoke testing."""
    import io

    import numpy as np
    import soundfile as sf

    samples = []
    silence = np.zeros(16000, dtype=np.float32)  # 1 sec silence
    buf = io.BytesIO()
    sf.write(buf, silence, 16000, format="WAV")

    for _ in range(5):
        samples.append(
            {
                "audio": {"array": silence, "sampling_rate": 16000},
                "sentence": "test",
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASR backend WER")
    parser.add_argument(
        "--backend",
        default="indicwhisper",
        choices=["indicwhisper", "bhashini", "wav2vec2"],
        help="ASR backend to evaluate",
    )
    parser.add_argument(
        "--lang",
        default="hi",
        help="Language code (hi, ta, te, bn, mr, gu, kn, ml, pa, or, as, ur)",
    )
    parser.add_argument(
        "--dataset",
        default="vistaar",
        choices=["vistaar", "fleurs", "commonvoice"],
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test samples (useful for quick checks)",
    )
    parser.add_argument("--all-langs", action="store_true", help="Evaluate all supported languages")
    parser.add_argument("--output-dir", default="eval_results", help="Directory to save evaluation reports")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.all_langs:
        langs = list(DATASET_CONFIGS.get(args.dataset, {}).keys())
        results = []
        for lang in langs:
            result = asyncio.run(evaluate_backend(args.backend, lang, args.dataset, args.max_samples, output_dir))
            if result:
                results.append(result)

        # Print summary table
        if results:
            print("\n\nSUMMARY TABLE")
            print(f"{'Lang':<6} {'WER%':<8} {'CER%':<8} {'Avg ms':<10} {'P95 ms':<10}")
            print("-" * 50)
            for r in sorted(results, key=lambda x: x["wer"]):
                print(f"{r['lang']:<6} {r['wer']:<8} {r['cer']:<8} {r['avg_latency_ms']:<10} {r['p95_latency_ms']:<10}")
    else:
        asyncio.run(evaluate_backend(args.backend, args.lang, args.dataset, args.max_samples, output_dir))


if __name__ == "__main__":
    main()
