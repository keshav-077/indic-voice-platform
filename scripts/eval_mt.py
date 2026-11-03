#!/usr/bin/env python3
"""
MT Evaluation Script — BLEU / ChrF on FLORES-200 and IN22 benchmarks.

Usage:
    python scripts/eval_mt.py --backend indictrans2 --src en --tgt hi
    python scripts/eval_mt.py --backend indictrans2 --all-pairs --dataset flores200
    python scripts/eval_mt.py --compare  # compare all backends on key pairs

References:
    - FLORES-200: https://huggingface.co/datasets/facebook/flores
    - IN22: https://huggingface.co/datasets/ai4bharat/IN22-Gen
      (AI4Bharat's benchmark: IN22-Gen + IN22-Conv, 22 Indic languages)
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

try:
    import sacrebleu
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Install eval extras: pip install 'bharatvoiceai[eval]'")
    raise

# FLORES-200 language codes mapping (VAI lang → FLORES split name)
FLORES_LANG_MAP = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "en": "eng_Latn",
    "mai": "mai_Deva",
    "sa": "san_Deva",
    "ne": "npi_Deva",
}

# Key language pairs to evaluate (covers most high-impact use cases)
KEY_PAIRS = [
    ("en", "hi"),
    ("en", "ta"),
    ("en", "te"),
    ("en", "bn"),
    ("hi", "en"),
    ("ta", "en"),
    ("hi", "ta"),
    ("hi", "bn"),
]


async def evaluate_mt(
    backend_name: str,
    src_lang: str,
    tgt_lang: str,
    dataset: str = "flores200",
    max_samples: int | None = 200,
    output_dir: Path = Path("eval_results"),
) -> dict:
    """Evaluate MT backend BLEU/ChrF on a standard benchmark."""
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Backend: {backend_name} | {src_lang}→{tgt_lang} | Dataset: {dataset}")
    print(f"{'=' * 60}")

    sources, references = _load_dataset(dataset, src_lang, tgt_lang, max_samples)

    if not sources:
        print("No samples loaded. Check dataset availability.")
        return {}

    print(f"Translating {len(sources)} sentences...")

    from api.schemas.models import MTRequest
    from services.mt.dispatcher import MTDispatcher

    dispatcher = MTDispatcher(default_backend=backend_name)
    hypotheses = []
    latencies = []
    errors = 0

    for src_text in tqdm(sources, desc=f"{src_lang}→{tgt_lang}"):
        req = MTRequest(text=src_text, src_lang=src_lang, tgt_lang=tgt_lang)
        try:
            import time

            t0 = time.perf_counter()
            result = await dispatcher.translate(req)
            latencies.append((time.perf_counter() - t0) * 1000)
            hypotheses.append(result.translation)
        except Exception as e:
            errors += 1
            hypotheses.append("")
            if errors <= 3:
                print(f"Error: {e}")

    if not hypotheses:
        return {}

    # Compute BLEU (sacreBLEU)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    ter = sacrebleu.corpus_ter(hypotheses, [references])

    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    result = {
        "backend": backend_name,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "dataset": dataset,
        "num_samples": len(sources),
        "errors": errors,
        "bleu": round(bleu.score, 2),
        "chrf": round(chrf.score, 2),
        "ter": round(ter.score, 2),
        "avg_latency_ms": round(avg_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "timestamp": datetime.now().isoformat(),
    }

    print("\nResults:")
    print(f"  BLEU:          {result['bleu']}  (target ≥30 for En↔Hi, ≥20 for others)")
    print(f"  ChrF:          {result['chrf']}")
    print(f"  TER:           {result['ter']}  (lower is better)")
    print(f"  Avg latency:   {result['avg_latency_ms']}ms")
    print(f"  P95 latency:   {result['p95_latency_ms']}ms")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"mt_{backend_name}_{src_lang}_{tgt_lang}_{dataset}_{ts}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    return result


def _load_dataset(dataset: str, src_lang: str, tgt_lang: str, max_samples: int | None):
    """Load sentence pairs from the benchmark dataset."""
    src_flores = FLORES_LANG_MAP.get(src_lang)
    tgt_flores = FLORES_LANG_MAP.get(tgt_lang)

    if not src_flores or not tgt_flores:
        print(f"Language pair {src_lang}/{tgt_lang} not in FLORES map. Using synthetic data.")
        return ["Hello, how are you?"] * 5, ["नमस्ते, आप कैसे हैं?"] * 5

    try:
        if dataset == "flores200":
            src_ds = load_dataset("facebook/flores", src_flores, split="devtest", trust_remote_code=True)
            tgt_ds = load_dataset("facebook/flores", tgt_flores, split="devtest", trust_remote_code=True)
            src_texts = [s["sentence"] for s in src_ds]
            tgt_texts = [s["sentence"] for s in tgt_ds]
        elif dataset == "in22":
            ds = load_dataset("ai4bharat/IN22-Gen", split="gen", trust_remote_code=True)
            src_texts = [s[f"sentence_{src_flores}"] for s in ds]
            tgt_texts = [s[f"sentence_{tgt_flores}"] for s in ds]
        else:
            return [], []

        if max_samples:
            src_texts = src_texts[:max_samples]
            tgt_texts = tgt_texts[:max_samples]

        return src_texts, tgt_texts
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return ["Hello world"] * 5, ["नमस्ते दुनिया"] * 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MT BLEU scores")
    parser.add_argument("--backend", default="indictrans2", choices=["indictrans2", "bhashini", "sarvam"])
    parser.add_argument("--src", default="en", help="Source language code")
    parser.add_argument("--tgt", default="hi", help="Target language code")
    parser.add_argument("--dataset", default="flores200", choices=["flores200", "in22"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--all-pairs", action="store_true", help="Evaluate all key language pairs")
    parser.add_argument("--compare", action="store_true", help="Compare all backends on key pairs")
    parser.add_argument("--output-dir", default="eval_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pairs = KEY_PAIRS if (args.all_pairs or args.compare) else [(args.src, args.tgt)]
    backends = ["indictrans2", "bhashini"] if args.compare else [args.backend]

    all_results = []
    for backend in backends:
        for src, tgt in pairs:
            result = asyncio.run(evaluate_mt(backend, src, tgt, args.dataset, args.max_samples, output_dir))
            if result:
                all_results.append(result)

    if len(all_results) > 1:
        print("\n\nSUMMARY")
        print(f"{'Backend':<14} {'Pair':<10} {'BLEU':<8} {'ChrF':<8} {'ms':<8}")
        print("-" * 55)
        for r in sorted(all_results, key=lambda x: (-x["bleu"], x["src_lang"])):
            pair = f"{r['src_lang']}→{r['tgt_lang']}"
            print(f"{r['backend']:<14} {pair:<10} {r['bleu']:<8} {r['chrf']:<8} {r['avg_latency_ms']:<8}")


if __name__ == "__main__":
    main()
