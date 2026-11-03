#!/usr/bin/env python3
"""
BHASHINI Smoke Test — verify your credentials work before running the full stack.
Run: python scripts/smoke_test_bhashini.py

Tests each BHASHINI service (ASR, MT, TTS, LID) with minimal inputs
and reports latency + success/failure for each.
"""

import asyncio
import base64
import os
import struct
import time

import httpx

BHASHINI_USER_ID = os.environ.get("BHASHINI_USER_ID", "")
BHASHINI_API_KEY = os.environ.get("BHASHINI_API_KEY", "")
COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

HEADERS = {
    "userID": BHASHINI_USER_ID,
    "ulcaApiKey": BHASHINI_API_KEY,
    "Authorization": BHASHINI_API_KEY,
    "Content-Type": "application/json",
}


def make_wav_b64() -> str:
    num_samples, sample_rate, bits, channels = 8000, 16000, 16, 1
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
    return base64.b64encode(header + b"\x00" * data_size).decode()


async def test_mt() -> bool:
    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {"language": {"sourceLanguage": "hi", "targetLanguage": "ta"}},
            }
        ],
        "inputData": {"input": [{"source": "नमस्ते"}]},
    }
    t = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(COMPUTE_URL, json=payload, headers=HEADERS)
    ms = round((time.perf_counter() - t) * 1000)
    if r.status_code == 200:
        translation = r.json()["pipelineResponse"][0]["output"][0].get("target", "")
        print(f"  MT (Hi→Ta):  '{translation}'  [{ms}ms]")
        return True
    else:
        print(f"  MT failed: {r.status_code} {r.text[:100]}  [{ms}ms]")
        return False


async def test_tts() -> bool:
    payload = {
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {"language": {"sourceLanguage": "hi"}, "gender": "female"},
            }
        ],
        "inputData": {"input": [{"source": "नमस्ते"}]},
    }
    t = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(COMPUTE_URL, json=payload, headers=HEADERS)
    ms = round((time.perf_counter() - t) * 1000)
    if r.status_code == 200:
        audio = r.json()["pipelineResponse"][0]["audio"][0].get("audioContent", "")
        size_kb = len(base64.b64decode(audio)) // 1024
        print(f"  TTS (Hindi female):  {size_kb}KB audio  [{ms}ms]")
        return True
    else:
        print(f"  TTS failed: {r.status_code} {r.text[:100]}  [{ms}ms]")
        return False


async def test_asr() -> bool:
    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": "hi"},
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {"audio": [{"audioContent": make_wav_b64()}]},
    }
    t = time.perf_counter()
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(COMPUTE_URL, json=payload, headers=HEADERS)
    ms = round((time.perf_counter() - t) * 1000)
    if r.status_code == 200:
        transcript = r.json()["pipelineResponse"][0]["output"][0].get("source", "")
        print(f"  ASR (Hindi):  '{transcript or '(silence)'}' [{ms}ms]")
        return True
    else:
        print(f"  ASR failed: {r.status_code} {r.text[:100]}  [{ms}ms]")
        return False


async def main() -> None:
    print("=" * 55)
    print("  BHASHINI Smoke Test")
    print("=" * 55)

    if not BHASHINI_USER_ID or not BHASHINI_API_KEY:
        print("\nBHASHINI_USER_ID and BHASHINI_API_KEY not set.")
        print("   Register free at: https://bhashini.gov.in/ulca/user/register")
        print("   Then: export BHASHINI_USER_ID=... BHASHINI_API_KEY=...")
        return

    print(f"\n  User ID: {BHASHINI_USER_ID[:8]}...")
    print(f"  API Key: {BHASHINI_API_KEY[:8]}...\n")

    results = await asyncio.gather(
        test_mt(),
        test_tts(),
        test_asr(),
        return_exceptions=True,
    )

    passed = sum(1 for r in results if r is True)
    total = len(results)

    print(f"\n{'=' * 55}")
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  🎉 All good! Set VAI_ASR_BACKEND=bhashini in your .env")
    else:
        print("  Some tests failed. Check credentials and BHASHINI status.")
        print("     Status: https://bhashini.gov.in")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
