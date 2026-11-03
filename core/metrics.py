"""Prometheus metrics for BharatVoiceAI."""

from core.config import settings

_counters: dict = {}
_histograms: dict = {}

if settings.PROMETHEUS_ENABLED:
    try:
        from prometheus_client import Counter, Gauge, Histogram

        API_CALLS = Counter(
            "vai_api_calls_total",
            "Total API calls",
            ["service", "backend", "success", "lang"],
        )
        API_LATENCY = Histogram(
            "vai_api_latency_ms",
            "API call latency in milliseconds",
            ["service"],
            buckets=[50, 100, 200, 500, 1000, 2000, 5000],
        )
        ACTIVE_REQUESTS = Gauge("vai_active_requests", "Active in-flight requests", ["service"])
        MODEL_WER = Gauge("vai_asr_wer", "ASR Word Error Rate on validation set", ["lang", "backend"])
        MODEL_BLEU = Gauge("vai_mt_bleu", "MT BLEU score on validation set", ["lang_pair", "backend"])

    except ImportError:
        pass


def record_api_call(service: str, backend: str, success: bool, lang: str = "unknown") -> None:
    if not settings.PROMETHEUS_ENABLED:
        return
    try:
        API_CALLS.labels(
            service=service,
            backend=backend,
            success=str(success).lower(),
            lang=lang,
        ).inc()
    except Exception:
        pass


def record_latency(service: str, latency_ms: float) -> None:
    if not settings.PROMETHEUS_ENABLED:
        return
    try:
        API_LATENCY.labels(service=service).observe(latency_ms)
    except Exception:
        pass
