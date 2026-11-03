"""Structured logging setup."""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # Suppress noisy third-party loggers
    for noisy in ["httpx", "httpcore", "transformers", "torch"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
