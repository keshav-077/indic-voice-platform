# BharatVoiceAI — Developer Makefile
# Usage: make <target>

.PHONY: help install dev test lint format docker-up docker-down eval-asr eval-mt clean

## Show this help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Install in development mode
install:
	pip install -e ".[dev]"
	pre-commit install

## Install with all model backends (requires 8GB+ RAM)
install-models:
	pip install -e ".[dev,models]"

## Run the API in development mode (auto-reload)
dev:
	uvicorn api.main:app --reload --port 8000 --log-level info

## Run unit tests
test:
	pytest tests/unit/ -v --tb=short

## Run all tests with coverage
test-cov:
	pytest tests/ -v --cov --cov-report=term-missing --cov-report=html

## Lint with ruff
lint:
	ruff check . --output-format=concise
	ruff format --check .

## Auto-fix lint issues
format:
	ruff check . --fix
	ruff format .

## Start full local stack
docker-up:
	docker compose up -d
	@echo "Stack running:"
	@echo "   API:        http://localhost:8000"
	@echo "   API docs:   http://localhost:8000/docs"
	@echo "   Grafana:    http://localhost:3000 (admin/admin)"
	@echo "   Prometheus: http://localhost:9090"

## Stop all containers
docker-down:
	docker compose down

## Build Docker image
docker-build:
	docker build -f infra/docker/Dockerfile.api -t bharatvoiceai:latest .

## Evaluate ASR (requires model installed)
eval-asr:
	python scripts/eval_asr.py --backend $(BACKEND) --lang $(LANG) --dataset vistaar --max-samples 100

## Evaluate MT (requires model installed)
eval-mt:
	python scripts/eval_mt.py --backend $(BACKEND) --src $(SRC) --tgt $(TGT) --max-samples 100

## Run BHASHINI smoke test
smoke-bhashini:
	python scripts/smoke_test_bhashini.py

## Generate admin token (for testing)
token:
	@python -c "\
from api.middleware.auth import create_access_token; \
print('Token:', create_access_token('dev-user', ['api']))"

## Show project structure
tree:
	find . -not -path './.venv/*' -not -path './.git/*' \
	  -not -path './__pycache__/*' -not -path './dist/*' \
	  -not -name '*.pyc' -not -name '*.egg-info' \
	  | sort | head -80

## Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/ htmlcov/ coverage/
	rm -f test.db

## Check if BHASHINI is configured
check-config:
	@python -c "\
from core.config import settings; \
print('BHASHINI configured:', settings.is_bhashini_configured); \
print('Sarvam configured:', settings.is_sarvam_configured); \
print('ASR backend:', settings.VAI_ASR_BACKEND); \
print('MT backend:', settings.VAI_MT_BACKEND); \
print('Audio logging:', settings.VAI_AUDIO_LOGGING); \
print('Consent required:', settings.CONSENT_REQUIRED)"
