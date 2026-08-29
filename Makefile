.PHONY: env install nlp-resources lint format test test-cov \
        dataset train evaluate export-onnx benchmark-latency pipeline \
        api compose-build compose-up compose-down \
        airflow-up airflow-down

# `poetry` nem sempre está no PATH (ex.: Windows Store Python instala o script
# de entrada numa pasta de usuário que não entra no PATH automaticamente).
# `python -m poetry` não depende dessa resolução — só precisa que o pacote
# `poetry` esteja instalado no interpretador `python` corrente.
POETRY := python -m poetry

# ─── Ambiente ─────────────────────────────────────────────────────────────────

env:
	python -m ensurepip --upgrade
	python -m pip install --upgrade pip --quiet
	python -m pip install poetry==1.8.3 --quiet
	$(POETRY) install --with dev

SPACY_MODEL_URL := https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

install: env nlp-resources

nlp-resources:
	$(POETRY) run python -m nltk.downloader stopwords
	$(POETRY) run python -m pip install --no-cache-dir $(SPACY_MODEL_URL)

# ─── Qualidade de Código ──────────────────────────────────────────────────────

lint:
	$(POETRY) run ruff check src/ tests/ scripts/
	$(POETRY) run ruff format --check src/ tests/ scripts/

format:
	$(POETRY) run ruff check --fix src/ tests/ scripts/
	$(POETRY) run ruff format src/ tests/ scripts/

# ─── Testes ───────────────────────────────────────────────────────────────────

test:
	$(POETRY) run pytest tests/ -v

test-cov:
	$(POETRY) run pytest tests/ --cov=src --cov-report=html

# ─── Pipeline de ML (local, sem Docker/Airflow) ────────────────────────────────

# Valida o Medical Abstracts TC Corpus e persiste treino, validação e teste.
dataset:
	$(POETRY) run python -m src.data.make_dataset

train:
	$(POETRY) run python -m src.training.trainer

evaluate:
	$(POETRY) run python -m src.evaluation.evaluate

export-onnx:
	$(POETRY) run python -m src.optimization.export_onnx

benchmark-latency:
	$(POETRY) run python -m scripts.measure_latency

# Pipeline completo: dataset → treino → avaliação → export ONNX → benchmark
pipeline: dataset train evaluate export-onnx benchmark-latency

# ─── Serviço local ──────────────────────────────────────────────────────────────

API_PORT ?= 8000
api:
	$(POETRY) run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

# ─── Docker: API + Prometheus + Grafana ────────────────────────────────────────

compose-build:
	docker compose build

compose-up:
	docker compose up

compose-down:
	docker compose down

# ─── Docker: Airflow (orquestração de treino/retreino) ─────────────────────────

airflow-up:
	docker compose -f docker-compose.airflow.yml up -d

airflow-down:
	docker compose -f docker-compose.airflow.yml down
