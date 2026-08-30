.PHONY: env install nlp-resources lint format test test-cov \
        dataset train evaluate promote export-onnx analyze-corpus benchmark-latency pipeline \
        api mlflow compose-build compose-up compose-down \
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
# scispaCy — vetores biomedicos usados pela Strategy features.type=embeddings
SCISPACY_MODEL_URL := https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

install: env nlp-resources

nlp-resources:
	$(POETRY) run python -m nltk.downloader stopwords
	$(POETRY) run python -m pip install --no-cache-dir $(SPACY_MODEL_URL)
	$(POETRY) run python -m pip install --no-cache-dir $(SCISPACY_MODEL_URL)

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

# Registra o melhor run (por metrics.<registry.metric> no MLflow) no Model
# Registry e promove para registry.stage. Requer MLflow acessível — rode
# depois de make train + make evaluate.
promote:
	$(POETRY) run python -m src.models.promote

export-onnx:
	$(POETRY) run python -m src.optimization.export_onnx

# Estrutura multi-rotulo do corpus e teto teorico de F1-macro (docs/plano_melhoria_f1.md)
analyze-corpus:
	$(POETRY) run python -m scripts.analyze_corpus

benchmark-latency:
	$(POETRY) run python -m scripts.measure_latency

# Pipeline completo: dataset → treino → avaliação → export ONNX → benchmark
pipeline: dataset train evaluate export-onnx benchmark-latency

# ─── Serviço local ──────────────────────────────────────────────────────────────

API_PORT ?= 8000
api:
	$(POETRY) run uvicorn src.serving.api:app \
		--host 0.0.0.0 --port $(API_PORT) --reload

# MLflow local (sem Docker) — abre na hora, sem pull/build de imagem.
# Porta parametrizável: make mlflow MLFLOW_PORT=5001 (ajuste MLFLOW_TRACKING_URI no .env)
MLFLOW_PORT ?= 5000
mlflow:
	$(POETRY) run mlflow server \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

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
