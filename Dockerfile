# Stage 1: builder — install production dependencies with Poetry
FROM python:3.11-slim AS builder

WORKDIR /app

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* ./

RUN poetry install --only main --no-root && rm -rf "${POETRY_CACHE_DIR}"

ARG SPACY_MODEL_URL=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
RUN .venv/bin/python -m nltk.downloader -d /app/.venv/nltk_data stopwords \
    && .venv/bin/python -m pip install --no-cache-dir "${SPACY_MODEL_URL}"


# Stage 2: train — jobs de treino/avaliação/otimização (usado pelo Airflow e
# pelo `docker compose run --rm train`)
FROM python:3.11-slim AS train

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/

ENV PATH="/app/.venv/bin:$PATH" \
    NLTK_DATA=/app/.venv/nltk_data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-m"]
CMD ["src.training.trainer"]


# Stage 3: api — FastAPI serving endpoint
FROM python:3.11-slim AS api

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY config/ config/

ENV PATH="/app/.venv/bin:$PATH" \
    NLTK_DATA=/app/.venv/nltk_data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
