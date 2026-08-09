"""API FastAPI de triagem automática de laudos médicos.

Endpoints:
  - GET  /         → metadados do serviço
  - GET  /health   → estado de carregamento do modelo
  - POST /predict  → classifica um laudo em normal / atencao / urgente
  - GET  /metrics  → métricas Prometheus (contagem de requisições e latência)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from src.serving.metrics import PREDICTION_COUNT, PrometheusMiddleware, metrics_response
from src.serving.model_loader import TriagePredictor, load_predictor
from src.serving.schemas import TriageRequest, TriageResponse

logger = logging.getLogger(__name__)

state: dict[str, TriagePredictor | None] = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo de triagem uma única vez no startup da API."""
    state["predictor"] = None
    try:
        state["predictor"] = load_predictor()
        logger.info(
            "Modelo de triagem carregado (backend=%s)", state["predictor"].backend
        )
    except Exception:  # noqa: BLE001 — API sobe em modo degradado
        logger.exception("Falha ao carregar o modelo; /health reportará degradado.")
    yield


app = FastAPI(
    title="Triage Classifier API",
    description="Classificação de urgência de laudos médicos "
    "(normal / atencao / urgente).",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(PrometheusMiddleware)


def _get_predictor() -> TriagePredictor:
    """Retorna o predictor carregado ou 503 se o modelo ainda não existe."""
    predictor = state["predictor"]
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Rode `make train` (e `make export-onnx` "
            "se model_backend=onnx) antes de usar a API.",
        )
    return predictor


@app.get("/")
def root() -> dict:
    """Metadados do serviço."""
    return {"service": "Triage Classifier API", "version": "0.1.0"}


@app.get("/health")
def health() -> dict:
    """Estado de carregamento do modelo."""
    predictor = state["predictor"]
    if predictor is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "backend": predictor.backend}


@app.post("/predict", response_model=TriageResponse)
def predict(request: TriageRequest) -> dict:
    """Classifica o texto do laudo em normal / atencao / urgente."""
    predictor = _get_predictor()
    label, scores = predictor.predict(request.text)
    PREDICTION_COUNT.labels(predicted_label=label).inc()
    return {"label": label, "scores": scores, "backend": predictor.backend}


@app.get("/metrics")
def metrics() -> Response:
    """Expõe as métricas no formato de texto do Prometheus."""
    return metrics_response()
