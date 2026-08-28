"""Public FastAPI routes for metadata, health, prediction and metrics."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response

from src.serving.dependencies import get_predictor
from src.serving.metrics import PREDICTION_COUNT, metrics_response
from src.serving.model_loader import TriagePredictor
from src.serving.schemas import (
    HealthResponse,
    ServiceMetadata,
    TriageRequest,
    TriageResponse,
)
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ServiceMetadata)
def root() -> dict[str, str]:
    """Return configured service name and version."""
    project = load_config().project
    return {"service": project.name, "version": project.version}


@router.get("/health", response_model=HealthResponse, response_model_exclude_none=True)
def health(request: Request) -> dict[str, str | bool]:
    """Report loaded backend or the non-fatal degraded state."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True, "backend": predictor.backend}


@router.post("/predict", response_model=TriageResponse)
def predict(
    request: TriageRequest,
    predictor: Annotated[TriagePredictor, Depends(get_predictor)],
) -> dict:
    """Classify one abstract without logging its content."""
    try:
        label, scores = predictor.predict(request.text)
    except Exception as error:  # noqa: BLE001 - map backend failures to API contract
        logger.exception("Unexpected model inference failure")
        raise HTTPException(status_code=500, detail="Model inference failed") from error
    PREDICTION_COUNT.labels(predicted_label=label).inc()
    return {"label": label, "scores": scores, "backend": predictor.backend}


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in its text format."""
    return metrics_response()
