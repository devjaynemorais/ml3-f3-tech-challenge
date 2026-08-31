"""Public FastAPI routes for metadata, health, prediction and metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from src.serving.demo_examples import SAMPLE_TEXTS
from src.serving.dependencies import get_predictor
from src.serving.metrics import PREDICTION_COUNT, metrics_response
from src.serving.model_loader import TriagePredictor
from src.serving.schemas import (
    ExplainResponse,
    HealthResponse,
    SampleText,
    ServiceMetadata,
    TriageRequest,
    TriageResponse,
)
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
router = APIRouter()
_DEMO_HTML_PATH = Path(__file__).parent / "static" / "demo.html"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_COMPARISON_PATH = _PROJECT_ROOT / "metrics" / "experiment_comparison.json"
_LATENCY_COMPARISON_PATH = _PROJECT_ROOT / "metrics" / "latency_comparison.json"
_PROMOTED_MODEL_PATH = _PROJECT_ROOT / "models" / "promoted_model.json"


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
        logger.error("Unexpected model inference failure (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Model inference failed") from None
    threshold = load_config().model.prediction_threshold
    labels = [item for item, score in scores.items() if score >= threshold]
    if label not in labels:
        labels.append(label)
    for selected_label in labels:
        PREDICTION_COUNT.labels(predicted_label=selected_label).inc()
    return {
        "label": label,
        "labels": labels,
        "scores": scores,
        "backend": predictor.backend,
    }


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics in its text format."""
    return metrics_response()


@router.get("/explain", response_model=ExplainResponse)
def explain(
    text: Annotated[str, Query(min_length=1)],
    predictor: Annotated[TriagePredictor, Depends(get_predictor)],
) -> dict:
    """Classify with the real preprocessing and term-attribution steps exposed.

    Demo-only: powers the interactive page at ``/demo``, not the product client.
    """
    try:
        return predictor.explain(text.strip())
    except Exception as error:  # noqa: BLE001 - map backend failures to API contract
        logger.error("Unexpected explain failure (%s)", type(error).__name__)
        raise HTTPException(status_code=500, detail="Model inference failed") from None


@router.get("/demo/sample-texts", response_model=list[SampleText])
def sample_texts() -> list[dict[str, str]]:
    """Return real example abstracts, one per canonical category."""
    return SAMPLE_TEXTS


@router.get("/demo/experiment-results")
def experiment_results() -> dict:
    """Return persisted model comparisons and the promotion rationale."""
    try:
        comparison = json.loads(_EXPERIMENT_COMPARISON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=503,
            detail="Experiment comparison is not available; run make experiments first",
        ) from None

    registry = load_config().registry
    candidates = []
    for key, result in comparison.items():
        test = result.get("test", {})
        candidates.append(
            {
                "key": key,
                "label": result.get("label", key),
                "run_id": result.get("mlflow_run_id"),
                "cv_macro_f1_mean": result.get("cv_macro_f1_mean"),
                "cv_macro_f1_std": result.get("cv_macro_f1_std"),
                "test_macro_f1": test.get("macro_avg", {}).get("f1"),
                "test_accuracy": test.get("accuracy"),
                "ml_accuracy": test.get("jaccard_samples"),
                "minority_recall": test.get("minority_class_recall_mean"),
            }
        )

    eligible = [item for item in candidates if item["cv_macro_f1_mean"] is not None]
    if not eligible:
        raise HTTPException(status_code=503, detail="No comparable experiment results")
    winner = sorted(
        eligible,
        key=lambda item: (
            -item["cv_macro_f1_mean"],
            item["cv_macro_f1_std"]
            if item["cv_macro_f1_std"] is not None
            else float("inf"),
        ),
    )[0]

    promoted = None
    try:
        promoted = json.loads(_PROMOTED_MODEL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "candidates": candidates,
        "winner_key": winner["key"],
        "selection": {
            "primary_metric": registry.metric,
            "primary_direction": "ascending" if registry.ascending else "descending",
            "tiebreak_metric": registry.tiebreak_metric,
            "tiebreak_direction": (
                "ascending" if registry.tiebreak_ascending else "descending"
            ),
            "latency_used": False,
        },
        "promoted": promoted,
    }


@router.get("/demo/latency-comparison")
def latency_comparison() -> dict:
    """Return the persisted sklearn-vs-ONNX latency benchmark."""
    try:
        return json.loads(_LATENCY_COMPARISON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise HTTPException(
            status_code=503,
            detail="Latency comparison unavailable; run make benchmark-latency first",
        ) from None


@router.get("/demo", response_class=HTMLResponse)
def demo() -> str:
    """Serve the self-contained interactive demo page (no build step)."""
    return _DEMO_HTML_PATH.read_text(encoding="utf-8")
