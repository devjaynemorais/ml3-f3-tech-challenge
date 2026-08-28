"""FastAPI dependencies backed by application-local state."""

from typing import cast

from fastapi import HTTPException, Request

from src.serving.model_loader import TriagePredictor


def get_predictor(request: Request) -> TriagePredictor:
    """Return the app predictor or report a degraded service as HTTP 503."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded; run `make train` and retry.",
        )
    return cast(TriagePredictor, predictor)
