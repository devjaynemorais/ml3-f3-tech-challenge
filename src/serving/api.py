"""FastAPI application factory and stable ``src.serving.api:app`` entrypoint."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from src.serving.metrics import PrometheusMiddleware
from src.serving.model_loader import TriagePredictor, load_predictor
from src.serving.routes import router
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)
PredictorLoader = Callable[[], TriagePredictor]


def _lifespan(
    predictor_loader: PredictorLoader,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build a lifespan context bound to one predictor loader."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.predictor = None
        try:
            application.state.predictor = predictor_loader()
            logger.info(
                "Medical classifier loaded (backend=%s)",
                application.state.predictor.backend,
            )
        except Exception:  # noqa: BLE001 - startup remains intentionally degraded
            logger.exception("Model loading failed; API started in degraded mode")
        yield
        application.state.predictor = None

    return lifespan


def create_app(predictor_loader: PredictorLoader = load_predictor) -> FastAPI:
    """Compose a FastAPI instance with isolated state, routes and middleware."""
    project = load_config().project
    application = FastAPI(
        title=project.name,
        description=(
            "Classifies public English medical abstracts as prioritization support; "
            "it is not a diagnostic system."
        ),
        version=project.version,
        lifespan=_lifespan(predictor_loader),
    )
    application.add_middleware(PrometheusMiddleware)
    application.include_router(router)
    return application


app = create_app()
