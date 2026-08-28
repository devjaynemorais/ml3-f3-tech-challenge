"""Load validated sklearn or ONNX medical-text predictors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import joblib
import numpy as np
from scipy import sparse
from sklearn.pipeline import Pipeline

from src.config.settings import settings
from src.models.artifact_contract import validate_artifact_classes
from src.models.registry import load_pipeline
from src.utils.config_loader import ExperimentConfig, load_config

if TYPE_CHECKING:
    from onnxruntime import InferenceSession


class TriagePredictor(Protocol):
    """Common contract for sklearn and ONNX inference backends."""

    backend: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Return predicted label and probability per canonical class."""
        ...


def _ordered_scores(
    probabilities: np.ndarray,
    probability_classes: list[str],
    expected_classes: list[str],
) -> dict[str, float]:
    """Map probability output to canonical response order."""
    by_class = dict(zip(probability_classes, probabilities.tolist(), strict=True))
    return {label: float(by_class[label]) for label in expected_classes}


def _label_from_scores(scores: dict[str, float]) -> str:
    """Return the highest-probability class."""
    return max(scores, key=scores.__getitem__)


class SklearnPredictor:
    """Inference backend using the complete persisted sklearn Pipeline."""

    backend = "sklearn"

    def __init__(self, pipeline: Pipeline, expected_classes: list[str]) -> None:
        self._pipeline = pipeline
        self._expected_classes = expected_classes
        self._probability_classes = validate_artifact_classes(
            list(pipeline.classes_), expected_classes
        )

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict with sklearn and return canonical ordered scores."""
        probabilities = np.asarray(self._pipeline.predict_proba([text])[0])
        scores = _ordered_scores(
            probabilities, self._probability_classes, self._expected_classes
        )
        return _label_from_scores(scores), scores


class OnnxPredictor:
    """Inference backend using a sklearn feature prefix and ONNX classifier."""

    backend = "onnx"

    def __init__(
        self,
        feature_pipeline: Pipeline,
        session: InferenceSession,
        probability_classes: list[str],
        expected_classes: list[str],
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._expected_classes = expected_classes
        self._probability_classes = validate_artifact_classes(
            probability_classes, expected_classes
        )

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Transform text once and infer canonical scores with ONNX Runtime."""
        features = self._feature_pipeline.transform([text])
        dense = features.toarray() if sparse.issparse(features) else features
        outputs = self._session.run(
            None, {self._input_name: np.asarray(dense, dtype=np.float32)}
        )
        probabilities = np.asarray(outputs[-1][0])
        scores = _ordered_scores(
            probabilities, self._probability_classes, self._expected_classes
        )
        return _label_from_scores(scores), scores


def _load_sklearn_predictor(config: ExperimentConfig) -> SklearnPredictor:
    """Load and validate the configured sklearn pipeline."""
    pipeline = load_pipeline(
        Path(settings.model_artifacts_path), config.artifacts.pipeline_file
    )
    return SklearnPredictor(pipeline, config.data.labels)


def _load_onnx_predictor(config: ExperimentConfig) -> OnnxPredictor:
    """Load and validate the configured ONNX artifact set."""
    import onnxruntime as ort

    path = Path(settings.model_onnx_path)
    feature_pipeline = joblib.load(path / config.artifacts.feature_pipeline_file)
    if not isinstance(feature_pipeline, Pipeline):
        raise TypeError("ONNX feature artifact is not an sklearn Pipeline")
    classes = json.loads(
        (path / config.artifacts.classes_file).read_text(encoding="utf-8")
    )
    session = ort.InferenceSession(
        str(path / config.artifacts.onnx_file), providers=["CPUExecutionProvider"]
    )
    return OnnxPredictor(feature_pipeline, session, classes, config.data.labels)


def load_predictor(backend: str | None = None) -> TriagePredictor:
    """Load the selected runtime backend and validate its artifact classes."""
    config = load_config()
    selected_backend = backend or settings.model_backend
    if selected_backend == "sklearn":
        return _load_sklearn_predictor(config)
    if selected_backend == "onnx":
        return _load_onnx_predictor(config)
    raise ValueError(f"unknown model backend: {selected_backend!r}")
