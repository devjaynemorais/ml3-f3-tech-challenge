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

    def explain(self, text: str) -> dict:
        """Return predict() fields plus the real preprocessing/attribution steps."""
        ...


def _ordered_scores(
    probabilities: np.ndarray,
    expected_classes: list[str],
) -> dict[str, float]:
    """Map multilabel probability columns to canonical response order."""
    if len(probabilities) != len(expected_classes):
        raise ValueError("probability columns do not match canonical labels")
    return dict(zip(expected_classes, probabilities.tolist(), strict=True))


def _label_from_scores(scores: dict[str, float]) -> str:
    """Return the highest-probability class."""
    return max(scores, key=scores.__getitem__)


def _ranked_terms(
    vector: sparse.spmatrix, weights: np.ndarray, names: np.ndarray, top_n: int
) -> list[dict]:
    """Rank the text's non-zero TF-IDF terms by contribution to the class score."""
    terms = [
        {
            "term": str(names[index]),
            "tfidf": float(vector[0, index]),
            "weight": float(weights[index]),
            "contribution": float(vector[0, index] * weights[index]),
        }
        for index in vector.nonzero()[1]
    ]
    terms.sort(key=lambda term: term["contribution"], reverse=True)
    return terms[:top_n]


def _term_contributions(
    vectorizer: object,
    classifier: object,
    preprocessed_text: str,
    label: str,
    top_n: int,
) -> list[dict] | None:
    """Explain a prediction via TF-IDF weight times linear-model coefficient.

    Returns ``None`` when the classifier has no ``coef_`` (e.g. tree ensembles),
    since term-level attribution only holds for linear decision functions.
    """
    estimators = getattr(classifier, "estimators_", None)
    if estimators is not None:
        label_index = getattr(classifier, "_canonical_labels", []).index(label)
        coefficients = getattr(estimators[label_index], "coef_", None)
    else:
        coefficients = getattr(classifier, "coef_", None)
    if coefficients is None or not hasattr(vectorizer, "get_feature_names_out"):
        return None
    vector = vectorizer.transform([preprocessed_text])
    weights = coefficients[0]
    names = vectorizer.get_feature_names_out()
    return _ranked_terms(vector, weights, names, top_n)


class SklearnPredictor:
    """Inference backend using the complete persisted sklearn Pipeline."""

    backend = "sklearn"

    def __init__(
        self,
        pipeline: Pipeline,
        expected_classes: list[str],
        threshold: float = 0.5,
    ) -> None:
        self._pipeline = pipeline
        self._expected_classes = expected_classes
        self._threshold = threshold
        classifier = pipeline.named_steps["classifier"]
        classifier._canonical_labels = expected_classes

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Predict with sklearn and return canonical ordered scores."""
        probabilities = np.asarray(self._pipeline.predict_proba([text])[0])
        scores = _ordered_scores(probabilities, self._expected_classes)
        return _label_from_scores(scores), scores

    def explain(self, text: str, top_n: int = 12) -> dict:
        """Predict and expose the real preprocessed text and term attribution."""
        label, scores = self.predict(text)
        labels = [item for item, score in scores.items() if score >= self._threshold]
        if label not in labels:
            labels.append(label)
        preprocessed = self._pipeline.named_steps["preprocessor"].transform([text])[0]
        vectorizer = self._pipeline.named_steps.get("tfidf")
        top_terms = None
        if vectorizer is not None:
            classifier = self._pipeline.named_steps["classifier"]
            top_terms = _term_contributions(
                vectorizer, classifier, preprocessed, label, top_n
            )
        return {
            "label": label,
            "labels": labels,
            "scores": scores,
            "backend": self.backend,
            "preprocessed_text": preprocessed,
            "top_terms": top_terms,
        }


class OnnxPredictor:
    """Inference backend using a sklearn feature prefix and ONNX classifier."""

    backend = "onnx"

    def __init__(
        self,
        feature_pipeline: Pipeline,
        session: InferenceSession,
        probability_classes: list[str],
        expected_classes: list[str],
        threshold: float = 0.5,
    ) -> None:
        self._feature_pipeline = feature_pipeline
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._expected_classes = expected_classes
        self._threshold = threshold
        validate_artifact_classes(probability_classes, expected_classes)

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Transform text once and infer canonical scores with ONNX Runtime."""
        features = self._feature_pipeline.transform([text])
        dense = features.toarray() if sparse.issparse(features) else features
        outputs = self._session.run(
            None, {self._input_name: np.asarray(dense, dtype=np.float32)}
        )
        probabilities = np.asarray(outputs[-1][0])
        scores = _ordered_scores(probabilities, self._expected_classes)
        return _label_from_scores(scores), scores

    def explain(self, text: str) -> dict:
        """Predict and expose the real preprocessed text.

        Term attribution is unavailable on this backend: the classifier runs
        inside the ONNX graph, not as an inspectable sklearn estimator.
        """
        label, scores = self.predict(text)
        labels = [item for item, score in scores.items() if score >= self._threshold]
        if label not in labels:
            labels.append(label)
        preprocessed = self._feature_pipeline.named_steps["preprocessor"].transform(
            [text]
        )[0]
        return {
            "label": label,
            "labels": labels,
            "scores": scores,
            "backend": self.backend,
            "preprocessed_text": preprocessed,
            "top_terms": None,
        }


def _load_sklearn_predictor(config: ExperimentConfig) -> SklearnPredictor:
    """Load and validate the configured sklearn pipeline."""
    pipeline = load_pipeline(
        Path(settings.model_artifacts_path), config.artifacts.pipeline_file
    )
    return SklearnPredictor(
        pipeline, config.data.labels, config.model.prediction_threshold
    )


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
    return OnnxPredictor(
        feature_pipeline,
        session,
        classes,
        config.data.labels,
        config.model.prediction_threshold,
    )


def load_predictor(backend: str | None = None) -> TriagePredictor:
    """Load the selected runtime backend and validate its artifact classes."""
    config = load_config()
    selected_backend = backend or settings.model_backend
    if selected_backend == "sklearn":
        return _load_sklearn_predictor(config)
    if selected_backend == "onnx":
        return _load_onnx_predictor(config)
    raise ValueError(f"unknown model backend: {selected_backend!r}")
