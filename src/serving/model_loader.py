"""Carrega o classificador de triagem no backend configurado (sklearn ou ONNX)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import joblib
import numpy as np

from src.config.settings import settings
from src.utils.config_loader import load_config

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class TriagePredictor(Protocol):
    """Interface comum entre o backend sklearn e o backend ONNX."""

    backend: str

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Retorna (rótulo previsto, probabilidades por classe)."""
        ...


class SklearnPredictor:
    """Backend de inferência usando o Pipeline scikit-learn completo."""

    backend = "sklearn"

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._classes = list(pipeline.classes_)

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Retorna (rótulo previsto, probabilidades por classe)."""
        proba = self._pipeline.predict_proba([text])[0]
        scores = dict(zip(self._classes, proba.tolist(), strict=True))
        label = self._classes[int(np.argmax(proba))]
        return label, scores


class OnnxPredictor:
    """Backend de inferência com TF-IDF (sklearn) + classificador via ONNX Runtime."""

    backend = "onnx"

    def __init__(
        self, vectorizer: TfidfVectorizer, session: InferenceSession, classes: list[str]
    ) -> None:
        self._vectorizer = vectorizer
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._classes = classes

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Retorna (rótulo previsto, probabilidades por classe)."""
        vector = self._vectorizer.transform([text]).toarray().astype(np.float32)
        _label_out, proba_out = self._session.run(None, {self._input_name: vector})
        proba = np.asarray(proba_out[0])
        scores = dict(zip(self._classes, proba.tolist(), strict=True))
        label = self._classes[int(np.argmax(proba))]
        return label, scores


def load_predictor(backend: str | None = None) -> TriagePredictor:
    """Instancia o predictor no backend pedido (ou `settings.model_backend`).

    Os nomes de arquivo dos artefatos (`pipeline_file`, `onnx_file`, ...) vêm do
    `config.yaml` — é o mesmo config que `src.training.trainer` e
    `src.optimization.export_onnx` usam para *salvar* esses artefatos. Já os
    diretórios onde procurar (`settings.model_artifacts_path` /
    `settings.model_onnx_path`) são configuráveis por ambiente, para permitir
    apontar o serving para um volume/local diferente do treino sem editar o
    config.yaml.
    """
    cfg = load_config()
    backend = backend or settings.model_backend

    if backend == "sklearn":
        from src.models.registry import load_pipeline

        pipeline = load_pipeline(
            Path(settings.model_artifacts_path), cfg["artifacts"]["pipeline_file"]
        )
        return SklearnPredictor(pipeline)

    if backend == "onnx":
        import onnxruntime as ort

        onnx_path = Path(settings.model_onnx_path)
        onnx_file = onnx_path / cfg["artifacts"]["onnx_file"]
        vectorizer = joblib.load(onnx_path / "vectorizer.joblib")
        classes = json.loads((onnx_path / "classes.json").read_text(encoding="utf-8"))
        session = ort.InferenceSession(
            str(onnx_file), providers=["CPUExecutionProvider"]
        )
        return OnnxPredictor(vectorizer, session, classes)

    raise ValueError(f"model_backend desconhecido: {backend!r}")
