"""Otimização de latência — exporta o classificador treinado para ONNX Runtime.

A vetorização TF-IDF permanece em scikit-learn (já é barata); a etapa cara de
inferência (Random Forest / Logistic Regression) é convertida para ONNX, que
roda via onnxruntime com overhead de dispatch bem menor que o predict do
scikit-learn puro. `scripts/measure_latency.py` compara os dois caminhos.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.models.registry import load_pipeline
from src.utils.config_loader import load_config

if TYPE_CHECKING:
    from onnx import ModelProto
    from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def export_classifier_to_onnx(pipeline: Pipeline, n_features: int) -> ModelProto:
    """Converte apenas o estágio `classifier` do pipeline sklearn para ONNX.

    O `zipmap=False` evita que o skl2onnx envolva a saída de probabilidades
    em uma lista de dicts — mantemos um tensor denso, mais simples e mais
    rápido de consumir no lado do serving.
    """
    classifier = pipeline.named_steps["classifier"]
    initial_type = [("input", FloatTensorType([None, n_features]))]
    model: ModelProto = convert_sklearn(
        classifier,
        initial_types=initial_type,
        options={id(classifier): {"zipmap": False}},
    )
    return model


def main() -> None:
    """Carrega o pipeline treinado e exporta o classificador para ONNX."""
    cfg = load_config()
    artifacts_path = Path(cfg["artifacts"]["model_path"])
    onnx_path = Path(cfg["artifacts"]["onnx_path"])
    onnx_path.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(artifacts_path, cfg["artifacts"]["pipeline_file"])
    vectorizer = pipeline.named_steps["tfidf"]
    n_features = len(vectorizer.idf_)

    onnx_model = export_classifier_to_onnx(pipeline, n_features)

    onnx_file = onnx_path / cfg["artifacts"]["onnx_file"]
    onnx_file.write_bytes(onnx_model.SerializeToString())

    # O vetorizador TF-IDF fica fora do grafo ONNX — precisa acompanhar o
    # classificador para que o serving em modo "onnx" saiba tokenizar o texto.
    vectorizer_file = onnx_path / "vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_file)

    # A ordem das colunas de probabilidade retornadas pelo ONNX runtime segue
    # `classifier.classes_` — precisa ser persistida para o serving remontar
    # o rótulo a partir do índice da maior probabilidade.
    classes_file = onnx_path / "classes.json"
    classifier = pipeline.named_steps["classifier"]
    classes_file.write_text(json.dumps(list(classifier.classes_)), encoding="utf-8")

    logger.info(
        "Classificador exportado para ONNX em %s (vetorizador em %s, %d features)",
        onnx_file,
        vectorizer_file,
        n_features,
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
