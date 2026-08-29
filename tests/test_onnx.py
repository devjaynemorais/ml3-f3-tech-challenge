from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

import src.optimization.export_onnx as exporter
from src.models.classifier import build_pipeline
from src.serving.model_loader import OnnxPredictor, SklearnPredictor
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config

SAMPLES = [
    "oncology tumor abstract",
    "digestive gastric abstract",
    "neurology neural abstract",
    "cardiac vascular abstract",
    "general pathology abstract",
]


def _config(model_type: str = "logistic_regression") -> ExperimentConfig:
    config = load_config()
    preprocessing = config.preprocessing.model_copy(
        update={"steps": ["unicode_normalization", "punctuation_removal"]}
    )
    features = config.features.model_copy(update={"min_df": 1, "ngram_range": (1, 1)})
    model = config.model.model_copy(update={"type": model_type})
    model = model.model_copy(
        update={
            "gradient_boosting": model.gradient_boosting.model_copy(
                update={"n_estimators": 5, "selection_k": 8}
            )
        }
    )
    return config.model_copy(
        update={
            "preprocessing": preprocessing,
            "features": features,
            "model": model,
        }
    )


def _fitted_pipeline(config: ExperimentConfig):
    texts = pd.Series(
        [f"{sample} training{index}" for sample in SAMPLES for index in range(4)]
    )
    labels = pd.Series([label for label in CANONICAL_LABELS for _ in range(4)])
    return build_pipeline(config).fit(texts, labels)


def test_export_persists_complete_feature_prefix(tmp_path: Path) -> None:
    config = _config("gradient_boosting")
    pipeline = _fitted_pipeline(config)

    exporter.export_pipeline_artifacts(
        pipeline, tmp_path, config.artifacts, CANONICAL_LABELS
    )
    prefix = joblib.load(tmp_path / config.artifacts.feature_pipeline_file)

    assert list(prefix.named_steps) == [
        "preprocessor",
        "tfidf",
        "feature_selection",
        "to_dense",
    ]
    assert (
        pipeline.named_steps["classifier"].n_features_in_
        == prefix.transform([SAMPLES[0]]).shape[1]
    )


def test_sklearn_and_onnx_predictions_are_numerically_compatible(
    tmp_path: Path,
) -> None:
    config = _config()
    pipeline = _fitted_pipeline(config)
    exporter.export_pipeline_artifacts(
        pipeline, tmp_path, config.artifacts, CANONICAL_LABELS
    )
    prefix = joblib.load(tmp_path / config.artifacts.feature_pipeline_file)
    classes = json.loads(
        (tmp_path / config.artifacts.classes_file).read_text(encoding="utf-8")
    )
    session = ort.InferenceSession(
        str(tmp_path / config.artifacts.onnx_file), providers=["CPUExecutionProvider"]
    )
    sklearn_predictor = SklearnPredictor(pipeline, CANONICAL_LABELS)
    onnx_predictor = OnnxPredictor(prefix, session, classes, CANONICAL_LABELS)

    for text in SAMPLES:
        sklearn_label, sklearn_scores = sklearn_predictor.predict(text)
        onnx_label, onnx_scores = onnx_predictor.predict(text)
        assert onnx_label == sklearn_label
        assert list(onnx_scores) == CANONICAL_LABELS
        np.testing.assert_allclose(
            list(onnx_scores.values()),
            list(sklearn_scores.values()),
            rtol=1e-5,
            atol=1e-6,
        )
