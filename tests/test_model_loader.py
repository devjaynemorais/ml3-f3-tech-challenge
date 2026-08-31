from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.serving.model_loader as model_loader
from src.config.settings import settings
from src.models.classifier import build_pipeline
from src.models.registry import save_pipeline
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config


def _config() -> ExperimentConfig:
    config = load_config()
    preprocessing = config.preprocessing.model_copy(
        update={"steps": ["unicode_normalization", "punctuation_removal"]}
    )
    features = config.features.model_copy(update={"min_df": 1, "ngram_range": (1, 1)})
    return config.model_copy(
        update={"preprocessing": preprocessing, "features": features}
    )


def _fitted_pipeline(labels: list[str] | None = None):
    training_labels = labels or CANONICAL_LABELS
    pipeline = build_pipeline(_config())
    texts = pd.Series(
        [
            f"class{index} abstract sample{sample}"
            for index in range(len(training_labels))
            for sample in range(3)
        ]
    )
    targets = np.asarray(
        [
            [int(row // 3 == column) for column in range(len(training_labels))]
            for row in range(len(texts))
        ]
    )
    return pipeline.fit(texts, targets)


def _save_pipeline(path: Path) -> None:
    config = load_config()
    save_pipeline(
        _fitted_pipeline(),
        path,
        config.artifacts.pipeline_file,
        config.artifacts.metadata_file,
        {"schema_version": 2},
    )


def test_load_predictor_uses_runtime_artifact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_pipeline(tmp_path)
    monkeypatch.setattr(settings, "model_backend", "sklearn")
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    predictor = model_loader.load_predictor()
    label, scores = predictor.predict("class2 abstract")

    assert isinstance(predictor, model_loader.SklearnPredictor)
    assert predictor.backend == "sklearn"
    assert label in CANONICAL_LABELS
    assert list(scores) == CANONICAL_LABELS
    assert all(0 <= score <= 1 for score in scores.values())


def test_explicit_backend_overrides_runtime_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_pipeline(tmp_path)
    monkeypatch.setattr(settings, "model_backend", "onnx")
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    predictor = model_loader.load_predictor(backend="sklearn")

    assert predictor.backend == "sklearn"


def test_old_three_class_artifact_is_rejected_at_prediction() -> None:
    pipeline = _fitted_pipeline(["normal", "atencao", "urgente"])
    predictor = model_loader.SklearnPredictor(pipeline, CANONICAL_LABELS)
    with pytest.raises(ValueError, match="probability columns"):
        predictor.predict("abstract")


def test_duplicate_or_missing_artifact_classes_are_rejected() -> None:
    with pytest.raises(ValueError, match="artifact classes"):
        model_loader.validate_artifact_classes(
            [*CANONICAL_LABELS[:-1], CANONICAL_LABELS[0]], CANONICAL_LABELS
        )


def test_load_predictor_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown model backend"):
        model_loader.load_predictor(backend="tensorflow")


def test_sklearn_explain_exposes_preprocessing_and_top_terms() -> None:
    predictor = model_loader.SklearnPredictor(_fitted_pipeline(), CANONICAL_LABELS)

    result = predictor.explain("Class2 Abstract Sample")

    assert result["label"] in CANONICAL_LABELS
    assert result["backend"] == "sklearn"
    assert result["preprocessed_text"] == "class2 abstract sample"
    assert result["top_terms"], "linear model over TF-IDF must expose top terms"
    top_term = result["top_terms"][0]
    assert {"term", "tfidf", "weight", "contribution"} == set(top_term)
    contributions = [term["contribution"] for term in result["top_terms"]]
    assert contributions == sorted(contributions, reverse=True)
