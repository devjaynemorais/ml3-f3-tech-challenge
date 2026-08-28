from pathlib import Path

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
    targets = pd.Series([label for label in training_labels for _ in range(3)])
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
    assert sum(scores.values()) == pytest.approx(1.0)


def test_explicit_backend_overrides_runtime_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _save_pipeline(tmp_path)
    monkeypatch.setattr(settings, "model_backend", "onnx")
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    predictor = model_loader.load_predictor(backend="sklearn")

    assert predictor.backend == "sklearn"


def test_old_three_class_artifact_is_rejected() -> None:
    pipeline = _fitted_pipeline(["normal", "atencao", "urgente"])

    with pytest.raises(ValueError, match="artifact classes"):
        model_loader.SklearnPredictor(pipeline, CANONICAL_LABELS)


def test_duplicate_or_missing_artifact_classes_are_rejected() -> None:
    with pytest.raises(ValueError, match="artifact classes"):
        model_loader.validate_artifact_classes(
            [*CANONICAL_LABELS[:-1], CANONICAL_LABELS[0]], CANONICAL_LABELS
        )


def test_load_predictor_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown model backend"):
        model_loader.load_predictor(backend="tensorflow")
