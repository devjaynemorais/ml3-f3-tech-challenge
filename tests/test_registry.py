import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.models.classifier import build_pipeline
from src.models.registry import load_pipeline, save_pipeline
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


def _fitted_pipeline():
    pipeline = build_pipeline(_config())
    texts = pd.Series(
        [f"class{index} sample{sample}" for index in range(5) for sample in range(3)]
    )
    labels = pd.Series([label for label in CANONICAL_LABELS for _ in range(3)])
    return pipeline.fit(texts, labels)


def test_save_and_load_pipeline_preserves_predictions(tmp_path: Path) -> None:
    pipeline = _fitted_pipeline()
    expected = pipeline.predict_proba(["class2 sample"])

    model_path = save_pipeline(
        pipeline,
        artifacts_path=tmp_path,
        pipeline_file="model.joblib",
        metadata_file="metadata.json",
        metadata={"schema_version": 2, "classes": CANONICAL_LABELS},
    )
    loaded = load_pipeline(tmp_path, "model.joblib")
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert model_path.exists()
    assert loaded.predict_proba(["class2 sample"]) == pytest.approx(expected)
    assert metadata["schema_version"] == 2
    assert metadata["classes"] == CANONICAL_LABELS
    assert "saved_at" in metadata


def test_load_pipeline_rejects_non_pipeline_artifact(tmp_path: Path) -> None:
    joblib.dump({"not": "a pipeline"}, tmp_path / "invalid.joblib")

    with pytest.raises(TypeError, match="sklearn Pipeline"):
        load_pipeline(tmp_path, "invalid.joblib")


def test_load_pipeline_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make train"):
        load_pipeline(tmp_path, "missing.joblib")
