"""Persist and load the fitted sklearn pipeline and versioned metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def save_pipeline(
    pipeline: Pipeline,
    artifacts_path: Path,
    pipeline_file: str,
    metadata_file: str,
    metadata: dict,
) -> Path:
    """Save a fitted pipeline and timestamped metadata side by side."""
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_path / pipeline_file
    joblib.dump(pipeline, model_path)
    persisted_metadata = {**metadata, "saved_at": datetime.now(UTC).isoformat()}
    metadata_path = artifacts_path / metadata_file
    metadata_path.write_text(
        json.dumps(persisted_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model_path


def load_pipeline(artifacts_path: Path, pipeline_file: str) -> Pipeline:
    """Load one sklearn Pipeline or report a missing/invalid artifact."""
    model_path = artifacts_path / pipeline_file
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}; run `make train`")
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"artifact at {model_path} is not an sklearn Pipeline")
    return pipeline
