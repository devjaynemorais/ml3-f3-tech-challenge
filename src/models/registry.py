"""Persistência simples do pipeline treinado (joblib + metadata.json)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_pipeline(
    pipeline: Pipeline,
    artifacts_path: Path,
    pipeline_file: str,
    metadata_file: str,
    metadata: dict[str, Any],
) -> Path:
    """Salva o pipeline treinado e um JSON de metadata ao lado dele."""
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_path / pipeline_file
    joblib.dump(pipeline, model_path)

    metadata = {**metadata, "saved_at": datetime.now(UTC).isoformat()}
    (artifacts_path / metadata_file).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return model_path


def load_pipeline(artifacts_path: Path, pipeline_file: str) -> Pipeline:
    """Carrega o pipeline treinado do disco."""
    model_path = artifacts_path / pipeline_file
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. Rode `make train` primeiro."
        )
    return joblib.load(model_path)
