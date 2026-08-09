from pathlib import Path

import pandas as pd
import pytest

from src.config.settings import settings
from src.models.classifier import build_pipeline
from src.models.registry import save_pipeline
from src.serving.model_loader import SklearnPredictor, load_predictor
from src.utils.config_loader import load_config

_CFG = {
    "features": {"max_features": 50, "ngram_range": [1, 1], "min_df": 1},
    "model": {
        "type": "logistic_regression",
        "logistic_regression": {"C": 1.0, "max_iter": 200, "random_state": 42},
    },
}


def _train_and_save(artifacts_path: Path, pipeline_file: str) -> None:
    pipeline = build_pipeline(_CFG)
    pipeline.fit(
        pd.Series(["dor leve", "febre moderada", "dor torácica intensa"]),
        pd.Series(["normal", "atencao", "urgente"]),
    )
    save_pipeline(pipeline, artifacts_path, pipeline_file, "metadata.json", {})


def test_load_predictor_usa_model_backend_das_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    pipeline_file = cfg["artifacts"]["pipeline_file"]
    _train_and_save(tmp_path, pipeline_file)

    monkeypatch.setattr(settings, "model_backend", "sklearn")
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    predictor = load_predictor()

    assert isinstance(predictor, SklearnPredictor)
    assert predictor.backend == "sklearn"


def test_load_predictor_backend_explicito_sobrescreve_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = load_config()
    pipeline_file = cfg["artifacts"]["pipeline_file"]
    _train_and_save(tmp_path, pipeline_file)

    monkeypatch.setattr(settings, "model_backend", "onnx")
    monkeypatch.setattr(settings, "model_artifacts_path", str(tmp_path))

    predictor = load_predictor(backend="sklearn")

    assert predictor.backend == "sklearn"


def test_load_predictor_backend_desconhecido_lanca_erro() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        load_predictor(backend="tensorflow")
