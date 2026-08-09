from pathlib import Path

import pandas as pd
import pytest

from src.models.classifier import build_pipeline
from src.models.registry import load_pipeline, save_pipeline

_CFG = {
    "features": {"max_features": 50, "ngram_range": [1, 1], "min_df": 1},
    "model": {
        "type": "logistic_regression",
        "logistic_regression": {"C": 1.0, "max_iter": 200, "random_state": 42},
    },
}
_TEXTS = ["dor leve", "febre moderada", "dor torácica intensa"]
_LABELS = ["normal", "atencao", "urgente"]


def test_save_e_load_pipeline_roundtrip(tmp_path: Path) -> None:
    pipeline = build_pipeline(_CFG)
    pipeline.fit(pd.Series(_TEXTS), pd.Series(_LABELS))

    model_path = save_pipeline(
        pipeline,
        artifacts_path=tmp_path,
        pipeline_file="model.joblib",
        metadata_file="metadata.json",
        metadata={"model_type": "logistic_regression"},
    )
    assert model_path.exists()
    assert (tmp_path / "metadata.json").exists()

    loaded = load_pipeline(tmp_path, "model.joblib")
    assert loaded.predict(["febre moderada"])[0] in _LABELS


def test_load_pipeline_inexistente_lanca_erro(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path, "nao_existe.joblib")
