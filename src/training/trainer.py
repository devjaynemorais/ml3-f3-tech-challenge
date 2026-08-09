"""Treina o classificador de urgência (TF-IDF + modelo leve) e salva o pipeline.

Usado tanto localmente (`make train`) quanto pela task de treino da DAG do
Airflow (`airflow/dags/triage_training_dag.py`) — ambos chamam `main()`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.make_dataset import load_raw_dataset, split_dataset
from src.models.classifier import build_pipeline
from src.models.registry import save_pipeline
from src.utils.config_loader import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def train(train_df: pd.DataFrame, cfg: dict) -> tuple[object, dict[str, float | str]]:
    """Ajusta o pipeline nos dados de treino e retorna (pipeline, metadata)."""
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    pipeline = build_pipeline(cfg)
    pipeline.fit(train_df[text_col], train_df[label_col])

    metadata = {
        "model_type": cfg["model"]["type"],
        "n_train_samples": len(train_df),
        "labels": sorted(train_df[label_col].unique().tolist()),
    }
    return pipeline, metadata


def main() -> None:
    """Pipeline de treino: carrega dados -> split -> treina -> salva artefato."""
    cfg = load_config()
    set_seed(cfg["split"]["random_state"])

    raw_path = Path(cfg["data"]["raw_path"]) / cfg["data"]["raw_file"]
    df = load_raw_dataset(
        raw_path, cfg["data"]["text_column"], cfg["data"]["label_column"]
    )
    train_df, _val_df, _test_df = split_dataset(
        df,
        cfg["data"]["label_column"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )

    pipeline, metadata = train(train_df, cfg)

    artifacts_path = Path(cfg["artifacts"]["model_path"])
    model_path = save_pipeline(
        pipeline,
        artifacts_path,
        cfg["artifacts"]["pipeline_file"],
        cfg["artifacts"]["metadata_file"],
        metadata,
    )
    logger.info("Modelo treinado e salvo em %s (%s)", model_path, metadata)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
