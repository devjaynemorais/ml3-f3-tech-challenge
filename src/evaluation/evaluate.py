"""Avalia o pipeline treinado no split de teste e salva métricas em JSON."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from src.data.make_dataset import load_raw_dataset, split_dataset
from src.models.registry import load_pipeline
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def evaluate(pipeline: Pipeline, test_df: pd.DataFrame, cfg: dict) -> dict:
    """Calcula accuracy, F1 macro e o classification report completo."""
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    y_true = test_df[label_col]
    y_pred = pipeline.predict(test_df[text_col])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "n_test_samples": len(test_df),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    """Reavalia o modelo salvo no split de teste e grava metrics/eval_metrics.json."""
    cfg = load_config()

    raw_path = Path(cfg["data"]["raw_path"]) / cfg["data"]["raw_file"]
    df = load_raw_dataset(
        raw_path, cfg["data"]["text_column"], cfg["data"]["label_column"]
    )
    _train_df, _val_df, test_df = split_dataset(
        df,
        cfg["data"]["label_column"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )

    pipeline = load_pipeline(
        Path(cfg["artifacts"]["model_path"]), cfg["artifacts"]["pipeline_file"]
    )
    metrics = evaluate(pipeline, test_df, cfg)

    metrics_path = Path("metrics/eval_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Avaliação concluída — accuracy=%.4f f1_macro=%.4f (salvo em %s)",
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics_path,
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
