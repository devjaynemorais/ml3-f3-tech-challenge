"""Evaluate a fitted multilabel classifier on processed validation and test data."""

from __future__ import annotations

# Optional (only the BERT baseline needs it) but must load before pandas: on
# Windows, torch's native DLL fails to initialize if pandas claims the
# process's DLL search path first.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
)

from src.data.make_dataset import load_processed_split
from src.models.registry import load_metadata, load_pipeline
from src.utils.config_loader import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Prediction interface required by multilabel evaluation."""

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        """Predict one binary indicator per canonical label."""
        ...


def _averaged_metrics(
    truth: np.ndarray, predicted: np.ndarray, average: str
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average=average, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _per_class_metrics(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float | int]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, average=None, zero_division=0
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }


def score_multilabel(
    truth: np.ndarray, predicted: np.ndarray, labels: list[str]
) -> dict:
    """Score an already predicted multi-hot matrix against its multi-hot truth.

    Shared by ``evaluate_split`` (sklearn Strategies, predicting from text) and
    ``scripts/finetune_bert.py`` (predictions already computed by the HF
    Trainer) so both report the same "metricas justas" — see
    ``docs/model_card.md#avaliacao``.
    """
    if predicted.shape != truth.shape:
        raise ValueError("predicted multilabel matrix must match target shape")
    per_class = _per_class_metrics(truth, predicted, labels)
    minority = sorted(labels, key=lambda item: per_class[item]["support"])[:2]
    minority_recall = float(np.mean([per_class[item]["recall"] for item in minority]))
    return {
        "n_samples": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "subset_accuracy": float(accuracy_score(truth, predicted)),
        "hamming_loss": float(hamming_loss(truth, predicted)),
        "jaccard_samples": float(
            jaccard_score(truth, predicted, average="samples", zero_division=0)
        ),
        "label_cardinality_truth": float(truth.sum(axis=1).mean()),
        "label_cardinality_predicted": float(predicted.sum(axis=1).mean()),
        "per_class": per_class,
        "macro_avg": _averaged_metrics(truth, predicted, "macro"),
        "micro_avg": _averaged_metrics(truth, predicted, "micro"),
        "weighted_avg": _averaged_metrics(truth, predicted, "weighted"),
        "minority_classes": minority,
        "minority_class_recall_mean": minority_recall,
        "labels": labels,
    }


def evaluate_split(
    predictor: Predictor,
    frame: pd.DataFrame,
    labels: list[str],
    text_column: str,
) -> dict:
    """Compute standard multilabel metrics for one processed split."""
    truth = frame[labels].to_numpy(dtype=int)
    predicted = np.asarray(predictor.predict(frame[text_column]), dtype=int)
    return score_multilabel(truth, predicted, labels)


def evaluate_splits(
    predictor: Predictor,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, dict]:
    """Evaluate validation and test without refitting the predictor."""
    arguments = (config.data.labels, config.data.text_column)
    return {
        "validation": evaluate_split(predictor, validation, *arguments),
        "test": evaluate_split(predictor, test, *arguments),
    }


def evaluate_from_processed(config: ExperimentConfig) -> dict[str, dict]:
    """Load persisted artifacts and evaluate both processed splits."""
    pipeline = load_pipeline(
        config.artifacts.model_path, config.artifacts.pipeline_file
    )
    validation = load_processed_split(config, config.data.validation_output_file)
    test = load_processed_split(config, config.data.test_output_file)
    return evaluate_splits(pipeline, validation, test, config)


def write_metrics(metrics: dict, path: Path) -> None:
    """Persist metrics as stable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_split_metrics(split_name: str, metrics: dict) -> None:
    for key in (
        "subset_accuracy",
        "hamming_loss",
        "jaccard_samples",
        "label_cardinality_truth",
        "label_cardinality_predicted",
        "minority_class_recall_mean",
    ):
        mlflow.log_metric(f"{split_name}_{key}", metrics[key])
    for average in ("macro", "micro", "weighted"):
        for metric, value in metrics[f"{average}_avg"].items():
            mlflow.log_metric(f"{split_name}_{average}_{metric}", value)


def log_metrics_to_mlflow(run_id: str, metrics: dict[str, dict]) -> None:
    """Attach multilabel metrics to the training run when MLflow is available."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for split_name, split_metrics in metrics.items():
                _log_split_metrics(split_name, split_metrics)
    except Exception:
        logger.warning("MLflow unavailable at %s; skipping metrics", tracking_uri)


def _cv_summary(metadata: dict) -> dict:
    return {
        "note": "refit inclui este split (CV-tuning); nao e held-out",
        "cv_macro_f1_mean": metadata["cv_macro_f1_mean"],
        "cv_macro_f1_std": metadata["cv_macro_f1_std"],
        "cv_folds": metadata["cv_folds"],
    }


def main() -> None:
    """Evaluate the configured model and persist multilabel metrics."""
    config = load_config()
    metadata = load_metadata(
        config.artifacts.model_path, config.artifacts.metadata_file
    )
    metrics = evaluate_from_processed(config)
    loggable = metrics
    if metadata.get("refit_includes_validation"):
        metrics["validation"] = _cv_summary(metadata)
        loggable = {"test": metrics["test"]}
    path = config.artifacts.metrics_path / config.artifacts.metrics_file
    write_metrics(metrics, path)
    if run_id := metadata.get("mlflow_run_id"):
        log_metrics_to_mlflow(run_id, loggable)
    logger.info("Multilabel metrics saved at %s", path)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
