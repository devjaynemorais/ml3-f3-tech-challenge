"""Evaluate a fitted classifier on processed validation and official test data."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.make_dataset import load_processed_split
from src.models.registry import load_pipeline
from src.utils.config_loader import ExperimentConfig, load_config

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Prediction interface required by evaluation."""

    def predict(self, texts: Iterable[str]) -> Iterable[str]:
        """Predict one label per text."""
        ...


def _averaged_metrics(
    truth: Iterable[str], predicted: Iterable[str], labels: list[str], average: str
) -> dict[str, float]:
    """Compute precision, recall and F1 for one averaging method."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average=average, zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _per_class_metrics(
    truth: Iterable[str], predicted: Iterable[str], labels: list[str]
) -> dict[str, dict[str, float | int]]:
    """Compute ordered precision, recall, F1 and support for every class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
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


def _minority_classes(truth: pd.Series, labels: list[str]) -> list[str]:
    """Select the two least frequent classes with canonical tie-breaking."""
    counts = truth.value_counts().to_dict()
    positions = {label: index for index, label in enumerate(labels)}
    return sorted(labels, key=lambda label: (counts.get(label, 0), positions[label]))[
        :2
    ]


def evaluate_split(
    predictor: Predictor,
    frame: pd.DataFrame,
    labels: list[str],
    text_column: str,
    label_column: str,
) -> dict:
    """Compute complete metrics for one immutable processed split."""
    truth = frame[label_column]
    predicted = list(predictor.predict(frame[text_column]))
    per_class = _per_class_metrics(truth, predicted, labels)
    minority = _minority_classes(truth, labels)
    minority_recall = sum(per_class[label]["recall"] for label in minority) / 2
    return {
        "n_samples": len(frame),
        "accuracy": float(accuracy_score(truth, predicted)),
        "per_class": per_class,
        "macro_avg": _averaged_metrics(truth, predicted, labels, "macro"),
        "weighted_avg": _averaged_metrics(truth, predicted, labels, "weighted"),
        "minority_classes": minority,
        "minority_class_recall_mean": float(minority_recall),
        "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def evaluate_splits(
    predictor: Predictor,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, dict]:
    """Evaluate validation and official test without fitting the predictor."""
    arguments = (config.data.labels, config.data.text_column, config.data.label_column)
    return {
        "validation": evaluate_split(predictor, validation, *arguments),
        "test": evaluate_split(predictor, test, *arguments),
    }


def evaluate_from_processed(config: ExperimentConfig) -> dict[str, dict]:
    """Load the persisted pipeline and both immutable evaluation splits."""
    pipeline = load_pipeline(
        config.artifacts.model_path, config.artifacts.pipeline_file
    )
    validation = load_processed_split(config, config.data.validation_output_file)
    test = load_processed_split(config, config.data.test_output_file)
    return evaluate_splits(pipeline, validation, test, config)


def write_metrics(metrics: dict, path: Path) -> None:
    """Persist JSON metrics with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Evaluate processed validation/test and persist eval_metrics.json."""
    config = load_config()
    metrics = evaluate_from_processed(config)
    metrics_path = config.artifacts.metrics_path / config.artifacts.metrics_file
    write_metrics(metrics, metrics_path)
    logger.info("Validation and test metrics saved at %s", metrics_path)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
