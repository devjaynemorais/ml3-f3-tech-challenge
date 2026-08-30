"""Evaluate a fitted classifier on processed validation and official test data."""

from __future__ import annotations

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
    confusion_matrix,
    jaccard_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.data.make_dataset import load_medical_corpus, load_processed_split
from src.models.registry import load_metadata, load_pipeline
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


def build_label_sets(config: ExperimentConfig) -> dict[str, set[str]]:
    """Map each raw abstract text to every label it carries in the corpus.

    The Medical Abstracts corpus is multi-label flattened to one row per
    (text, label) pair: the same abstract can appear under several
    conditions. This reconstructs the full valid-label set per abstract so
    evaluation can tell "wrong" from "a different equally valid label."
    """
    corpus = load_medical_corpus(config.data)
    combined = pd.concat([corpus.train, corpus.test], ignore_index=True)
    grouped = combined.groupby(config.data.text_column)[config.data.label_column]
    return grouped.agg(set).to_dict()


def _accuracy_by_label_count(
    predicted: list[str],
    truth: pd.Series,
    label_counts: pd.Series,
) -> dict[str, float]:
    """Stratify accuracy by how many valid labels each abstract carries."""
    correct = pd.Series(predicted).reset_index(drop=True) == truth.reset_index(
        drop=True
    )
    counts = label_counts.reset_index(drop=True)
    return {str(k): float(correct[counts == k].mean()) for k in sorted(counts.unique())}


def _top_2_accuracy(
    pipeline: Pipeline, texts: pd.Series, truth: pd.Series
) -> float | None:
    """Fraction of rows where truth is among the two highest-scored labels."""
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba(texts)
    classes = np.array(pipeline.classes_)
    top_2 = classes[np.argsort(-probabilities, axis=1)[:, :2]]
    hits = [label in row for label, row in zip(truth, top_2, strict=True)]
    return float(np.mean(hits))


def _multilabel_metrics(
    predicted: list[str],
    texts: pd.Series,
    label_sets: dict[str, set[str]],
    labels: list[str],
) -> dict[str, float]:
    """Score the single prediction against the full multi-label ground truth.

    The corpus keeps every valid label per abstract (``build_label_sets``);
    the model still emits one label per prediction. This binarizes both sides
    (predicted label -> one-hot, valid label set -> multi-hot) and scores them
    with standard multilabel precision/recall/F1/accuracy, so a hit on any
    valid label counts as correct rather than only the one row the corpus
    happened to keep after flattening to multiclass. See
    ``docs/metodologia_experimentos.md`` for why this is the metric that
    reflects real-world value on this corpus, ahead of the flattened
    single-label scores in ``evaluate_split``.
    """
    binarizer = MultiLabelBinarizer(classes=labels)
    truth_matrix = binarizer.fit_transform(
        [label_sets.get(text, set()) for text in texts]
    )
    predicted_matrix = binarizer.transform([[label] for label in predicted])
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth_matrix, predicted_matrix, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        truth_matrix, predicted_matrix, average="micro", zero_division=0
    )
    # Samples-averaged Jaccard: intersection-over-union per row, then averaged.
    # With exactly one predicted label, a hit scores 1/|valid label set| — the
    # standard multilabel "accuracy" (Godbole & Sarawagi, 2004), unlike
    # in_set_accuracy which does not discount for how many labels were valid.
    accuracy = float(
        jaccard_score(
            truth_matrix, predicted_matrix, average="samples", zero_division=0
        )
    )
    return {
        "multilabel_precision_macro": float(macro_precision),
        "multilabel_recall_macro": float(macro_recall),
        "multilabel_f1_macro": float(macro_f1),
        "multilabel_precision_micro": float(micro_precision),
        "multilabel_recall_micro": float(micro_recall),
        "multilabel_f1_micro": float(micro_f1),
        "multilabel_accuracy": accuracy,
    }


def evaluate_with_label_sets(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    config: ExperimentConfig,
    label_sets: dict[str, set[str]],
) -> dict:
    """Add corpus-aware honesty metrics on top of the standard split metrics."""
    texts = frame[config.data.text_column]
    predicted = list(pipeline.predict(texts))
    probabilities = (
        np.asarray(pipeline.predict_proba(texts))
        if hasattr(pipeline, "predict_proba")
        else None
    )
    classes = list(pipeline.classes_) if probabilities is not None else None
    return evaluate_predictions_with_label_sets(
        predicted, probabilities, classes, frame, config, label_sets
    )


def evaluate_predictions_with_label_sets(
    predicted: list[str],
    probabilities: np.ndarray | None,
    classes: list[str] | None,
    frame: pd.DataFrame,
    config: ExperimentConfig,
    label_sets: dict[str, set[str]],
) -> dict:
    """Compute corpus-aware metrics from already generated model outputs."""
    text_column, label_column = config.data.text_column, config.data.label_column
    texts, truth = frame[text_column], frame[label_column]
    if len(predicted) != len(frame):
        raise ValueError("predicted labels must match the evaluated frame length")
    if probabilities is not None and probabilities.shape[0] != len(frame):
        raise ValueError("probabilities must match the evaluated frame length")
    if probabilities is not None and (
        classes is None or probabilities.shape[1] != len(classes)
    ):
        raise ValueError("probability columns must match the supplied classes")
    label_counts = texts.map(lambda text: len(label_sets.get(text, set())))
    in_set = [
        p in label_sets.get(t, set()) for p, t in zip(predicted, texts, strict=True)
    ]
    top_2_accuracy = None
    if probabilities is not None and classes is not None:
        top_2 = np.asarray(classes)[np.argsort(-probabilities, axis=1)[:, :2]]
        top_2_accuracy = float(
            np.mean([label in row for label, row in zip(truth, top_2, strict=True)])
        )
    return {
        "in_set_accuracy": float(np.mean(in_set)),
        "accuracy_by_label_count": _accuracy_by_label_count(
            predicted, truth, label_counts
        ),
        "top_2_accuracy": top_2_accuracy,
        **_multilabel_metrics(predicted, texts, label_sets, config.data.labels),
    }


def write_metrics(metrics: dict, path: Path) -> None:
    """Persist JSON metrics with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_honesty_metrics(split_name: str, split_metrics: dict) -> None:
    """Log the corpus-aware metrics from ``evaluate_with_label_sets``, if present."""
    if "in_set_accuracy" in split_metrics:
        mlflow.log_metric(
            f"{split_name}_in_set_accuracy", split_metrics["in_set_accuracy"]
        )
    if split_metrics.get("top_2_accuracy") is not None:
        mlflow.log_metric(
            f"{split_name}_top_2_accuracy", split_metrics["top_2_accuracy"]
        )
    for key, value in split_metrics.items():
        if key.startswith("multilabel_") and value is not None:
            mlflow.log_metric(f"{split_name}_{key}", value)


def _log_split_metrics(split_name: str, split_metrics: dict) -> None:
    """Log one split's tracked summary metrics to the active MLflow run."""
    mlflow.log_metric(f"{split_name}_accuracy", split_metrics["accuracy"])
    mlflow.log_metric(
        f"{split_name}_macro_precision", split_metrics["macro_avg"]["precision"]
    )
    mlflow.log_metric(f"{split_name}_macro_f1", split_metrics["macro_avg"]["f1"])
    mlflow.log_metric(f"{split_name}_weighted_f1", split_metrics["weighted_avg"]["f1"])
    mlflow.log_metric(
        f"{split_name}_minority_recall_mean",
        split_metrics["minority_class_recall_mean"],
    )
    _log_honesty_metrics(split_name, split_metrics)


def log_metrics_to_mlflow(run_id: str, metrics: dict[str, dict]) -> None:
    """Attach validation/test summary metrics to the training's MLflow run."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for split_name, split_metrics in metrics.items():
                _log_split_metrics(split_name, split_metrics)
    except Exception:
        logger.warning(
            "MLflow tracking unavailable at %s; skipping metric log", tracking_uri
        )


def _cv_summary(metadata: dict) -> dict:
    """Build a validation placeholder from CV stats: this split is not held-out."""
    return {
        "note": "refit inclui este split (CV-tuning); nao e held-out",
        "cv_macro_f1_mean": metadata["cv_macro_f1_mean"],
        "cv_macro_f1_std": metadata["cv_macro_f1_std"],
        "cv_folds": metadata["cv_folds"],
    }


def main() -> None:
    """Evaluate processed validation/test and persist eval_metrics.json."""
    config = load_config()
    metadata = load_metadata(
        config.artifacts.model_path, config.artifacts.metadata_file
    )
    metrics = evaluate_from_processed(config)
    pipeline = load_pipeline(
        config.artifacts.model_path, config.artifacts.pipeline_file
    )
    test = load_processed_split(config, config.data.test_output_file)
    label_sets = build_label_sets(config)
    metrics["test"].update(evaluate_with_label_sets(pipeline, test, config, label_sets))
    loggable_metrics = metrics
    if metadata.get("refit_includes_validation"):
        metrics["validation"] = _cv_summary(metadata)
        loggable_metrics = {"test": metrics["test"]}
    metrics_path = config.artifacts.metrics_path / config.artifacts.metrics_file
    write_metrics(metrics, metrics_path)
    run_id = metadata.get("mlflow_run_id")
    if run_id:
        log_metrics_to_mlflow(run_id, loggable_metrics)
    logger.info("Validation and test metrics saved at %s", metrics_path)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
