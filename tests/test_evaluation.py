from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pytest

import src.evaluation.evaluate as evaluation
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config


class _FixedPredictor:
    def __init__(self, predictions: dict[str, str]) -> None:
        self.predictions = predictions

    def fit(self, *_args: object) -> None:
        raise AssertionError("evaluation must not fit the pipeline")

    def predict(self, texts: Iterable[str]) -> list[str]:
        return [self.predictions[text] for text in texts]


def _frame(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [f"{prefix}-{index}" for index in range(5)],
            "label": CANONICAL_LABELS,
        }
    )


def _predictions(*prefixes: str) -> dict[str, str]:
    predicted = [
        CANONICAL_LABELS[0],
        CANONICAL_LABELS[0],
        CANONICAL_LABELS[2],
        CANONICAL_LABELS[3],
        CANONICAL_LABELS[4],
    ]
    return {
        f"{prefix}-{index}": label
        for prefix in prefixes
        for index, label in enumerate(predicted)
    }


def _config() -> ExperimentConfig:
    return load_config()


def test_evaluate_splits_reports_validation_and_test_without_refit() -> None:
    predictor = _FixedPredictor(_predictions("validation", "test"))

    metrics = evaluation.evaluate_splits(
        predictor, _frame("validation"), _frame("test"), _config()
    )

    assert set(metrics) == {"validation", "test"}
    assert metrics["test"]["accuracy"] == pytest.approx(0.8)
    assert metrics["test"]["labels"] == CANONICAL_LABELS
    assert metrics["test"]["confusion_matrix"] == [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]


def test_evaluate_split_reports_complete_class_and_aggregate_metrics() -> None:
    predictor = _FixedPredictor(_predictions("test"))

    metrics = evaluation.evaluate_split(
        predictor, _frame("test"), CANONICAL_LABELS, "text", "label"
    )

    assert list(metrics["per_class"]) == CANONICAL_LABELS
    assert set(metrics["per_class"][CANONICAL_LABELS[0]]) == {
        "precision",
        "recall",
        "f1",
        "support",
    }
    assert set(metrics["macro_avg"]) == {"precision", "recall", "f1"}
    assert set(metrics["weighted_avg"]) == {"precision", "recall", "f1"}
    assert metrics["minority_classes"] == CANONICAL_LABELS[:2]
    assert metrics["minority_class_recall_mean"] == pytest.approx(0.5)
