from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

import src.evaluation.evaluate as evaluation
from src.utils.config_loader import CANONICAL_LABELS, load_config


class _FixedPredictor:
    def __init__(self, predictions: dict[str, list[int]]) -> None:
        self.predictions = predictions

    def fit(self, *_args: object) -> None:
        raise AssertionError("evaluation must not fit the pipeline")

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        return np.asarray([self.predictions[text] for text in texts])


def _frame(prefix: str) -> pd.DataFrame:
    rows = []
    for index in range(5):
        row = {"text": f"{prefix}-{index}"}
        row.update(
            {
                label: int(position == index)
                for position, label in enumerate(CANONICAL_LABELS)
            }
        )
        rows.append(row)
    rows[0][CANONICAL_LABELS[1]] = 1
    return pd.DataFrame(rows)


def _predictions(*prefixes: str) -> dict[str, list[int]]:
    predictions = np.eye(5, dtype=int)
    predictions[1] = predictions[0]
    return {
        f"{prefix}-{index}": predictions[index].tolist()
        for prefix in prefixes
        for index in range(5)
    }


def test_evaluate_splits_reports_multilabel_metrics_without_refit() -> None:
    predictor = _FixedPredictor(_predictions("validation", "test"))
    metrics = evaluation.evaluate_splits(
        predictor, _frame("validation"), _frame("test"), load_config()
    )

    assert set(metrics) == {"validation", "test"}
    assert metrics["test"]["subset_accuracy"] == pytest.approx(0.6)
    assert metrics["test"]["micro_avg"]["f1"] > 0
    assert metrics["test"]["hamming_loss"] > 0
    assert metrics["test"]["labels"] == CANONICAL_LABELS


def test_evaluate_split_rejects_wrong_prediction_shape() -> None:
    class WrongShape:
        def predict(self, texts: Iterable[str]) -> np.ndarray:
            return np.zeros((len(list(texts)), 4), dtype=int)

    with pytest.raises(ValueError, match="must match target shape"):
        evaluation.evaluate_split(
            WrongShape(), _frame("test"), CANONICAL_LABELS, "text"
        )


def test_multilabel_metrics_report_label_cardinality() -> None:
    frame = _frame("test")
    predictor = _FixedPredictor(_predictions("test"))
    metrics = evaluation.evaluate_split(predictor, frame, CANONICAL_LABELS, "text")

    assert metrics["label_cardinality_truth"] == pytest.approx(1.2)
    assert metrics["label_cardinality_predicted"] == pytest.approx(1.0)
    assert list(metrics["per_class"]) == CANONICAL_LABELS
    assert set(metrics["macro_avg"]) == {"precision", "recall", "f1"}
