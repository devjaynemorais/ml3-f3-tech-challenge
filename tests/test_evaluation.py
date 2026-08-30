from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
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


class _FixedProbaPredictor:
    def __init__(self, classes: list[str], probabilities: np.ndarray) -> None:
        self.classes_ = classes
        self._probabilities = probabilities

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        return self._probabilities


class _FixedFullPredictor:
    def __init__(
        self, predictions: dict[str, str], probabilities: np.ndarray, classes: list[str]
    ) -> None:
        self.predictions = predictions
        self._probabilities = probabilities
        self.classes_ = classes

    def predict(self, texts: Iterable[str]) -> list[str]:
        return [self.predictions[text] for text in texts]

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        return self._probabilities


def _write_raw_corpus(root: Path, config: ExperimentConfig) -> ExperimentConfig:
    """Write a tiny raw corpus where one abstract carries multiple labels."""
    raw_path = root / "raw"
    raw_path.mkdir()
    labels = pd.DataFrame(
        {"condition_label": [1, 2, 3, 4, 5], "condition_name": CANONICAL_LABELS}
    )
    train = pd.DataFrame(
        {
            "medical_abstract": ["shared abstract", "shared abstract", "only in train"],
            "condition_label": [1, 2, 3],
        }
    )
    test = pd.DataFrame(
        {
            "medical_abstract": ["shared abstract", "only in test"],
            "condition_label": [4, 5],
        }
    )
    labels.to_csv(raw_path / config.data.labels_file, index=False)
    train.to_csv(raw_path / config.data.train_file, index=False)
    test.to_csv(raw_path / config.data.test_file, index=False)
    return config.model_copy(
        update={"data": config.data.model_copy(update={"raw_path": raw_path})}
    )


def test_build_label_sets_collects_every_label_per_abstract(tmp_path: Path) -> None:
    config = _write_raw_corpus(tmp_path, _config())

    label_sets = evaluation.build_label_sets(config)

    assert label_sets["shared abstract"] == {
        CANONICAL_LABELS[0],
        CANONICAL_LABELS[1],
        CANONICAL_LABELS[3],
    }
    assert label_sets["only in train"] == {CANONICAL_LABELS[2]}
    assert label_sets["only in test"] == {CANONICAL_LABELS[4]}


def test_accuracy_by_label_count_stratifies_correctly() -> None:
    predicted = ["a", "b", "a", "a"]
    truth = pd.Series(["a", "a", "a", "b"])
    label_counts = pd.Series([1, 1, 2, 2])

    result = evaluation._accuracy_by_label_count(predicted, truth, label_counts)

    assert result == {"1": pytest.approx(0.5), "2": pytest.approx(0.5)}


def test_top_2_accuracy_counts_truth_within_top_two_scores() -> None:
    predictor = _FixedProbaPredictor(
        classes=CANONICAL_LABELS,
        probabilities=np.array(
            [
                [0.5, 0.3, 0.1, 0.05, 0.05],
                [0.05, 0.6, 0.1, 0.2, 0.05],
            ]
        ),
    )
    truth = pd.Series([CANONICAL_LABELS[1], CANONICAL_LABELS[0]])

    result = evaluation._top_2_accuracy(predictor, pd.Series(["t1", "t2"]), truth)

    assert result == pytest.approx(0.5)


def test_top_2_accuracy_returns_none_without_predict_proba() -> None:
    result = evaluation._top_2_accuracy(object(), pd.Series(["t"]), pd.Series(["x"]))

    assert result is None


def test_multilabel_metrics_scores_hits_against_the_full_valid_label_set() -> None:
    labels = CANONICAL_LABELS
    texts = pd.Series(["shared", "only-a", "miss"])
    predicted = [CANONICAL_LABELS[0], CANONICAL_LABELS[2], CANONICAL_LABELS[4]]
    label_sets = {
        # predicted[0] is one of two valid labels here: a hit, discounted by
        # the size of the valid set (samples-averaged Jaccard).
        "shared": {CANONICAL_LABELS[0], CANONICAL_LABELS[1]},
        # predicted[1] is the only valid label: a full-credit hit.
        "only-a": {CANONICAL_LABELS[2]},
        # predicted[2] is not in the valid set: a miss.
        "miss": {CANONICAL_LABELS[3]},
    }

    metrics = evaluation._multilabel_metrics(predicted, texts, label_sets, labels)

    assert metrics["multilabel_accuracy"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    # 2 of 3 predictions land inside their valid label set.
    assert metrics["multilabel_precision_micro"] == pytest.approx(2 / 3)
    # 2 correctly-covered labels out of 4 total valid labels across rows.
    assert metrics["multilabel_recall_micro"] == pytest.approx(2 / 4)


def test_evaluate_with_label_sets_merges_honesty_metrics() -> None:
    config = _config()
    frame = pd.DataFrame(
        {
            config.data.text_column: ["shared", "only-a"],
            config.data.label_column: [CANONICAL_LABELS[1], CANONICAL_LABELS[2]],
        }
    )
    label_sets = {
        "shared": {CANONICAL_LABELS[0], CANONICAL_LABELS[1]},
        "only-a": {CANONICAL_LABELS[2]},
    }
    predictor = _FixedFullPredictor(
        predictions={"shared": CANONICAL_LABELS[0], "only-a": CANONICAL_LABELS[2]},
        probabilities=np.array(
            [
                [0.6, 0.3, 0.05, 0.03, 0.02],
                [0.05, 0.05, 0.8, 0.06, 0.04],
            ]
        ),
        classes=CANONICAL_LABELS,
    )

    metrics = evaluation.evaluate_with_label_sets(predictor, frame, config, label_sets)

    assert metrics["in_set_accuracy"] == pytest.approx(1.0)
    assert metrics["accuracy_by_label_count"] == {
        "1": pytest.approx(1.0),
        "2": pytest.approx(0.0),
    }
    assert metrics["top_2_accuracy"] == pytest.approx(1.0)
    # Both predictions are hits, but "shared" has 2 valid labels, so its
    # per-row Jaccard is 1/2 rather than the full 1/1 "only-a" gets.
    assert metrics["multilabel_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert metrics["multilabel_f1_macro"] > 0
