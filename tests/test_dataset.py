from pathlib import Path

import pandas as pd
import pytest

from src.data import make_dataset
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config


def _write_corpus(root: Path) -> None:
    labels = pd.DataFrame(
        {
            "condition_label": range(1, 6),
            "condition_name": CANONICAL_LABELS,
        }
    )
    train_rows = [
        {"condition_label": label_id, "medical_abstract": f"train-{label_id}-{i}"}
        for label_id in range(1, 6)
        for i in range(10)
    ]
    test_rows = [
        {"condition_label": label_id, "medical_abstract": f"test-{label_id}-{i}"}
        for label_id in range(1, 6)
        for i in range(2)
    ]
    labels.to_csv(root / "medical_tc_labels.csv", index=False)
    pd.DataFrame(train_rows).to_csv(root / "medical_tc_train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(root / "medical_tc_test.csv", index=False)


def _config_for(root: Path) -> ExperimentConfig:
    config = load_config()
    data = config.data.model_copy(
        update={"raw_path": root, "processed_path": root / "processed"}
    )
    return config.model_copy(update={"data": data})


def test_prepare_dataset_preserves_official_test_rows(tmp_path: Path) -> None:
    _write_corpus(tmp_path)

    summary = make_dataset.prepare_dataset(_config_for(tmp_path))

    actual = pd.read_csv(summary.test_path)
    assert actual["text"].tolist() == [
        f"test-{label_id}-{i}" for label_id in range(1, 6) for i in range(2)
    ]
    assert list(actual.columns) == ["text", "label"]


def test_validation_is_deterministic_and_comes_only_from_training(
    tmp_path: Path,
) -> None:
    _write_corpus(tmp_path)
    config = _config_for(tmp_path)

    first = make_dataset.prepare_dataset(config)
    first_validation = pd.read_csv(first.validation_path)
    second = make_dataset.prepare_dataset(config)
    second_validation = pd.read_csv(second.validation_path)

    assert first_validation.equals(second_validation)
    assert len(first_validation) == 5
    assert set(first_validation["label"]) == set(CANONICAL_LABELS)
    assert not set(first_validation["text"]) & {
        f"test-{label_id}-{i}" for label_id in range(1, 6) for i in range(2)
    }


def test_prepare_dataset_rejects_missing_source_column(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    train_path = tmp_path / "medical_tc_train.csv"
    train = pd.read_csv(train_path).drop(columns="medical_abstract")
    train.to_csv(train_path, index=False)

    with pytest.raises(ValueError, match="missing columns.*medical_abstract"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_rejects_non_unique_label_mapping(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    labels_path = tmp_path / "medical_tc_labels.csv"
    labels = pd.read_csv(labels_path)
    duplicate = pd.DataFrame([{"condition_label": 1, "condition_name": "neoplasms"}])
    pd.concat([labels, duplicate], ignore_index=True).to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="unique condition_label"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_rejects_unknown_condition_id(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    train_path = tmp_path / "medical_tc_train.csv"
    train = pd.read_csv(train_path)
    train.loc[0, "condition_label"] = 99
    train.to_csv(train_path, index=False)

    with pytest.raises(ValueError, match="unknown condition_label.*99"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_rejects_unknown_class(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    labels_path = tmp_path / "medical_tc_labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[0, "condition_name"] = "unknown class"
    labels.to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="unknown classes"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_drops_only_rows_without_text_or_label(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    train_path = tmp_path / "medical_tc_train.csv"
    train = pd.read_csv(train_path)
    train.loc[0, "medical_abstract"] = None
    train.to_csv(train_path, index=False)

    summary = make_dataset.prepare_dataset(_config_for(tmp_path))

    assert summary.dropped_train_rows == 1
    assert summary.dropped_test_rows == 0
    assert summary.train_rows + summary.validation_rows == 49
