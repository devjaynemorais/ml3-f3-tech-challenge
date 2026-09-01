import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import make_dataset
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config


def _write_corpus(root: Path) -> None:
    labels = pd.DataFrame(
        {"condition_label": range(1, 6), "condition_name": CANONICAL_LABELS}
    )
    train_rows = [
        {"condition_label": label_id, "medical_abstract": f"train-{label_id}-{i}"}
        for label_id in range(1, 6)
        for i in range(10)
    ]
    train_rows.extend(
        [
            {"condition_label": 1, "medical_abstract": "shared multilabel train"},
            {"condition_label": 2, "medical_abstract": "shared multilabel train"},
            {"condition_label": 1, "medical_abstract": "cross split abstract"},
            {"condition_label": 3, "medical_abstract": "cross split abstract"},
        ]
    )
    test_rows = [
        {"condition_label": label_id, "medical_abstract": f"test-{label_id}-{i}"}
        for label_id in range(1, 6)
        for i in range(2)
    ]
    test_rows.extend(
        [
            {"condition_label": 4, "medical_abstract": "cross split abstract"},
            {"condition_label": 3, "medical_abstract": "shared multilabel test"},
            {"condition_label": 5, "medical_abstract": "shared multilabel test"},
        ]
    )
    labels.to_csv(root / "medical_tc_labels.csv", index=False)
    pd.DataFrame(train_rows).to_csv(root / "medical_tc_train.csv", index=False)
    pd.DataFrame(test_rows).to_csv(root / "medical_tc_test.csv", index=False)


def _config_for(root: Path) -> ExperimentConfig:
    config = load_config()
    data = config.data.model_copy(
        update={"raw_path": root, "processed_path": root / "processed"}
    )
    return config.model_copy(update={"data": data})


def test_prepare_dataset_aggregates_each_split_independently(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    summary = make_dataset.prepare_dataset(_config_for(tmp_path))
    combined_train = pd.concat(
        [pd.read_csv(summary.train_path), pd.read_csv(summary.validation_path)]
    )
    test = pd.read_csv(summary.test_path)

    train_row = combined_train.loc[combined_train.text.eq("shared multilabel train")]
    test_row = test.loc[test.text.eq("shared multilabel test")]
    assert len(train_row) == 1
    assert train_row[CANONICAL_LABELS[:2]].iloc[0].tolist() == [1, 1]
    assert len(test_row) == 1
    selected = test_row[[CANONICAL_LABELS[2], CANONICAL_LABELS[4]]]
    assert selected.iloc[0].tolist() == [1, 1]


def test_cross_split_abstract_is_removed_only_from_training(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    summary = make_dataset.prepare_dataset(_config_for(tmp_path))
    training = pd.concat(
        [pd.read_csv(summary.train_path), pd.read_csv(summary.validation_path)]
    )
    test = pd.read_csv(summary.test_path)

    assert "cross split abstract" not in set(training.text)
    kept = test.loc[test.text.eq("cross split abstract")].iloc[0]
    assert kept[CANONICAL_LABELS[3]] == 1
    assert kept[CANONICAL_LABELS[0]] == 0
    assert kept[CANONICAL_LABELS[2]] == 0
    assert summary.overlap_groups_removed_from_train == 1
    assert summary.overlap_rows_removed_from_train == 2
    audit = json.loads(
        (summary.train_path.parent / make_dataset.DATASET_AUDIT_FILE).read_text()
    )
    assert audit["final_overlap_groups"] == 0
    assert audit["labels_transferred_between_splits"] is False


def test_validation_is_deterministic_and_has_no_test_overlap(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    config = _config_for(tmp_path)
    first = make_dataset.prepare_dataset(config)
    first_validation = pd.read_csv(first.validation_path)
    second = make_dataset.prepare_dataset(config)
    second_validation = pd.read_csv(second.validation_path)
    test = pd.read_csv(second.test_path)

    assert first_validation.equals(second_validation)
    assert not set(first_validation.text) & set(test.text)
    assert set(CANONICAL_LABELS).issubset(first_validation.columns)


def test_prepare_dataset_rejects_missing_source_column(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    path = tmp_path / "medical_tc_train.csv"
    pd.read_csv(path).drop(columns="medical_abstract").to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns.*medical_abstract"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_rejects_non_unique_label_mapping(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    path = tmp_path / "medical_tc_labels.csv"
    labels = pd.read_csv(path)
    duplicate = pd.DataFrame([{"condition_label": 1, "condition_name": "neoplasms"}])
    pd.concat([labels, duplicate], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique condition_label"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_rejects_unknown_condition_id(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    path = tmp_path / "medical_tc_train.csv"
    train = pd.read_csv(path)
    train.loc[0, "condition_label"] = 99
    train.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown condition_label.*99"):
        make_dataset.prepare_dataset(_config_for(tmp_path))


def test_prepare_dataset_drops_rows_without_text(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    path = tmp_path / "medical_tc_train.csv"
    train = pd.read_csv(path)
    train.loc[0, "medical_abstract"] = None
    train.to_csv(path, index=False)
    summary = make_dataset.prepare_dataset(_config_for(tmp_path))
    assert summary.dropped_train_rows == 1
