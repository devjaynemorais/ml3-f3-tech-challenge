"""Prepare leakage-safe multilabel splits from the flattened medical corpus."""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config_loader import DataConfig, ExperimentConfig, load_config

logger = logging.getLogger(__name__)
_GROUP_KEY = "_abstract_group_key"
DATASET_AUDIT_FILE = "dataset_audit.json"


@dataclass(frozen=True)
class CorpusSplits:
    """Canonical official row-level splits and their dropped-row counts."""

    train: pd.DataFrame
    test: pd.DataFrame
    dropped_train_rows: int
    dropped_test_rows: int


@dataclass(frozen=True)
class SplitSummary:
    """Paths and audit counts produced by dataset preparation."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    dropped_train_rows: int
    dropped_test_rows: int
    overlap_groups_removed_from_train: int
    overlap_rows_removed_from_train: int


def normalize_abstract_key(text: str) -> str:
    """Build a conservative comparison key without linguistic preprocessing."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(normalized.split())


def _source_paths(config: DataConfig) -> dict[str, Path]:
    return {
        "train": config.raw_path / config.train_file,
        "test": config.raw_path / config.test_file,
        "labels": config.raw_path / config.labels_file,
    }


def _read_source(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing Medical Abstracts file: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
    return frame


def _validate_mapped_classes(actual: list[str], expected: list[str]) -> None:
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise ValueError(f"label mapping contains unknown classes: {unknown}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"label mapping is missing canonical classes: {missing}")


def _validate_label_mapping(labels: pd.DataFrame, config: DataConfig) -> dict:
    id_column, name_column = config.labels_id_column, config.labels_name_column
    if labels[[id_column, name_column]].isna().any().any():
        raise ValueError("label mapping contains missing identifiers or classes")
    if labels[id_column].duplicated().any():
        raise ValueError("label mapping must have unique condition_label values")
    if labels[name_column].duplicated().any():
        raise ValueError("label mapping must have unique condition_name values")
    _validate_mapped_classes(labels[name_column].tolist(), config.labels)
    return dict(zip(labels[id_column], labels[name_column], strict=True))


def _canonicalize_split(
    frame: pd.DataFrame, mapping: dict, config: DataConfig, split_name: str
) -> tuple[pd.DataFrame, int]:
    source_ids = set(frame[config.source_label_column].dropna())
    unknown = sorted(source_ids - set(mapping))
    if unknown:
        raise ValueError(f"{split_name} has unknown condition_label values: {unknown}")
    canonical = pd.DataFrame(
        {
            config.text_column: frame[config.source_text_column],
            config.label_column: frame[config.source_label_column].map(mapping),
        }
    )
    missing = canonical.isna().any(axis=1)
    blank = canonical[config.text_column].fillna("").astype(str).str.strip().eq("")
    clean = canonical.loc[~(missing | blank)].reset_index(drop=True)
    return clean, int((missing | blank).sum())


def load_medical_corpus(config: DataConfig) -> CorpusSplits:
    """Load and validate the untouched official CSV files."""
    paths = _source_paths(config)
    schema = {config.source_text_column, config.source_label_column}
    label_schema = {config.labels_id_column, config.labels_name_column}
    train = _read_source(paths["train"], schema)
    test = _read_source(paths["test"], schema)
    labels = _read_source(paths["labels"], label_schema)
    mapping = _validate_label_mapping(labels, config)
    train_clean, train_dropped = _canonicalize_split(train, mapping, config, "train")
    test_clean, test_dropped = _canonicalize_split(test, mapping, config, "test")
    return CorpusSplits(train_clean, test_clean, train_dropped, test_dropped)


def aggregate_multilabel_split(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Collapse one split independently to one row and five targets per abstract."""
    text_column, label_column = config.text_column, config.label_column
    keyed = frame.copy()
    keyed[_GROUP_KEY] = keyed[text_column].map(normalize_abstract_key)
    first_text = keyed.groupby(_GROUP_KEY, sort=False)[text_column].first()
    label_sets = keyed.groupby(_GROUP_KEY, sort=False)[label_column].agg(set)
    result = pd.DataFrame({text_column: first_text})
    for label in config.labels:
        result[label] = label_sets.map(lambda values, item=label: int(item in values))
    return result.reset_index()


def remove_train_test_overlap(
    training: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """Remove from training every abstract group present in the official test."""
    test_keys = set(test[_GROUP_KEY])
    overlap = training[_GROUP_KEY].isin(test_keys)
    removed_groups = int(training.loc[overlap, _GROUP_KEY].nunique())
    clean = training.loc[~overlap].reset_index(drop=True)
    return clean, removed_groups, int(overlap.sum())


def _multilabel_validation_indices(
    targets: np.ndarray, validation_size: float, random_state: int
) -> np.ndarray:
    """Greedily select a deterministic validation set matching label prevalence."""
    rng = np.random.default_rng(random_state)
    n_validation = max(1, round(len(targets) * validation_size))
    desired = targets.sum(axis=0) * validation_size
    selected: list[int] = []
    available = np.ones(len(targets), dtype=bool)
    current = np.zeros(targets.shape[1], dtype=float)
    tie_noise = rng.random(len(targets)) * 1e-9
    frequencies = np.maximum(targets.sum(axis=0), 1)
    for _ in range(n_validation):
        deficit = np.maximum(desired - current, 0)
        scores = (targets * (deficit / frequencies)).sum(axis=1) + tie_noise
        scores[~available] = -1
        chosen = int(np.argmax(scores))
        selected.append(chosen)
        available[chosen] = False
        current += targets[chosen]
    return np.asarray(selected)


def split_training_data(
    training: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic multilabel validation split after grouping."""
    targets = training[config.data.labels].to_numpy(dtype=int)
    validation_indices = _multilabel_validation_indices(
        targets, config.split.validation_size, config.split.random_state
    )
    validation_mask = np.zeros(len(training), dtype=bool)
    validation_mask[validation_indices] = True
    return (
        training.loc[~validation_mask].reset_index(drop=True),
        training.loc[validation_mask].reset_index(drop=True),
    )


def _output_paths(config: DataConfig) -> tuple[Path, Path, Path]:
    return (
        config.processed_path / config.train_output_file,
        config.processed_path / config.validation_output_file,
        config.processed_path / config.test_output_file,
    )


def persist_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: DataConfig,
) -> tuple[Path, Path, Path]:
    """Persist public processed columns without the internal grouping key."""
    config.processed_path.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(config)
    columns = [config.text_column, *config.labels]
    for frame, path in zip((train, validation, test), paths, strict=True):
        frame[columns].to_csv(path, index=False)
    return paths


def prepare_dataset(config: ExperimentConfig) -> SplitSummary:
    """Build independent multilabel splits and purge test overlap from training."""
    corpus = load_medical_corpus(config.data)
    test_keys = set(corpus.test[config.data.text_column].map(normalize_abstract_key))
    raw_overlap_rows = int(
        corpus.train[config.data.text_column]
        .map(normalize_abstract_key)
        .isin(test_keys)
        .sum()
    )
    training = aggregate_multilabel_split(corpus.train, config.data)
    test = aggregate_multilabel_split(corpus.test, config.data)
    training, overlap_groups, _ = remove_train_test_overlap(training, test)
    train, validation = split_training_data(training, config)
    paths = persist_splits(train, validation, test, config.data)
    audit = {
        "policy": "aggregate_splits_separately_and_remove_overlap_from_train",
        "raw_train_rows": len(corpus.train),
        "raw_test_rows": len(corpus.test),
        "overlap_groups_removed_from_train": overlap_groups,
        "overlap_rows_removed_from_train": raw_overlap_rows,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "final_overlap_groups": 0,
        "labels_transferred_between_splits": False,
    }
    (config.data.processed_path / DATASET_AUDIT_FILE).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    audit = {
        "policy": "aggregate_splits_separately_and_remove_overlap_from_train",
        "raw_train_rows": len(corpus.train),
        "raw_test_rows": len(corpus.test),
        "overlap_groups_removed_from_train": overlap_groups,
        "overlap_rows_removed_from_train": raw_overlap_rows,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "final_overlap_groups": 0,
        "labels_transferred_between_splits": False,
    }
    (config.data.processed_path / DATASET_AUDIT_FILE).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return SplitSummary(
        *paths,
        len(train),
        len(validation),
        len(test),
        corpus.dropped_train_rows,
        corpus.dropped_test_rows,
        overlap_groups,
        raw_overlap_rows,
    )


def load_processed_split(config: ExperimentConfig, filename: str) -> pd.DataFrame:
    """Load one processed multilabel split and validate binary targets."""
    path = config.data.processed_path / filename
    if not path.exists():
        raise FileNotFoundError(
            f"processed split not found at {path}; run `make dataset`"
        )
    required = {config.data.text_column, *config.data.labels}
    frame = _read_source(path, required)
    if not frame[config.data.labels].isin([0, 1]).all().all():
        raise ValueError(f"{path.name} contains non-binary multilabel targets")
    if frame[config.data.labels].sum(axis=1).eq(0).any():
        raise ValueError(f"{path.name} contains an abstract without labels")
    return frame


def main() -> None:
    """Prepare the configured multilabel dataset."""
    summary = prepare_dataset(load_config())
    logger.info(
        "Multilabel splits: train=%d validation=%d test=%d; overlap removed=%d",
        summary.train_rows,
        summary.validation_rows,
        summary.test_rows,
        summary.overlap_groups_removed_from_train,
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
