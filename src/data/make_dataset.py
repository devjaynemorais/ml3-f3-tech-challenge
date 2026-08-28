"""Validate the Medical Abstracts corpus and persist canonical splits."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import DataConfig, ExperimentConfig, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSplits:
    """Canonical official splits and their dropped-row counts."""

    train: pd.DataFrame
    test: pd.DataFrame
    dropped_train_rows: int
    dropped_test_rows: int


@dataclass(frozen=True)
class SplitSummary:
    """Paths and counts produced by one dataset preparation run."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int
    dropped_train_rows: int
    dropped_test_rows: int


def _source_paths(config: DataConfig) -> dict[str, Path]:
    """Resolve every required source file."""
    return {
        "train": config.raw_path / config.train_file,
        "test": config.raw_path / config.test_file,
        "labels": config.raw_path / config.labels_file,
    }


def _ensure_source_files(paths: dict[str, Path]) -> None:
    """Fail with every missing corpus path in one actionable message."""
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing Medical Abstracts files: {joined}")


def _read_source(path: Path, required: set[str]) -> pd.DataFrame:
    """Read one CSV and validate its required schema."""
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
    return frame


def _validate_label_mapping(labels: pd.DataFrame, config: DataConfig) -> dict:
    """Validate the label table and return its identifier-to-name mapping."""
    id_column = config.labels_id_column
    name_column = config.labels_name_column
    if labels[[id_column, name_column]].isna().any().any():
        raise ValueError("label mapping contains missing identifiers or classes")
    if labels[id_column].duplicated().any():
        raise ValueError("label mapping must have unique condition_label values")
    if labels[name_column].duplicated().any():
        raise ValueError("label mapping must have unique condition_name values")
    _validate_mapped_classes(labels[name_column].tolist(), config.labels)
    return dict(zip(labels[id_column], labels[name_column], strict=True))


def _validate_mapped_classes(actual: list[str], expected: list[str]) -> None:
    """Reject unknown or missing classes in the mapping table."""
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise ValueError(f"label mapping contains unknown classes: {unknown}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"label mapping is missing canonical classes: {missing}")


def _validate_condition_ids(
    frame: pd.DataFrame, mapping: dict, config: DataConfig, split_name: str
) -> None:
    """Reject source identifiers that are absent from the label mapping."""
    source_ids = set(frame[config.source_label_column].dropna())
    unknown = sorted(source_ids - set(mapping))
    if unknown:
        raise ValueError(f"{split_name} has unknown condition_label values: {unknown}")


def _canonicalize_split(
    frame: pd.DataFrame, mapping: dict, config: DataConfig, split_name: str
) -> tuple[pd.DataFrame, int]:
    """Map source fields to text/label and drop only missing records."""
    _validate_condition_ids(frame, mapping, config, split_name)
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
    """Load, validate and canonicalize the official train and test files."""
    paths = _source_paths(config)
    _ensure_source_files(paths)
    split_schema = {config.source_text_column, config.source_label_column}
    label_schema = {config.labels_id_column, config.labels_name_column}
    train = _read_source(paths["train"], split_schema)
    test = _read_source(paths["test"], split_schema)
    labels = _read_source(paths["labels"], label_schema)
    mapping = _validate_label_mapping(labels, config)
    train_clean, train_dropped = _canonicalize_split(train, mapping, config, "train")
    test_clean, test_dropped = _canonicalize_split(test, mapping, config, "test")
    return CorpusSplits(train_clean, test_clean, train_dropped, test_dropped)


def split_training_data(
    training: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create validation only from the official training split."""
    train, validation = train_test_split(
        training,
        test_size=config.split.validation_size,
        random_state=config.split.random_state,
        stratify=training[config.data.label_column],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def _output_paths(config: DataConfig) -> tuple[Path, Path, Path]:
    """Resolve canonical processed split paths."""
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
    """Persist all canonical splits together."""
    config.processed_path.mkdir(parents=True, exist_ok=True)
    train_path, validation_path, test_path = _output_paths(config)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, validation_path, test_path


def prepare_dataset(config: ExperimentConfig) -> SplitSummary:
    """Prepare and persist train, validation and untouched official test data."""
    corpus = load_medical_corpus(config.data)
    train, validation = split_training_data(corpus.train, config)
    paths = persist_splits(train, validation, corpus.test, config.data)
    return SplitSummary(
        *paths,
        len(train),
        len(validation),
        len(corpus.test),
        corpus.dropped_train_rows,
        corpus.dropped_test_rows,
    )


def main() -> None:
    """CLI entry point used by Make and Airflow."""
    summary = prepare_dataset(load_config())
    logger.info(
        "Processed splits saved: train=%d validation=%d test=%d; dropped=%d/%d",
        summary.train_rows,
        summary.validation_rows,
        summary.test_rows,
        summary.dropped_train_rows,
        summary.dropped_test_rows,
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
