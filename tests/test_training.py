from pathlib import Path

import pandas as pd
import pytest

import src.training.trainer as trainer
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config


def _config_for(root: Path) -> ExperimentConfig:
    config = load_config()
    data = config.data.model_copy(
        update={
            "raw_path": root / "raw-does-not-exist",
            "processed_path": root / "processed",
        }
    )
    preprocessing = config.preprocessing.model_copy(
        update={"steps": ["unicode_normalization", "punctuation_removal"]}
    )
    features = config.features.model_copy(update={"min_df": 1, "ngram_range": (1, 1)})
    return config.model_copy(
        update={"data": data, "preprocessing": preprocessing, "features": features}
    )


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                f"class{class_index} abstract sample{sample_index}"
                for class_index in range(5)
                for sample_index in range(4)
            ],
            "label": [label for label in CANONICAL_LABELS for _ in range(4)],
        }
    )


def _write_training_split(config: ExperimentConfig) -> None:
    config.data.processed_path.mkdir(parents=True)
    path = config.data.processed_path / config.data.train_output_file
    _training_frame().to_csv(path, index=False)


def test_train_from_processed_uses_only_training_split(tmp_path: Path) -> None:
    config = _config_for(tmp_path)
    _write_training_split(config)

    pipeline, metadata = trainer.train_from_processed(config)

    assert metadata["schema_version"] == 2
    assert metadata["model_type"] == "logistic_regression"
    assert metadata["model_parameters"]["C"] == 0.3
    assert metadata["preprocessing_strategies"] == config.preprocessing.steps
    assert metadata["classes"] == CANONICAL_LABELS
    assert metadata["n_train_samples"] == 20
    assert set(pipeline.classes_) == set(CANONICAL_LABELS)


def test_train_from_processed_rejects_unknown_label(tmp_path: Path) -> None:
    config = _config_for(tmp_path)
    _write_training_split(config)
    path = config.data.processed_path / config.data.train_output_file
    frame = pd.read_csv(path)
    frame.loc[0, "label"] = "unknown class"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unknown labels"):
        trainer.train_from_processed(config)
