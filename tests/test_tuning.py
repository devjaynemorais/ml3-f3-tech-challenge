from __future__ import annotations

import pandas as pd
import pytest

from src.training.tuning import build_param_grid, search_best_pipeline
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config

TRAIN_TEXTS = [
    f"class {label_index} abstract sample {sample_index}"
    for label_index in range(5)
    for sample_index in range(6)
]
TRAIN_LABELS = [label for label in CANONICAL_LABELS for _ in range(6)]

_MINIMAL_GRID = {
    "tfidf": {"max_features": [50]},
    "logistic_regression": {"C": [0.5, 1.0]},
    "linear_svm": {"C": [0.1, 0.3]},
    "complement_nb": {"alpha": [0.5, 1.0]},
}


def _tuning_config(model_type: str) -> ExperimentConfig:
    config = load_config()
    preprocessing = config.preprocessing.model_copy(
        update={"steps": ["unicode_normalization", "punctuation_removal"]}
    )
    features = config.features.model_copy(
        update={"max_features": 50, "ngram_range": (1, 1), "min_df": 1}
    )
    model = config.model.model_copy(update={"type": model_type})
    model = model.model_copy(
        update={"linear_svm": model.linear_svm.model_copy(update={"calibration_cv": 2})}
    )
    tuning = config.tuning.model_copy(update={"cv_folds": 2, "grid": _MINIMAL_GRID})
    return config.model_copy(
        update={
            "preprocessing": preprocessing,
            "features": features,
            "model": model,
            "tuning": tuning,
        }
    )


def test_build_param_grid_prefixes_calibrated_svm_as_nested_estimator() -> None:
    config = _tuning_config("linear_svm")

    grid = build_param_grid(config)

    assert grid == {"tfidf__max_features": [50], "classifier__estimator__C": [0.1, 0.3]}


def test_build_param_grid_skips_tfidf_params_for_embeddings() -> None:
    config = _tuning_config("logistic_regression")
    config = config.model_copy(
        update={"features": config.features.model_copy(update={"type": "embeddings"})}
    )

    grid = build_param_grid(config)

    assert grid == {"classifier__C": [0.5, 1.0]}


def test_build_param_grid_skips_untuned_model_types() -> None:
    config = _tuning_config("random_forest")

    grid = build_param_grid(config)

    assert grid == {"tfidf__max_features": [50]}


@pytest.mark.parametrize(
    "model_type", ["logistic_regression", "linear_svm", "complement_nb"]
)
def test_search_best_pipeline_refits_winner_on_all_data(model_type: str) -> None:
    config = _tuning_config(model_type)

    result = search_best_pipeline(
        pd.Series(TRAIN_TEXTS), pd.Series(TRAIN_LABELS), config
    )

    assert set(result.pipeline.classes_) == set(CANONICAL_LABELS)
    assert 0.0 <= result.cv_mean <= 1.0
    assert result.cv_std >= 0.0
    assert result.best_params
    assert result.pipeline.get_params()["memory"] is None
