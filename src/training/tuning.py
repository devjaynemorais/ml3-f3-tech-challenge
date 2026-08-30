"""Hyperparameter search via stratified cross-validation, scored on F1-macro.

Runs against the exact servable pipeline (e.g. the calibrated Linear SVM,
not a bare LinearSVC) so the winning configuration is what gets deployed —
not a variant that cannot serve ``predict_proba``.
"""

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import NamedTuple

import pandas as pd
from joblib import Memory
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models.classifier import build_pipeline
from src.utils.config_loader import ExperimentConfig

_CLASSIFIER_PREFIX = {
    "logistic_regression": "classifier__",
    "complement_nb": "classifier__",
    "linear_svm": "classifier__estimator__",
}


class TuningResult(NamedTuple):
    """Winning pipeline (already refit on all provided data) and CV stats."""

    pipeline: Pipeline
    best_params: dict
    cv_mean: float
    cv_std: float


def build_param_grid(config: ExperimentConfig) -> dict[str, list]:
    """Flatten the configured grid into sklearn's step__param notation."""
    grid = config.tuning.grid
    flat: dict[str, list] = {}
    if config.features.type == "tfidf":
        for param, values in grid.get("tfidf", {}).items():
            flat[f"tfidf__{param}"] = values
    prefix = _CLASSIFIER_PREFIX.get(config.model.type)
    if prefix:
        for param, values in grid.get(config.model.type, {}).items():
            flat[f"{prefix}{param}"] = values
    return flat


def search_best_pipeline(
    texts: pd.Series, labels: pd.Series, config: ExperimentConfig
) -> TuningResult:
    """Grid-search the active Strategy; return the refit-on-all-data winner."""
    with TemporaryDirectory() as cache_dir:
        pipeline = build_pipeline(config)
        pipeline.set_params(memory=Memory(location=cache_dir, verbose=0))
        cv = StratifiedKFold(
            n_splits=config.tuning.cv_folds,
            shuffle=True,
            random_state=config.split.random_state,
        )
        search = GridSearchCV(
            pipeline,
            build_param_grid(config),
            scoring=config.tuning.scoring,
            cv=cv,
            refit=True,
        )
        search.fit(texts, labels)
        std = search.cv_results_["std_test_score"][search.best_index_]
        best_pipeline = search.best_estimator_
        best_pipeline.set_params(memory=None)  # drop the cache dir before persisting
        return TuningResult(
            best_pipeline, search.best_params_, search.best_score_, float(std)
        )
