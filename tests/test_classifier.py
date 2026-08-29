from __future__ import annotations

import numpy as np
import pytest

import src.models.classifier as classifier
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config

TRAIN_TEXTS = [
    f"class {label_index} abstract sample {sample_index}"
    for label_index in range(5)
    for sample_index in range(4)
]
TRAIN_LABELS = [label for label in CANONICAL_LABELS for _ in range(4)]


def _config(model_type: str) -> ExperimentConfig:
    config = load_config()
    preprocessing = config.preprocessing.model_copy(
        update={"steps": ["unicode_normalization", "punctuation_removal"]}
    )
    features = config.features.model_copy(
        update={"max_features": 100, "ngram_range": (1, 1), "min_df": 1}
    )
    model = config.model.model_copy(update={"type": model_type})
    model = model.model_copy(
        update={
            "random_forest": model.random_forest.model_copy(update={"n_estimators": 8}),
            "gradient_boosting": model.gradient_boosting.model_copy(
                update={"n_estimators": 5, "selection_k": 8}
            ),
        }
    )
    return config.model_copy(
        update={"preprocessing": preprocessing, "features": features, "model": model}
    )


@pytest.mark.parametrize(
    "model_type", ["logistic_regression", "random_forest", "gradient_boosting"]
)
def test_each_model_trains_and_returns_five_probabilities(model_type: str) -> None:
    pipeline = classifier.build_pipeline(_config(model_type))

    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    probabilities = pipeline.predict_proba(["class 3 abstract sample"])

    assert probabilities.shape == (1, 5)
    assert probabilities.sum() == pytest.approx(1.0)
    assert set(pipeline.classes_) == set(CANONICAL_LABELS)


def test_logistic_regression_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("logistic_regression").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert steps[0][1].get_params()["C"] == 1.0
    assert steps[0][1].get_params()["class_weight"] == "balanced"
    assert steps[0][1].get_params()["n_jobs"] == -1


def test_random_forest_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("random_forest").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert steps[0][1].get_params()["n_estimators"] == 160
    assert steps[0][1].get_params()["max_depth"] is None
    assert steps[0][1].get_params()["class_weight"] == "balanced"


def test_gradient_boosting_owns_chi2_and_dense_conversion() -> None:
    pipeline = classifier.build_pipeline(_config("gradient_boosting"))

    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    transformed = pipeline[:-1].transform(["class 2 abstract sample"])
    selector = pipeline.named_steps["feature_selection"]
    feature_count = len(pipeline.named_steps["tfidf"].get_feature_names_out())

    assert list(pipeline.named_steps) == [
        "preprocessor",
        "tfidf",
        "feature_selection",
        "to_dense",
        "classifier",
    ]
    assert selector.k_ == min(8, feature_count)
    assert isinstance(transformed, np.ndarray)


def test_model_factory_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown model type"):
        classifier.ModelFactory.create("unknown")


def test_model_strategies_satisfy_runtime_protocol() -> None:
    strategies = [
        classifier.ModelFactory.create(model_type)
        for model_type in (
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        )
    ]

    assert all(
        isinstance(strategy, classifier.ModelStrategy) for strategy in strategies
    )
