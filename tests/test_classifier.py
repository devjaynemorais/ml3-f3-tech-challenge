from __future__ import annotations

import numpy as np
import pytest
import spacy

import src.models.classifier as classifier
from src.utils.config_loader import ExperimentConfig, load_config
from tests.conftest import fake_nlp_with_vectors

TRAIN_TEXTS = [
    f"class {label_index} abstract sample {sample_index}"
    for label_index in range(5)
    for sample_index in range(4)
]
TRAIN_LABELS = np.asarray(
    [[int(row % 5 == column) for column in range(5)] for row in range(20)]
)


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
            "linear_svm": model.linear_svm.model_copy(update={"calibration_cv": 2}),
        }
    )
    return config.model_copy(
        update={"preprocessing": preprocessing, "features": features, "model": model}
    )


@pytest.mark.parametrize(
    "model_type",
    [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "linear_svm",
        "complement_nb",
    ],
)
def test_each_model_trains_and_returns_five_probabilities(model_type: str) -> None:
    pipeline = classifier.build_pipeline(_config(model_type))

    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    probabilities = pipeline.predict_proba(["class 3 abstract sample"])

    assert probabilities.shape == (1, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_logistic_regression_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("logistic_regression").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert steps[0][1].get_params()["C"] == 0.3
    assert steps[0][1].get_params()["class_weight"] == "balanced"
    assert steps[0][1].get_params()["n_jobs"] == 1


def test_random_forest_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("random_forest").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert steps[0][1].get_params()["n_estimators"] == 160
    assert steps[0][1].get_params()["max_depth"] is None
    assert steps[0][1].get_params()["class_weight"] == "balanced"


def test_linear_svm_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("linear_svm").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    calibrated = steps[0][1]
    assert calibrated.cv == config.linear_svm.calibration_cv
    assert calibrated.estimator.C == config.linear_svm.C
    assert calibrated.estimator.class_weight == "balanced"


def test_complement_nb_strategy_uses_configured_parameters() -> None:
    config = load_config().model
    steps = classifier.ModelFactory.create("complement_nb").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert steps[0][1].get_params()["alpha"] == config.complement_nb.alpha
    assert steps[0][1].get_params()["norm"] == config.complement_nb.norm


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


def test_gradient_boosting_ignores_ambient_gpu_when_device_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset TRAINING_DEVICE must stay on sklearn even on a GPU-equipped machine."""
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)
    monkeypatch.setattr("src.utils.device._nvidia_gpu_present", lambda: True)

    config = load_config().model
    steps = classifier.ModelFactory.create("gradient_boosting").build_steps(config)

    expected = ["feature_selection", "to_dense", "classifier"]
    assert [name for name, _ in steps] == expected
    assert type(steps[-1][1]).__module__.startswith("sklearn")


def test_gradient_boosting_uses_xgboost_when_gpu_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "cuda")
    monkeypatch.setattr("src.utils.device._nvidia_gpu_present", lambda: True)

    config = load_config().model
    steps = classifier.ModelFactory.create("gradient_boosting").build_steps(config)

    assert [name for name, _ in steps] == ["classifier"]
    assert type(steps[-1][1]).__module__.startswith("xgboost")
    assert steps[-1][1].get_params()["device"] == "cuda"


def test_gradient_boosting_raises_when_gpu_requested_but_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "gpu")
    monkeypatch.setattr("src.utils.device._nvidia_gpu_present", lambda: False)

    config = load_config().model
    with pytest.raises(RuntimeError, match="nenhuma GPU"):
        classifier.ModelFactory.create("gradient_boosting").build_steps(config)


def test_build_pipeline_uses_embeddings_feature_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spacy, "load", lambda *_a, **_kw: fake_nlp_with_vectors())
    config = _config("logistic_regression")
    features = config.features.model_copy(update={"type": "embeddings"})
    config = config.model_copy(update={"features": features})

    pipeline = classifier.build_pipeline(config)
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    probabilities = pipeline.predict_proba(["class 3 abstract sample"])

    assert list(pipeline.named_steps) == ["preprocessor", "embeddings", "classifier"]
    assert probabilities.shape == (1, 5)


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
            "linear_svm",
            "complement_nb",
        )
    ]

    assert all(
        isinstance(strategy, classifier.ModelStrategy) for strategy in strategies
    )
