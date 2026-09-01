"""Model Strategies and the common sklearn text-classification pipeline."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.features.embeddings import SpacyEmbeddingVectorizer
from src.features.text_preprocessing import TextPreprocessor
from src.utils.config_loader import (
    ExperimentConfig,
    FeatureConfig,
    GradientBoostingConfig,
    ModelConfig,
)
from src.utils.device import gpu_explicitly_requested, resolve_training_device


class AdaptiveChi2Selector(BaseEstimator, TransformerMixin):
    """Select up to ``max_k`` non-negative features with chi-squared scores."""

    def __init__(self, max_k: int) -> None:
        self.max_k = max_k

    def fit(self, features: Any, labels: Any) -> AdaptiveChi2Selector:
        """Fit chi-squared selection with a dataset-safe feature count."""
        self.k_ = min(self.max_k, features.shape[1])
        self.selector_ = SelectKBest(score_func=chi2, k=self.k_)
        self.selector_.fit(features, labels)
        return self

    def transform(self, features: Any) -> Any:
        """Apply the fitted feature selection."""
        check_is_fitted(self, "selector_")
        return self.selector_.transform(features)


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse selected features to the dense matrix GB requires."""

    def fit(self, features: Any, labels: Any = None) -> DenseTransformer:
        """Keep sklearn compatibility without fitted state."""
        self.is_fitted_ = True
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Return a dense NumPy array."""
        if sparse.issparse(features):
            return features.toarray()
        return np.asarray(features)


@runtime_checkable
class ModelStrategy(Protocol):
    """Minimum contract for a configured classifier Strategy."""

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Build classifier-specific sklearn pipeline steps."""
        ...


class LogisticRegressionStrategy:
    """Build the Logistic Regression classifier step."""

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create Logistic Regression from its typed configuration."""
        parameters = config.logistic_regression.model_dump()
        return [("classifier", LogisticRegression(**parameters))]


class RandomForestStrategy:
    """Build the Random Forest classifier step."""

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create Random Forest from its typed configuration."""
        parameters = config.random_forest.model_dump()
        return [("classifier", RandomForestClassifier(**parameters))]


class GradientBoostingStrategy:
    """Own chi-squared selection, dense conversion and Gradient Boosting.

    Switches to GPU-accelerated XGBoost only on an EXPLICIT
    TRAINING_DEVICE=cuda/gpu, never via auto-detection: scikit-learn's
    GradientBoostingClassifier has no GPU backend, but swapping in XGBoost is
    an algorithm change, not just a speed knob, so picking it based on
    ambient hardware would make training results depend on which machine
    happens to run it. XGBoost also handles the sparse TF-IDF matrix
    natively, so the chi-squared selection and dense conversion steps
    (needed only for sklearn's dense-only GB) are skipped on that path.
    """

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create the complete Gradient Boosting-specific suffix."""
        parameters = config.gradient_boosting
        if gpu_explicitly_requested():
            resolve_training_device()  # raises a clear error if no GPU is present
            return [("classifier", _build_xgboost_classifier(parameters))]
        return self._sklearn_steps(parameters)

    def _sklearn_steps(
        self, parameters: GradientBoostingConfig
    ) -> list[tuple[str, BaseEstimator]]:
        """Build the CPU-only sklearn Gradient Boosting suffix."""
        classifier = GradientBoostingClassifier(
            n_estimators=parameters.n_estimators,
            learning_rate=parameters.learning_rate,
            max_depth=parameters.max_depth,
            random_state=parameters.random_state,
        )
        return [
            ("feature_selection", AdaptiveChi2Selector(parameters.selection_k)),
            ("to_dense", DenseTransformer()),
            ("classifier", classifier),
        ]


def _build_xgboost_classifier(parameters: GradientBoostingConfig) -> XGBClassifier:
    """Build the GPU-accelerated XGBoost classifier for one binary label."""
    return XGBClassifier(
        n_estimators=parameters.n_estimators,
        learning_rate=parameters.learning_rate,
        max_depth=parameters.max_depth,
        random_state=parameters.random_state,
        tree_method="hist",
        device="cuda",
    )


class LinearSvmStrategy:
    """Build a calibrated Linear SVM classifier step.

    ``LinearSVC`` has no ``predict_proba``; ``CalibratedClassifierCV`` adds
    Platt-scaled probabilities on top, required by the serving contract.
    """

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create a calibrated Linear SVM from its typed configuration."""
        parameters = config.linear_svm.model_dump()
        cv = parameters.pop("calibration_cv")
        base = LinearSVC(**parameters)
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        return [("classifier", calibrated)]


class ComplementNbStrategy:
    """Build the Complement Naive Bayes classifier step.

    Requires non-negative input — pairs with TF-IDF, not word embeddings.
    """

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create Complement Naive Bayes from its typed configuration."""
        parameters = config.complement_nb.model_dump()
        return [("classifier", ComplementNB(**parameters))]


class ModelFactory:
    """Create one supported model Strategy by name."""

    @classmethod
    def create(cls, model_type: str) -> ModelStrategy:
        """Return the requested Strategy or reject the model type."""
        strategies: dict[str, type[ModelStrategy]] = {
            "logistic_regression": LogisticRegressionStrategy,
            "random_forest": RandomForestStrategy,
            "gradient_boosting": GradientBoostingStrategy,
            "linear_svm": LinearSvmStrategy,
            "complement_nb": ComplementNbStrategy,
        }
        if model_type not in strategies:
            raise ValueError(f"unknown model type: {model_type!r}")
        return strategies[model_type]()


def build_tfidf(config: FeatureConfig) -> TfidfVectorizer:
    """Build the configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
    )


def build_features_step(config: FeatureConfig) -> tuple[str, BaseEstimator]:
    """Build the configured feature Strategy: TF-IDF or word embeddings."""
    if config.type == "embeddings":
        return "embeddings", SpacyEmbeddingVectorizer(config.embeddings.spacy_model)
    return "tfidf", build_tfidf(config)


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """Compose preprocessing, features and one binary classifier per label."""
    common_steps: list[tuple[str, BaseEstimator]] = [
        ("preprocessor", TextPreprocessor(config.preprocessing)),
        build_features_step(config.features),
    ]
    strategy = ModelFactory.create(config.model.type)
    model_steps = strategy.build_steps(config.model)
    step_name, estimator = model_steps[-1]
    if step_name != "classifier":
        raise ValueError("the final model step must be named 'classifier'")
    multilabel_steps = [*model_steps[:-1], (step_name, OneVsRestClassifier(estimator))]
    return Pipeline([*common_steps, *multilabel_steps])
