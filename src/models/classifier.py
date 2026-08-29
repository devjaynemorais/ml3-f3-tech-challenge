"""Model Strategies and the common sklearn text-classification pipeline."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.features.text_preprocessing import TextPreprocessor
from src.utils.config_loader import ExperimentConfig, FeatureConfig, ModelConfig


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
    """Own chi-squared selection, dense conversion and Gradient Boosting."""

    def build_steps(self, config: ModelConfig) -> list[tuple[str, BaseEstimator]]:
        """Create the complete Gradient Boosting-specific suffix."""
        parameters = config.gradient_boosting
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


class ModelFactory:
    """Create one supported model Strategy by name."""

    @classmethod
    def create(cls, model_type: str) -> ModelStrategy:
        """Return the requested Strategy or reject the model type."""
        strategies: dict[str, type[ModelStrategy]] = {
            "logistic_regression": LogisticRegressionStrategy,
            "random_forest": RandomForestStrategy,
            "gradient_boosting": GradientBoostingStrategy,
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


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    """Compose preprocessing, TF-IDF and the selected model Strategy."""
    common_steps: list[tuple[str, BaseEstimator]] = [
        ("preprocessor", TextPreprocessor(config.preprocessing)),
        ("tfidf", build_tfidf(config.features)),
    ]
    strategy = ModelFactory.create(config.model.type)
    return Pipeline([*common_steps, *strategy.build_steps(config.model)])
