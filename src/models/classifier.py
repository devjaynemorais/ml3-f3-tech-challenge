"""Pipeline do classificador de texto: TF-IDF + modelo leve (sklearn)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.text_preprocessing import clean_text


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Monta o Pipeline sklearn `tfidf -> classifier` a partir do config.yaml."""
    features_cfg = cfg["features"]
    model_cfg = cfg["model"]

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        max_features=features_cfg["max_features"],
        ngram_range=tuple(features_cfg["ngram_range"]),
        min_df=features_cfg["min_df"],
    )

    model_type = model_cfg["type"]
    if model_type == "random_forest":
        params = model_cfg["random_forest"]
        classifier = RandomForestClassifier(**params)
    elif model_type == "logistic_regression":
        params = model_cfg["logistic_regression"]
        classifier = LogisticRegression(**params)
    else:
        raise ValueError(f"model.type desconhecido: {model_type!r}")

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
