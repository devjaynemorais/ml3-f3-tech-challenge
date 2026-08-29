"""Document embedding features via averaged pretrained spaCy word vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


class SpacyEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Represent each document as its mean pretrained word-vector embedding.

    Unlike TF-IDF, static word vectors place semantically related terms
    (e.g. "carcinoma" and "tumor") close in space even without exact
    vocabulary overlap. Outputs a dense array — pairs with linear
    classifiers; incompatible with the chi2-based feature selection used by
    the Gradient Boosting Strategy, which requires non-negative input.
    """

    def __init__(self, spacy_model: str) -> None:
        self.spacy_model = spacy_model

    def fit(self, texts: Any, labels: Any = None) -> SpacyEmbeddingVectorizer:
        """Load the pretrained spaCy pipeline; no corpus-specific fitting."""
        self._nlp = spacy.load(
            self.spacy_model, disable=["tagger", "parser", "ner", "lemmatizer"]
        )
        return self

    def transform(self, texts: Any) -> np.ndarray:
        """Return one averaged word-vector embedding per document."""
        return np.vstack([doc.vector for doc in self._nlp.pipe(texts)])
