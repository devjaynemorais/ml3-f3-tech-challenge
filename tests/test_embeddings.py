from __future__ import annotations

import pytest
import spacy

from src.features.embeddings import SpacyEmbeddingVectorizer
from tests.conftest import fake_nlp_with_vectors


@pytest.fixture(autouse=True)
def _fake_spacy_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the real (heavy) model for a tiny synthetic pipeline in this file."""
    monkeypatch.setattr(
        spacy, "load", lambda *_args, **_kwargs: fake_nlp_with_vectors()
    )


def test_transform_returns_one_vector_per_document() -> None:
    vectorizer = SpacyEmbeddingVectorizer(spacy_model="en_core_sci_md")
    vectorizer.fit(["first document", "second document"])

    vectors = vectorizer.transform(["first document", "second document", "third"])

    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0


def test_transform_is_deterministic_for_the_same_text() -> None:
    vectorizer = SpacyEmbeddingVectorizer(spacy_model="en_core_sci_md")
    vectorizer.fit(["irrelevant for a blank pipeline"])

    first = vectorizer.transform(["repeated text"])
    second = vectorizer.transform(["repeated text"])

    assert (first == second).all()
