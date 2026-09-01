"""Shared test helpers."""

from __future__ import annotations

import numpy as np
import spacy


def fake_nlp_with_vectors() -> spacy.language.Language:
    """Blank pipeline with a tiny synthetic vector table, for fast tests."""
    nlp = spacy.blank("en")
    words = ("first", "second", "third", "document", "repeated", "text")
    nlp.vocab.vectors.resize((len(words), 8))
    rng = np.random.default_rng(0)
    for word in words:
        nlp.vocab.set_vector(word, rng.random(8, dtype="float32"))
    return nlp
