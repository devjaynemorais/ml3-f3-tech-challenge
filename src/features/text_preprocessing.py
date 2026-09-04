"""Composable preprocessing Strategies for English medical abstracts."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol, runtime_checkable

import spacy
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode

from src.utils.config_loader import PreprocessingConfig

_WHITESPACE_RE = re.compile(r"\s+")


class LemmaToken(Protocol):
    """Token fields required from an NLP pipeline."""

    text: str
    lemma_: str


class NlpPipeline(Protocol):
    """Callable lemmatization pipeline used by the Strategy."""

    def __call__(self, text: str) -> Iterable[LemmaToken]:
        """Tokenize and annotate text."""
        ...


@runtime_checkable
class PreprocessingStrategy(Protocol):
    """Minimum contract implemented by every text preprocessing step."""

    def transform(self, text: str) -> str:
        """Transform one text value."""
        ...


def _normalize_spaces(text: str) -> str:
    """Collapse whitespace and trim text."""
    return _WHITESPACE_RE.sub(" ", text).strip()


class UnicodeNormalizationStrategy:
    """Lowercase, transliterate Unicode and normalize whitespace."""

    def transform(self, text: str) -> str:
        """Normalize one text without removing semantic tokens."""
        return _normalize_spaces(unidecode(text.lower()))


class PunctuationRemovalStrategy:
    """Remove punctuation while optionally preserving numbers."""

    def __init__(self, preserve_numbers: bool) -> None:
        self.preserve_numbers = preserve_numbers

    def transform(self, text: str) -> str:
        """Replace disallowed characters with spaces."""
        cleaned = "".join(self._normalize_character(character) for character in text)
        return _normalize_spaces(cleaned)

    def _normalize_character(self, character: str) -> str:
        """Preserve letters, configured digits and whitespace."""
        if character.isalpha() or character.isspace():
            return character
        if self.preserve_numbers and character.isdigit():
            return character
        return " "


class LemmatizationStrategy:
    """Tokenize and lemmatize text with a preloaded spaCy pipeline."""

    def __init__(self, model_name: str, nlp: NlpPipeline | None = None) -> None:
        self.model_name = model_name
        self.nlp = nlp

    def __getstate__(self) -> dict[str, str]:
        """Persist only the model name, avoiding OS-specific spaCy paths."""
        return {"model_name": self.model_name}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a portable state, including artifacts from the old format."""
        model_name = state.get("model_name")
        if not isinstance(model_name, str):
            legacy_nlp = state["nlp"]
            metadata = legacy_nlp.meta  # type: ignore[attr-defined]
            model_name = f"{metadata['lang']}_{metadata['name']}"
        self.model_name = model_name
        self.nlp = None

    def transform(self, text: str) -> str:
        """Return whitespace-separated token lemmas."""
        if self.nlp is None:
            self.nlp = _load_spacy_model(self.model_name)
        lemmas = [token.lemma_ or token.text for token in self.nlp(text)]
        return _normalize_spaces(" ".join(lemmas))


class StopwordRemovalStrategy:
    """Remove NLTK stopwords and one-character tokens from lemmas."""

    def __init__(self, words: set[str]) -> None:
        self.words = words

    def transform(self, text: str) -> str:
        """Filter stopwords and one-character tokens."""
        tokens = [
            token
            for token in text.split()
            if token.lower() not in self.words and len(token) > 1
        ]
        return " ".join(tokens)


def _load_spacy_model(model_name: str) -> NlpPipeline:
    """Load the configured spaCy model or fail without runtime download."""
    try:
        return spacy.load(model_name, disable=["parser", "ner", "textcat"])
    except OSError as error:
        message = f"spaCy model {model_name!r} is unavailable; run `make install`"
        raise RuntimeError(message) from error


def _load_stopwords(language: str) -> set[str]:
    """Load configured NLTK stopwords or fail without runtime download."""
    try:
        return set(stopwords.words(language))
    except LookupError as error:
        message = f"NLTK stopwords for {language!r} are unavailable; run `make install`"
        raise RuntimeError(message) from error


class PreprocessingFactory:
    """Create preprocessing Strategies by configured name."""

    @classmethod
    def create(
        cls, step_name: str, config: PreprocessingConfig
    ) -> PreprocessingStrategy:
        """Build one Strategy and reject unknown names."""
        builders = {
            "unicode_normalization": UnicodeNormalizationStrategy,
            "punctuation_removal": lambda: PunctuationRemovalStrategy(
                config.preserve_numbers
            ),
            "lemmatization": lambda: LemmatizationStrategy(config.spacy_model),
            "stopword_removal": lambda: StopwordRemovalStrategy(
                _load_stopwords(config.language)
            ),
        }
        if step_name not in builders:
            raise ValueError(f"unknown preprocessing step: {step_name!r}")
        return builders[step_name]()


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn transformer that applies configured Strategies in order."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._strategies = [
            PreprocessingFactory.create(step, config) for step in config.steps
        ]

    def fit(self, texts: Iterable[str], y: object = None) -> TextPreprocessor:
        """Keep sklearn compatibility; preprocessing has no fitted state."""
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        """Transform every input with the persisted Strategy list."""
        return [self._transform_one(text) for text in texts]

    def _transform_one(self, text: str) -> str:
        """Apply each configured Strategy to one text."""
        for strategy in self._strategies:
            text = strategy.transform(text)
        return text


def clean_text(text: str) -> str:
    """Compatibility wrapper retained until the classifier builder is migrated."""
    return UnicodeNormalizationStrategy().transform(text)
