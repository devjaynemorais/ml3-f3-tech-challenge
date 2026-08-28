from __future__ import annotations

from dataclasses import dataclass

import pytest

import src.features.text_preprocessing as preprocessing
from src.utils.config_loader import PreprocessingConfig


@dataclass
class _FakeToken:
    text: str
    lemma_: str


class _FakeNlp:
    _lemmas = {"patients": "patient", "were": "be", "running": "run"}

    def __call__(self, text: str) -> list[_FakeToken]:
        return [
            _FakeToken(token, self._lemmas.get(token, token)) for token in text.split()
        ]


def _config(
    steps: list[str] | None = None, *, preserve_numbers: bool = True
) -> PreprocessingConfig:
    return PreprocessingConfig(
        steps=steps
        or [
            "unicode_normalization",
            "punctuation_removal",
            "lemmatization",
            "stopword_removal",
        ],
        language="english",
        spacy_model="en_core_web_sm",
        preserve_numbers=preserve_numbers,
    )


def test_unicode_strategy_normalizes_case_accents_and_whitespace() -> None:
    strategy = preprocessing.UnicodeNormalizationStrategy()

    assert strategy.transform("  CAFÉ\tPatient  220  ") == "cafe patient 220"


def test_punctuation_strategy_preserves_letters_and_numbers() -> None:
    strategy = preprocessing.PunctuationRemovalStrategy(preserve_numbers=True)

    assert strategy.transform("patient's dose: 220 mg.") == "patient s dose 220 mg"


def test_punctuation_strategy_can_remove_numbers_from_config() -> None:
    strategy = preprocessing.PunctuationRemovalStrategy(preserve_numbers=False)

    assert strategy.transform("dose 220 mg") == "dose mg"


def test_lemmatization_strategy_uses_model_lemmas() -> None:
    strategy = preprocessing.LemmatizationStrategy(_FakeNlp())

    assert strategy.transform("patients were running") == "patient be run"


def test_stopword_strategy_filters_lemmas_and_one_character_tokens() -> None:
    strategy = preprocessing.StopwordRemovalStrategy({"be", "at"})

    assert strategy.transform("patient be running at x 220 mg") == (
        "patient running 220 mg"
    )


def test_text_preprocessor_applies_configured_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preprocessing, "_load_spacy_model", lambda _name: _FakeNlp())
    monkeypatch.setattr(
        preprocessing, "_load_stopwords", lambda _language: {"be", "at"}
    )
    transformer = preprocessing.TextPreprocessor(_config())

    assert transformer.transform(["  PÁTIENTS were RUNNING at 220 mg.  "]) == [
        "patient run 220 mg"
    ]


def test_strategies_satisfy_runtime_protocol() -> None:
    strategies = [
        preprocessing.UnicodeNormalizationStrategy(),
        preprocessing.PunctuationRemovalStrategy(True),
        preprocessing.LemmatizationStrategy(_FakeNlp()),
        preprocessing.StopwordRemovalStrategy(set()),
    ]

    assert all(
        isinstance(strategy, preprocessing.PreprocessingStrategy)
        for strategy in strategies
    )


def test_factory_rejects_unknown_step() -> None:
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        preprocessing.PreprocessingFactory.create("unknown", _config())


def test_factory_reports_missing_stopwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingStopwords:
        @staticmethod
        def words(_language: str) -> list[str]:
            raise LookupError("resource missing")

    monkeypatch.setattr(preprocessing, "stopwords", _MissingStopwords())

    with pytest.raises(RuntimeError, match="NLTK stopwords.*make install"):
        preprocessing.PreprocessingFactory.create("stopword_removal", _config())


def test_factory_reports_missing_spacy_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_name: str, *, disable: list[str]) -> None:
        raise OSError(f"missing with {disable}")

    monkeypatch.setattr(preprocessing.spacy, "load", _missing)

    with pytest.raises(RuntimeError, match="spaCy model.*make install"):
        preprocessing.PreprocessingFactory.create("lemmatization", _config())
