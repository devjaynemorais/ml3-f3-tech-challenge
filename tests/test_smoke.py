"""Smoke tests for imports and the project configuration."""

from src.serving.api import app
from src.utils.config_loader import load_config


def test_config_has_expected_typed_sections() -> None:
    config = load_config()

    assert config.data
    assert config.split
    assert config.preprocessing
    assert config.features
    assert config.model
    assert config.artifacts


def test_labels_follow_the_corpus_canonical_order() -> None:
    config = load_config()

    assert config.data.labels == [
        "neoplasms",
        "digestive system diseases",
        "nervous system diseases",
        "cardiovascular diseases",
        "general pathological conditions",
    ]


def test_fastapi_app_loads_with_public_routes() -> None:
    routes = set(app.openapi()["paths"])

    assert {"/", "/health", "/predict", "/metrics"}.issubset(routes)
    assert app.docs_url == "/docs"
