"""Reproducibility contracts for local, Docker, Airflow and CI environments."""

from pathlib import Path

import yaml

SPACY_MODEL = "en_core_web_sm-3.8.0"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_make_install_provisions_nlp_resources() -> None:
    makefile = _read("Makefile")

    assert "nltk.downloader stopwords" in makefile
    assert SPACY_MODEL in makefile


def test_api_image_provisions_nlp_resources() -> None:
    dockerfile = _read("Dockerfile")

    assert "nltk.downloader" in dockerfile
    assert SPACY_MODEL in dockerfile
    assert "NLTK_DATA" in dockerfile


def test_airflow_uses_reproducible_custom_image() -> None:
    dockerfile = _read("Dockerfile.airflow")
    compose_text = _read("docker-compose.airflow.yml")
    compose = yaml.safe_load(compose_text)
    common = compose["x-airflow-common"]

    assert "apache/airflow:2.9.3-python3.11" in dockerfile
    assert SPACY_MODEL in dockerfile
    assert common["build"]["dockerfile"] == "Dockerfile.airflow"
    assert "_PIP_ADDITIONAL_REQUIREMENTS" not in common["environment"]


def test_ci_provisions_nlp_resources() -> None:
    workflow = _read(".github/workflows/ci.yml")

    assert "nltk.downloader stopwords" in workflow
    assert SPACY_MODEL in workflow
