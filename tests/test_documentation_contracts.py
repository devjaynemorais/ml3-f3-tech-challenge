"""Public documentation and examples must match the five-class API."""

import json
from pathlib import Path

from src.utils.config_loader import CANONICAL_LABELS

PUBLIC_DOCS = [
    Path("README.md"),
    Path("docs/architecture.md"),
    Path("docs/model_card.md"),
]


def test_public_docs_describe_the_official_corpus_and_five_classes() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_DOCS)

    assert "medical_tc_train.csv" in combined
    assert "medical_tc_test.csv" in combined
    assert "abstracts públicos em inglês" in combined
    assert "normal` / `atencao` / `urgente" not in combined
    for label in CANONICAL_LABELS:
        assert label in combined


def test_agents_guide_has_no_obsolete_data_contract() -> None:
    guide = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "data/raw/triage_reports.csv" not in guide
    assert "As labels válidas são `normal`" not in guide
    assert "geração de dataset sintético" not in guide


def test_postman_collection_has_gets_and_five_safe_prediction_examples() -> None:
    collection = json.loads(
        Path("postman/Triage-API.postman_collection.json").read_text(encoding="utf-8")
    )
    items = collection["item"]
    gets = {
        item["request"]["url"] for item in items if item["request"]["method"] == "GET"
    }
    predictions = [item for item in items if item["request"]["method"] == "POST"]

    assert gets == {
        "{{base_url}}/",
        "{{base_url}}/health",
        "{{base_url}}/metrics",
        "{{base_url}}/docs",
    }
    assert len(predictions) == 5
    for item in predictions:
        payload = json.loads(item["request"]["body"]["raw"])
        assert payload["text"].strip()


def test_dashboard_uses_condition_language() -> None:
    dashboard = json.loads(
        Path("monitoring/grafana/dashboards/triage-api-overview.json").read_text(
            encoding="utf-8"
        )
    )

    titles = {panel["title"] for panel in dashboard["panels"]}
    assert "Distribuição de Categorias Médicas" in titles
    assert "Distribuição de Classificações de Urgência" not in titles
