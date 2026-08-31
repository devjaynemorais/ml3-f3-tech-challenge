import logging

import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_module
from src.utils.config_loader import CANONICAL_LABELS


class _FakePredictor:
    backend = "fake"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        scores = {label: 0.05 for label in CANONICAL_LABELS}
        scores[CANONICAL_LABELS[0]] = 0.8
        return CANONICAL_LABELS[0], scores

    def explain(self, text: str) -> dict:
        label, scores = self.predict(text)
        return {
            "label": label,
            "labels": [label],
            "scores": scores,
            "backend": self.backend,
            "preprocessed_text": text.lower(),
            "top_terms": None,
        }


class _FailingPredictor:
    backend = "fake"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        raise RuntimeError(text)

    def explain(self, text: str) -> dict:
        raise RuntimeError(text)


def _missing_predictor():
    raise FileNotFoundError("model not found")


def test_root_endpoint_reports_medical_classifier_name_and_version() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "Medical Text Classifier API",
        "version": "0.1.0",
    }


def test_health_reports_loaded_backend() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/health")

    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "backend": "fake",
    }


def test_health_is_degraded_when_loading_fails() -> None:
    with TestClient(api_module.create_app(_missing_predictor)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "model_loaded": False}


def test_app_instances_keep_predictors_isolated() -> None:
    loaded_app = api_module.create_app(lambda: _FakePredictor())
    degraded_app = api_module.create_app(_missing_predictor)

    with TestClient(loaded_app) as loaded, TestClient(degraded_app) as degraded:
        assert loaded.get("/health").json()["status"] == "ok"
        assert degraded.get("/health").json()["status"] == "degraded"


def test_predict_returns_five_canonical_scores() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.post("/predict", json={"text": "neutral abstract"})

    assert response.status_code == 200
    assert response.json()["label"] == CANONICAL_LABELS[0]
    assert response.json()["labels"] == [CANONICAL_LABELS[0]]
    assert response.json()["backend"] == "fake"
    assert list(response.json()["scores"]) == CANONICAL_LABELS
    assert all(0 <= score <= 1 for score in response.json()["scores"].values())


@pytest.mark.parametrize("text", ["", "   "])
def test_predict_rejects_empty_or_blank_text(text: str) -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.post("/predict", json={"text": text})

    assert response.status_code == 422


def test_predict_returns_503_without_predictor() -> None:
    with TestClient(api_module.create_app(_missing_predictor)) as client:
        response = client.post("/predict", json={"text": "neutral abstract"})

    assert response.status_code == 503


def test_inference_failure_returns_500_without_logging_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_text = "private-text-that-must-not-be-logged"
    caplog.set_level(logging.ERROR)

    with TestClient(api_module.create_app(lambda: _FailingPredictor())) as client:
        response = client.post("/predict", json={"text": secret_text})

    assert response.status_code == 500
    assert secret_text not in caplog.text
    assert "Unexpected model inference failure (RuntimeError)" in caplog.text


def test_metrics_endpoint_exposes_preserved_prometheus_names() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        client.get("/")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert b"http_requests_total" in response.content
    assert b"http_request_duration_seconds" in response.content
    assert b"triage_predictions_total" in response.content


def test_swagger_docs_are_available() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/docs")

    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_explain_returns_preprocessing_and_five_canonical_scores() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/explain", params={"text": "Sample Abstract"})

    body = response.json()
    assert response.status_code == 200
    assert body["label"] == CANONICAL_LABELS[0]
    assert body["preprocessed_text"] == "sample abstract"
    assert list(body["scores"]) == CANONICAL_LABELS
    assert body["top_terms"] is None


def test_explain_rejects_blank_text() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/explain", params={"text": ""})

    assert response.status_code == 422


def test_explain_returns_503_without_predictor() -> None:
    with TestClient(api_module.create_app(_missing_predictor)) as client:
        response = client.get("/explain", params={"text": "abstract"})

    assert response.status_code == 503


def test_explain_failure_returns_500_without_logging_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_text = "private-text-that-must-not-be-logged"
    caplog.set_level(logging.ERROR)

    with TestClient(api_module.create_app(lambda: _FailingPredictor())) as client:
        response = client.get("/explain", params={"text": secret_text})

    assert response.status_code == 500
    assert secret_text not in caplog.text
    assert "Unexpected explain failure (RuntimeError)" in caplog.text


def test_sample_texts_covers_the_five_canonical_categories() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/demo/sample-texts")

    body = response.json()
    assert response.status_code == 200
    assert sorted(sample["label"] for sample in body) == sorted(CANONICAL_LABELS)
    assert all(sample["text"].strip() for sample in body)


def test_demo_experiment_results_explains_model_selection() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/demo/experiment-results")

    body = response.json()
    assert response.status_code == 200
    assert len(body["candidates"]) >= 3
    assert body["selection"]["primary_metric"] == "cv_macro_f1_mean"
    assert body["selection"]["tiebreak_metric"] == "cv_macro_f1_std"
    assert body["selection"]["latency_used"] is False
    winner = next(
        candidate
        for candidate in body["candidates"]
        if candidate["key"] == body["winner_key"]
    )
    assert winner["cv_macro_f1_mean"] == max(
        candidate["cv_macro_f1_mean"]
        for candidate in body["candidates"]
        if candidate["cv_macro_f1_mean"] is not None
    )


def test_demo_page_is_served_as_html() -> None:
    with TestClient(api_module.create_app(lambda: _FakePredictor())) as client:
        response = client.get("/demo")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Medical Text Classifier" in response.text
    assert "Escreva o seu próprio texto" in response.text
    assert "Quero testar com um exemplo pronto" in response.text
    assert "Comparação final dos modelos" in response.text
    assert '<div class="choice-divider" aria-hidden="true">OU</div>' in response.text
    assert '$(".chip:not(.oos)")?.click()' not in response.text
