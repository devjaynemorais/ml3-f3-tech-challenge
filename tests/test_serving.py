import pytest
from fastapi.testclient import TestClient

import src.serving.api as api_module


class _FakePredictor:
    backend = "fake"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        return "urgente", {"normal": 0.05, "atencao": 0.15, "urgente": 0.8}


def test_root_endpoint() -> None:
    with TestClient(api_module.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["service"] == "Triage Classifier API"


def test_health_degraded_sem_modelo_treinado(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise FileNotFoundError("modelo não encontrado")

    monkeypatch.setattr(api_module, "load_predictor", _raise)
    with TestClient(api_module.app) as client:
        resp = client.get("/health")
        assert resp.json() == {"status": "degraded", "model_loaded": False}


def test_predict_endpoint_com_modelo_carregado(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_module, "load_predictor", lambda: _FakePredictor())
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={"text": "dor torácica intensa"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "urgente"
        assert body["backend"] == "fake"
        assert pytest.approx(sum(body["scores"].values()), rel=1e-6) == 1.0


def test_predict_endpoint_503_sem_modelo(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        raise FileNotFoundError("modelo não encontrado")

    monkeypatch.setattr(api_module, "load_predictor", _raise)
    with TestClient(api_module.app) as client:
        resp = client.post("/predict", json={"text": "dor torácica intensa"})
        assert resp.status_code == 503


def test_metrics_endpoint_expoe_formato_prometheus() -> None:
    with TestClient(api_module.app) as client:
        client.get("/")  # gera ao menos uma amostra antes de ler /metrics
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"http_requests_total" in resp.content
        assert b"http_request_duration_seconds" in resp.content
