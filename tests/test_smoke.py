"""Smoke tests — verificam imports e a configuração básica do projeto.

Não dependem de modelo treinado nem de dataset em disco, então passam em
qualquer clone limpo do repositório.
"""

from src.serving.api import app
from src.utils.config_loader import load_config


def test_config_tem_as_secoes_esperadas() -> None:
    """config.yaml deve expor data/split/features/model/artifacts."""
    cfg = load_config()
    for section in ("data", "split", "features", "model", "artifacts"):
        assert section in cfg


def test_labels_sao_as_tres_classes_de_urgencia() -> None:
    """As classes do domínio devem ser normal/atencao/urgente."""
    cfg = load_config()
    assert cfg["data"]["labels"] == ["normal", "atencao", "urgente"]


def test_fastapi_app_carrega_sem_erro() -> None:
    """A app FastAPI deve instanciar e expor as rotas principais."""
    routes = {route.path for route in app.routes}
    assert {"/", "/health", "/predict", "/metrics"}.issubset(routes)
