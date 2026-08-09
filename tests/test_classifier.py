import pandas as pd
import pytest

from src.models.classifier import build_pipeline

_TRAIN_TEXTS = [
    "dor de cabeça leve, sem febre",
    "consulta de rotina, paciente assintomático",
    "febre moderada e tosse persistente há 2 dias",
    "pressão arterial elevada, paciente hipertenso",
    "dor torácica intensa irradiando para o braço esquerdo",
    "dificuldade respiratória grave, saturação baixa",
]
_TRAIN_LABELS = ["normal", "normal", "atencao", "atencao", "urgente", "urgente"]


def _tiny_cfg(model_type: str = "logistic_regression") -> dict:
    return {
        "features": {"max_features": 50, "ngram_range": [1, 1], "min_df": 1},
        "model": {
            "type": model_type,
            "logistic_regression": {"C": 1.0, "max_iter": 200, "random_state": 42},
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42,
            },
        },
    }


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest"])
def test_build_pipeline_treina_e_prediz(model_type: str) -> None:
    cfg = _tiny_cfg(model_type)
    pipeline = build_pipeline(cfg)

    df = pd.DataFrame({"text": _TRAIN_TEXTS, "label": _TRAIN_LABELS})
    pipeline.fit(df["text"], df["label"])

    proba = pipeline.predict_proba(["dor no peito muito forte"])
    assert proba.shape == (1, 3)
    assert pytest.approx(proba.sum(), rel=1e-6) == 1.0

    pred = pipeline.predict(["consulta de rotina"])
    assert pred[0] in {"normal", "atencao", "urgente"}


def test_build_pipeline_tipo_invalido_lanca_erro() -> None:
    cfg = _tiny_cfg()
    cfg["model"]["type"] = "modelo_inexistente"
    with pytest.raises(ValueError, match="desconhecido"):
        build_pipeline(cfg)
