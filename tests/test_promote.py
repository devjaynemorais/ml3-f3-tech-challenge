"""Tests for MLflow best-run selection and Model Registry promotion.

Use a temporary SQLite tracking backend with artifacts in tmp_path — no
MLflow server and no real training data, so these pass on any clean clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from src.models import promote
from src.utils import mlflow_tracking
from src.utils.config_loader import RegistryConfig


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configure an isolated SQLite tracking URI with artifacts in tmp_path."""
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    mlflow.create_experiment("test_exp", artifact_location=artifact_location)
    mlflow.set_experiment("test_exp")
    return tmp_path


def _log_run(metric_value: float) -> str:
    """Create one run with a validation metric and a logged sklearn model."""
    with mlflow.start_run() as run:
        mlflow.log_metric("validation_macro_f1", metric_value)
        model = LogisticRegression().fit([[0], [1]], [0, 1])
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


def test_find_best_model_run_rejects_runs_without_model_artifact(
    tracking: Path,
) -> None:
    with mlflow.start_run():
        mlflow.log_metric("validation_macro_f1", 0.99)  # no logged model
    with pytest.raises(ValueError, match="no run"):
        mlflow_tracking.find_best_model_run("test_exp", "validation_macro_f1", False)


def test_find_best_model_run_picks_highest_metric(tracking: Path) -> None:
    _log_run(0.55)
    best = _log_run(0.90)
    _log_run(0.60)

    run = mlflow_tracking.find_best_model_run("test_exp", "validation_macro_f1", False)

    assert run.info.run_id == best


def test_find_best_model_run_missing_experiment(tracking: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        mlflow_tracking.find_best_model_run(
            "does_not_exist", "validation_macro_f1", False
        )


def test_register_model_reuses_already_registered_name(tracking: Path) -> None:
    run_a = _log_run(0.70)
    run_b = _log_run(0.90)

    version_a = mlflow_tracking.register_model(run_a, "repeated_model")
    version_b = mlflow_tracking.register_model(run_b, "repeated_model")

    assert version_a != version_b


def test_register_and_promote_transitions_to_production(tracking: Path) -> None:
    run_id = _log_run(0.80)

    version = mlflow_tracking.register_model(run_id, "test_model", stage="Staging")
    mlflow_tracking.promote_model("test_model", version, stage="Production")

    client = MlflowClient(tracking_uri=mlflow_tracking.tracking_uri())
    production = client.get_latest_versions("test_model", stages=["Production"])[0]
    assert production.version == version
    assert production.run_id == run_id


def test_write_record_persists_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = tmp_path / "promoted_model.json"
    monkeypatch.setattr(promote, "RECORD_PATH", record_path)

    promote._write_record({"model_name": "x", "version": "1"})

    assert json.loads(record_path.read_text()) == {"model_name": "x", "version": "1"}


def test_promote_best_run_finds_registers_and_promotes(tracking: Path) -> None:
    run_id = _log_run(0.85)
    reg = RegistryConfig(
        model_name="promo_test_model",
        metric="validation_macro_f1",
        ascending=False,
        stage="Production",
    )

    record = promote.promote_best_run("test_exp", reg)

    assert record["run_id"] == run_id
    assert record["model_name"] == "promo_test_model"
    assert record["stage"] == "Production"
    assert record["value"] == pytest.approx(0.85)
