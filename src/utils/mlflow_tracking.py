"""MLflow helpers: best-run selection and Model Registry promotion."""

from __future__ import annotations

import os

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

MODEL_ARTIFACT_PATH = "model"


def tracking_uri() -> str:
    """Resolve the MLflow tracking server endpoint."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _order_by(
    metric: str, ascending: bool, tiebreak_metric: str | None, tiebreak_ascending: bool
) -> list[str]:
    """Build the primary metric ordering plus an optional tiebreak clause."""
    order = "ASC" if ascending else "DESC"
    clauses = [f"metrics.{metric} {order}"]
    if tiebreak_metric:
        tiebreak_order = "ASC" if tiebreak_ascending else "DESC"
        clauses.append(f"metrics.{tiebreak_metric} {tiebreak_order}")
    return clauses


def find_best_model_run(
    experiment_name: str,
    metric: str,
    ascending: bool,
    tiebreak_metric: str | None = None,
    tiebreak_ascending: bool = True,
) -> Run:
    """Return the best run by ``metric`` (ties broken by ``tiebreak_metric``)."""
    client = MlflowClient(tracking_uri=tracking_uri())
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"experiment {experiment_name!r} not found")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"metrics.{metric} > -1e30",
        order_by=_order_by(metric, ascending, tiebreak_metric, tiebreak_ascending),
        max_results=10,
    )
    for run in runs:
        if client.list_artifacts(run.info.run_id, MODEL_ARTIFACT_PATH):
            return run
    raise ValueError(f"no run with a model artifact and metric {metric!r}")


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> str:
    """Register a run's model artifact in the MLflow Model Registry."""
    client = MlflowClient(tracking_uri=tracking_uri())
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # already exists
    version = client.create_model_version(model_name, model_uri, run_id)
    client.transition_model_version_stage(
        name=model_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=True,
    )
    return version.version


def promote_model(model_name: str, version: str, stage: str = "Production") -> None:
    """Transition an existing registered model version to a new stage."""
    client = MlflowClient(tracking_uri=tracking_uri())
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=True
    )
