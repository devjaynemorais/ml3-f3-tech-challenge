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


def _within_accuracy_tolerance(
    value: float, best_value: float, ascending: bool, tolerance: float
) -> bool:
    """True if ``value`` is at most ``tolerance`` worse than ``best_value``."""
    gap = (value - best_value) if ascending else (best_value - value)
    return gap <= tolerance


def _fastest_within_tolerance(
    candidates: list[Run],
    metric: str,
    ascending: bool,
    latency_metric: str,
    accuracy_tolerance: float,
) -> Run:
    """Among runs statistically tied on ``metric``, prefer the lowest latency.

    ``candidates`` is already sorted best-``metric``-first. Latency only
    breaks ties within ``accuracy_tolerance`` of the best value — it never
    outweighs a real accuracy difference. Runs missing ``latency_metric``
    (never benchmarked) are excluded from the latency comparison; if none of
    the tied candidates have it, this falls back to the pure-accuracy pick
    so promotion behaves exactly as before until latency is benchmarked.
    """
    best_value = candidates[0].data.metrics[metric]
    tied = [
        run
        for run in candidates
        if (value := run.data.metrics.get(metric)) is not None
        and _within_accuracy_tolerance(value, best_value, ascending, accuracy_tolerance)
    ]
    benchmarked = [run for run in tied if latency_metric in run.data.metrics]
    if not benchmarked:
        return candidates[0]
    return min(benchmarked, key=lambda run: run.data.metrics[latency_metric])


def _passes_min_metric(value: float, ascending: bool, min_metric: float) -> bool:
    """True if ``value`` clears the quality floor in the metric's own direction."""
    return value <= min_metric if ascending else value >= min_metric


def find_best_model_run(
    experiment_name: str,
    metric: str,
    ascending: bool,
    tiebreak_metric: str | None = None,
    tiebreak_ascending: bool = True,
    latency_metric: str | None = None,
    accuracy_tolerance: float = 0.0,
    min_metric: float | None = None,
) -> Run:
    """Return the best run by ``metric`` (ties broken by ``tiebreak_metric``).

    ``min_metric`` is a hard quality gate: runs that do not clear it are
    never candidates, even if they are the only ones with a model artifact —
    this fails loudly (raises) rather than silently promoting a run known to
    be below the acceptable bar.

    When ``latency_metric`` is set, runs within ``accuracy_tolerance`` of the
    best ``metric`` value (among those that already cleared ``min_metric``)
    are treated as statistically tied (see ``docs/model_card.md#avaliacao``
    for why ~0.015 is the documented noise floor on this validation set) and
    the fastest one among them is promoted instead — trading a real accuracy
    edge for speed is never allowed, only noise-level ties are.
    """
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
    candidates = [
        run
        for run in runs
        if client.list_artifacts(run.info.run_id, MODEL_ARTIFACT_PATH)
    ]
    if min_metric is not None:
        candidates = [
            run
            for run in candidates
            if _passes_min_metric(run.data.metrics[metric], ascending, min_metric)
        ]
    if not candidates:
        raise ValueError(
            f"no run with a model artifact and metric {metric!r}"
            + (f" >= {min_metric}" if min_metric is not None else "")
        )
    if not latency_metric:
        return candidates[0]
    return _fastest_within_tolerance(
        candidates, metric, ascending, latency_metric, accuracy_tolerance
    )


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
