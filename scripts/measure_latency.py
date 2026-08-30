"""Compare sklearn and ONNX predictor latency with one neutral English text."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable

import mlflow
import numpy as np

from src.models.registry import load_metadata
from src.serving.model_loader import load_predictor
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
SAMPLE_TEXT = (
    "The medical abstract describes a cardiovascular condition with persistent "
    "vascular inflammation and reduced ventricular function."
)


def _benchmark(
    operation: Callable[[], object], runs: int, warmup: int = 5
) -> dict[str, float]:
    """Measure mean and latency percentiles in milliseconds."""
    for _ in range(warmup):
        operation()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        timings.append((time.perf_counter() - start) * 1000)
    values = np.asarray(timings)
    return {
        "mean_ms": float(values.mean()),
        "p25_ms": float(np.percentile(values, 25)),
        "p50_ms": float(np.percentile(values, 50)),
        "p75_ms": float(np.percentile(values, 75)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def main() -> None:
    """Benchmark both TriagePredictor implementations and persist results."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=200)
    arguments = parser.parse_args()
    sklearn_predictor = load_predictor("sklearn")
    onnx_predictor = load_predictor("onnx")
    sklearn_result = _benchmark(
        lambda: sklearn_predictor.predict(SAMPLE_TEXT), arguments.n_runs
    )
    onnx_result = _benchmark(
        lambda: onnx_predictor.predict(SAMPLE_TEXT), arguments.n_runs
    )
    config = load_config()
    comparison = _write_results(config, arguments.n_runs, sklearn_result, onnx_result)
    metadata = load_metadata(
        config.artifacts.model_path, config.artifacts.metadata_file
    )
    run_id = metadata.get("mlflow_run_id")
    if run_id:
        log_latency_to_mlflow(run_id, comparison)


def _write_results(
    config: ExperimentConfig, runs: int, sklearn_result: dict, onnx_result: dict
) -> dict:
    """Persist benchmark results under the configured metrics path."""
    comparison = {
        "n_runs": runs,
        "sklearn": sklearn_result,
        "onnx": onnx_result,
        "speedup_x": sklearn_result["mean_ms"] / onnx_result["mean_ms"],
    }
    path = config.artifacts.metrics_path / "latency_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    logger.info("Latency comparison saved at %s", path)
    return comparison


def log_latency_to_mlflow(run_id: str, comparison: dict) -> None:
    """Attach the sklearn/ONNX latency comparison to the training's MLflow run."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=run_id):
            for backend in ("sklearn", "onnx"):
                for stat, value in comparison[backend].items():
                    mlflow.log_metric(f"latency_{backend}_{stat}", value)
            mlflow.log_metric("latency_speedup_x", comparison["speedup_x"])
    except Exception:
        logger.warning(
            "MLflow tracking unavailable at %s; skipping latency log", tracking_uri
        )


if __name__ == "__main__":
    main()
