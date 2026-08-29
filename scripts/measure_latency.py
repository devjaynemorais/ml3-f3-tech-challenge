"""Compare sklearn and ONNX predictor latency with one neutral English text."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable

import numpy as np

from src.serving.model_loader import load_predictor
from src.utils.config_loader import load_config
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
        "p50_ms": float(np.percentile(values, 50)),
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
    _write_results(arguments.n_runs, sklearn_result, onnx_result)


def _write_results(runs: int, sklearn_result: dict, onnx_result: dict) -> None:
    """Persist benchmark results under the configured metrics path."""
    config = load_config()
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


if __name__ == "__main__":
    main()
