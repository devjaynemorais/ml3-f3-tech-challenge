"""Compara a latência de inferência: pipeline scikit-learn puro vs. ONNX Runtime.

Gera `metrics/latency_comparison.json`, usado no README/vídeo STAR (Etapa 4)
como evidência da otimização de latência aplicada.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort

from src.models.registry import load_pipeline
from src.utils.config_loader import load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)

_SAMPLE_TEXT = (
    "Paciente do sexo feminino, 45 anos, comparece à emergência. "
    "Relato clínico: dor torácica intensa e súbita, irradiando para o braço esquerdo."
)


def _benchmark(
    fn: Callable[[], object], n_runs: int, n_warmup: int = 5
) -> dict[str, float]:
    for _ in range(n_warmup):
        fn()
    timings = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)  # ms
    arr = np.array(timings)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> None:
    """Roda o benchmark sklearn-vs-ONNX e grava o comparativo em metrics/."""
    configure_logging()
    cfg = load_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=200)
    args = parser.parse_args()

    artifacts_path = Path(cfg["artifacts"]["model_path"])
    onnx_path = Path(cfg["artifacts"]["onnx_path"])

    sklearn_pipeline = load_pipeline(artifacts_path, cfg["artifacts"]["pipeline_file"])
    sklearn_result = _benchmark(
        lambda: sklearn_pipeline.predict([_SAMPLE_TEXT]), args.n_runs
    )

    onnx_file = onnx_path / cfg["artifacts"]["onnx_file"]
    if not onnx_file.exists():
        raise FileNotFoundError(
            f"Modelo ONNX não encontrado em {onnx_file}. "
            "Rode `make export-onnx` primeiro."
        )
    vectorizer = joblib.load(onnx_path / "vectorizer.joblib")
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def _onnx_predict() -> None:
        vector = vectorizer.transform([_SAMPLE_TEXT]).toarray().astype(np.float32)
        session.run(None, {input_name: vector})

    onnx_result = _benchmark(_onnx_predict, args.n_runs)

    speedup = sklearn_result["mean_ms"] / onnx_result["mean_ms"]
    comparison = {
        "n_runs": args.n_runs,
        "sklearn": sklearn_result,
        "onnx": onnx_result,
        "speedup_x": speedup,
    }

    metrics_path = Path("metrics/latency_comparison.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Latência média — sklearn=%.3fms onnx=%.3fms (speedup=%.2fx). Salvo em %s",
        sklearn_result["mean_ms"],
        onnx_result["mean_ms"],
        speedup,
        metrics_path,
    )


if __name__ == "__main__":
    main()
