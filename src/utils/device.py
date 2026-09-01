"""Resolve TRAINING_DEVICE for GPU-capable training paths (env var: TRAINING_DEVICE)."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _nvidia_gpu_present() -> bool:
    """Detect an NVIDIA GPU via nvidia-smi, without depending on torch/xgboost."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def gpu_explicitly_requested() -> bool:
    """True only when TRAINING_DEVICE explicitly asks for GPU, never via auto-detect."""
    return os.environ.get("TRAINING_DEVICE", "").strip().lower() in ("cuda", "gpu")


def resolve_training_device() -> str:
    """Resolve TRAINING_DEVICE (auto|cpu|cuda/gpu) into "cpu" or "cuda"."""
    requested = os.environ.get("TRAINING_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if gpu_explicitly_requested():
        if not _nvidia_gpu_present():
            raise RuntimeError(
                "TRAINING_DEVICE=cuda mas nenhuma GPU NVIDIA foi detectada "
                "(nvidia-smi indisponivel ou sem GPU). Instale os drivers "
                "NVIDIA ou use TRAINING_DEVICE=cpu/auto."
            )
        return "cuda"
    if requested not in ("auto", ""):
        logger.warning("TRAINING_DEVICE=%r invalido; usando auto-deteccao.", requested)
    return "cuda" if _nvidia_gpu_present() else "cpu"
