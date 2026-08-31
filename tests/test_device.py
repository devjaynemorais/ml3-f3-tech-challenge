from __future__ import annotations

import pytest

from src.utils import device


@pytest.mark.parametrize("value", ["cuda", "gpu", "CUDA", " Gpu "])
def test_gpu_explicitly_requested_accepts_cuda_and_gpu(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", value)
    assert device.gpu_explicitly_requested() is True


@pytest.mark.parametrize("value", [None, "", "auto", "cpu"])
def test_gpu_explicitly_requested_false_otherwise(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("TRAINING_DEVICE", raising=False)
    else:
        monkeypatch.setenv("TRAINING_DEVICE", value)
    assert device.gpu_explicitly_requested() is False


def test_resolve_training_device_cpu_forces_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "cpu")
    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: True)

    assert device.resolve_training_device() == "cpu"


def test_resolve_training_device_cuda_with_gpu_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "cuda")
    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: True)

    assert device.resolve_training_device() == "cuda"


def test_resolve_training_device_cuda_without_gpu_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "cuda")
    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: False)

    with pytest.raises(RuntimeError, match="nenhuma GPU"):
        device.resolve_training_device()


def test_resolve_training_device_auto_detects_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRAINING_DEVICE", raising=False)
    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: True)
    assert device.resolve_training_device() == "cuda"

    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: False)
    assert device.resolve_training_device() == "cpu"


def test_resolve_training_device_warns_on_unknown_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TRAINING_DEVICE", "tpu")
    monkeypatch.setattr(device, "_nvidia_gpu_present", lambda: False)

    with caplog.at_level("WARNING"):
        result = device.resolve_training_device()

    assert result == "cpu"
    assert "invalido" in caplog.text
