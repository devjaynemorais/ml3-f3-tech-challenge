"""Airflow graph contract without requiring Airflow in the test environment."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class FakeTaskNode:
    name: str
    edges: set[tuple[str, str]]

    def __rshift__(self, other: FakeTaskNode) -> FakeTaskNode:
        self.edges.add((self.name, other.name))
        return other


@dataclass(frozen=True)
class FakeDag:
    metadata: dict[str, Any]
    edges: set[tuple[str, str]]


def _fake_airflow_modules() -> tuple[ModuleType, ModuleType]:
    airflow = ModuleType("airflow")
    decorators = ModuleType("airflow.decorators")
    edges: set[tuple[str, str]] = set()

    def task(function):
        def invoke(*upstream):
            current = FakeTaskNode(function.__name__, edges)
            for dependency in upstream:
                if isinstance(dependency, FakeTaskNode):
                    edges.add((dependency.name, current.name))
            return current

        return invoke

    def dag(**metadata):
        def decorate(function):
            def build():
                function()
                return FakeDag(metadata, edges)

            return build

        return decorate

    decorators.task = task
    decorators.dag = dag
    airflow.decorators = decorators
    return airflow, decorators


def test_training_dag_is_manual_and_preserves_task_order(monkeypatch) -> None:
    airflow, decorators = _fake_airflow_modules()
    monkeypatch.setitem(sys.modules, "airflow", airflow)
    monkeypatch.setitem(sys.modules, "airflow.decorators", decorators)
    path = Path("airflow/dags/triage_training_dag.py")
    spec = importlib.util.spec_from_file_location("tested_training_dag", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dag = module.triage_training
    assert dag.metadata["dag_id"] == "triage_training"
    assert dag.metadata["schedule"] is None
    assert dag.edges == {
        ("ingest_data", "train_model"),
        ("train_model", "evaluate_model"),
        ("evaluate_model", "export_onnx"),
    }
