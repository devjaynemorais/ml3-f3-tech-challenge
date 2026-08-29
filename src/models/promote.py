"""Register the best tracked MLflow run and promote it to Production.

Runs after `make evaluate`. Finds the best run by the configured validation
metric, registers its model artifact in the MLflow Model Registry (Staging)
and promotes it to the configured final stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.config_loader import ExperimentConfig, RegistryConfig, load_config
from src.utils.mlflow_tracking import find_best_model_run, promote_model, register_model

logger = logging.getLogger(__name__)

RECORD_PATH = Path("models/promoted_model.json")


def _write_record(record: dict) -> None:
    """Persist the promotion record for traceability."""
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def promote_best_run(experiment_name: str, reg: RegistryConfig) -> dict:
    """Find the best run, register it and promote it; return the record."""
    best = find_best_model_run(experiment_name, reg.metric, reg.ascending)
    version = register_model(best.info.run_id, reg.model_name, stage="Staging")
    promote_model(reg.model_name, version, stage=reg.stage)
    return {
        "model_name": reg.model_name,
        "version": version,
        "stage": reg.stage,
        "run_id": best.info.run_id,
        "metric": reg.metric,
        "value": best.data.metrics.get(reg.metric),
    }


def main() -> None:
    """Promote the best tracked run for the configured experiment."""
    config: ExperimentConfig = load_config()
    record = promote_best_run(config.project.name, config.registry)
    _write_record(record)
    logger.info(
        "Promoted %s v%s to %s",
        record["model_name"],
        record["version"],
        record["stage"],
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
