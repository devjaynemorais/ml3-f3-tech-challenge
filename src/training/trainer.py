"""Train the configured medical-text classifier from processed data."""

from __future__ import annotations

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.make_dataset import load_processed_split
from src.models.classifier import build_pipeline
from src.models.registry import save_pipeline
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _active_model_parameters(config: ExperimentConfig) -> dict:
    """Return the parameters for the selected model Strategy."""
    active_config = getattr(config.model, config.model.type)
    return active_config.model_dump()


def build_metadata(config: ExperimentConfig, sample_count: int) -> dict:
    """Build versioned metadata shared by training and serving."""
    return {
        "schema_version": config.artifacts.schema_version,
        "model_type": config.model.type,
        "feature_type": config.features.type,
        "model_parameters": _active_model_parameters(config),
        "preprocessing_strategies": config.preprocessing.steps,
        "classes": config.data.labels,
        "n_train_samples": sample_count,
    }


def train(training: pd.DataFrame, config: ExperimentConfig) -> tuple[Pipeline, dict]:
    """Fit one configured pipeline on the canonical training split."""
    pipeline = build_pipeline(config)
    pipeline.fit(training[config.data.text_column], training[config.data.label_column])
    return pipeline, build_metadata(config, len(training))


def train_from_processed(config: ExperimentConfig) -> tuple[Pipeline, dict]:
    """Load only processed train.csv and fit the selected model."""
    training = load_processed_split(config, config.data.train_output_file)
    return train(training, config)


def log_run_to_mlflow(
    config: ExperimentConfig, metadata: dict, pipeline: Pipeline
) -> str | None:
    """Log params and the fitted pipeline to MLflow; return the run id, or None."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # MLflow prints a run-URL summary with emoji; Windows consoles default
        # to cp1252, which raises UnicodeEncodeError on that summary.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_name = f"{metadata['model_type']}-{metadata['feature_type']}"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.project.name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(metadata["model_parameters"])
            mlflow.log_param("model_type", metadata["model_type"])
            mlflow.log_param("feature_type", metadata["feature_type"])
            mlflow.log_param("n_train_samples", metadata["n_train_samples"])
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            return run.info.run_id
    except Exception:
        logger.warning(
            "MLflow tracking unavailable at %s; skipping run log", tracking_uri
        )
        return None


def main() -> None:
    """Train from processed data and persist the pipeline and metadata."""
    config = load_config()
    set_seed(config.split.random_state)
    pipeline, metadata = train_from_processed(config)
    run_id = log_run_to_mlflow(config, metadata, pipeline)
    if run_id:
        metadata["mlflow_run_id"] = run_id
    model_path = save_pipeline(
        pipeline,
        config.artifacts.model_path,
        config.artifacts.pipeline_file,
        config.artifacts.metadata_file,
        metadata,
    )
    logger.info("Model trained and saved at %s", model_path)


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
