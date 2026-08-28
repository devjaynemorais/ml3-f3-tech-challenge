"""Airflow DAG for the reproducible medical classifier training pipeline.

Run ``make airflow-up`` and trigger ``triage_training`` manually. The DAG only
orchestrates production entry points from ``src`` and contains no ML logic.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

DEFAULT_ARGS = {"owner": "mle-f3", "retries": 1}


@dag(
    dag_id="triage_training",
    description="Prepare data, train, evaluate and export the medical classifier",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["medical-text", "training"],
)
def triage_training_dag():
    """Compose the manually triggered training workflow."""

    @task
    def ingest_data() -> str:
        """Validate the corpus and persist the three canonical splits."""
        from src.data.make_dataset import main as dataset_main
        from src.utils.config_loader import load_config

        dataset_main()
        config = load_config()
        path = config.data.processed_path / config.data.train_output_file
        return str(path)

    @task
    def train_model(train_path: str) -> str:
        """Train and persist the configured classifier pipeline."""
        from src.training.trainer import main as train_main

        train_main()
        return train_path

    @task
    def evaluate_model(train_path: str) -> None:
        """Evaluate the persisted pipeline on validation and official test."""
        from src.evaluation.evaluate import main as evaluate_main

        evaluate_main()

    @task
    def export_onnx() -> None:
        """Export the persisted classifier while retaining its feature prefix."""
        from src.optimization.export_onnx import main as export_main

        export_main()

    prepared = ingest_data()
    trained = train_model(prepared)
    evaluate_model(trained) >> export_onnx()


triage_training = triage_training_dag()
