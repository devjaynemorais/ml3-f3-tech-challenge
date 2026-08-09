"""DAG de treino/retreino do classificador de triagem.

Fluxo: ingestão de dados -> treino -> avaliação -> exportação para ONNX
(otimização de latência). Cada task chama diretamente as funções de
`src/`, reaproveitando o mesmo código usado localmente via `make train` /
`make evaluate` / `make export-onnx` — a DAG só orquestra a ordem de
execução, sem duplicar lógica de ML.

Como rodar: `make airflow-up` sobe o Airflow (LocalExecutor + Postgres) via
docker-compose.airflow.yml; a UI fica em http://localhost:8080 (admin/admin).
A DAG é criada pausada — dispare manualmente pela UI ou CLI
(`airflow dags trigger triage_training`) para o primeiro teste.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "mle-f3",
    "retries": 1,
}


@dag(
    dag_id="triage_training",
    description="Ingestão -> treino -> avaliação -> exportação ONNX do classificador de triagem",
    schedule=None,  # dispare manualmente ou troque por um cron (ex.: "@weekly") para retreino periódico
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["triage", "training"],
)
def triage_training_dag():
    @task
    def ingest_data() -> str:
        """Garante que o CSV bruto de laudos exista (gera sintético se ausente)."""
        from src.utils.config_loader import load_config

        cfg = load_config()
        raw_path = Path(cfg["data"]["raw_path"]) / cfg["data"]["raw_file"]
        if not raw_path.exists():
            from scripts.generate_synthetic_dataset import generate_dataset

            df = generate_dataset(
                cfg["data"]["n_synthetic_samples"], cfg["split"]["random_state"]
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path, index=False, encoding="utf-8")
        return str(raw_path)

    @task
    def train_model(raw_path: str) -> str:
        """Treina o pipeline TF-IDF + classificador e salva o artefato."""
        from src.training.trainer import main as train_main

        train_main()
        return raw_path

    @task
    def evaluate_model(raw_path: str) -> None:
        """Avalia o modelo salvo no split de teste e grava metrics/eval_metrics.json."""
        from src.evaluation.evaluate import main as evaluate_main

        evaluate_main()

    @task
    def export_onnx() -> None:
        """Otimização de latência: converte o classificador treinado para ONNX."""
        from src.optimization.export_onnx import main as export_main

        export_main()

    raw_path = ingest_data()
    trained = train_model(raw_path)
    evaluate_model(trained) >> export_onnx()


triage_training_dag()
