"""Fine-tune a generic BERT classifier as a comparison baseline (P1.1).

Experimental: not wired into the servable Strategy/Factory or the API.
Reports macro F1 on the official test split so it can be compared against
the linear models in docs/model_card.md, and logs the run to MLflow under
the same experiment for side-by-side comparison.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.data.make_dataset import load_processed_split
from src.evaluation.evaluate import (
    build_label_sets,
    evaluate_predictions_with_label_sets,
)
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
OUTPUT_DIR = "models/bert_experiment"


class AbstractsDataset(Dataset):
    """Tokenized abstracts with integer labels, for the HF Trainer API."""

    def __init__(self, encodings: dict, labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def build_dataset(
    tokenizer: AutoTokenizer,
    frame: pd.DataFrame,
    config: ExperimentConfig,
    label_to_id: dict,
) -> AbstractsDataset:
    """Tokenize one processed split into a Trainer-ready Dataset."""
    texts = list(frame[config.data.text_column])
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    labels = frame[config.data.label_column].map(label_to_id).to_numpy()
    return AbstractsDataset(encodings, labels)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Macro precision/recall/F1 and accuracy for the HF Trainer callback."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_pred.label_ids, predictions, average="macro", zero_division=0
    )
    accuracy = float((predictions == eval_pred.label_ids).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def log_to_mlflow(
    train_metrics: dict, test_metrics: dict, corpus_metrics: dict
) -> None:
    """Log this experimental run to the same MLflow experiment for comparison."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Medical Text Classifier API")
        with mlflow.start_run(run_name=f"{MODEL_NAME}-experimental"):
            mlflow.log_param("model_type", MODEL_NAME)
            mlflow.log_param("feature_type", "transformer")
            mlflow.log_param("max_length", MAX_LENGTH)
            mlflow.log_param("num_train_epochs", 3)
            for key, value in train_metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"train_{key.replace('train_', '')}", value)
            for key, value in test_metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"test_{key.replace('test_', '')}", value)
            for key, value in corpus_metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"test_{key}", value)
            mlflow.log_dict(corpus_metrics, "corpus_aware_metrics.json")
    except Exception:
        logger.warning(
            "MLflow tracking unavailable at %s; skipping run log", tracking_uri
        )


def persist_experiment_outputs(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    truth: list[str],
    predicted: list[str],
    probabilities: np.ndarray,
    corpus_metrics: dict,
    test_metrics: dict,
) -> None:
    """Save the checkpoint and per-sample outputs required for later auditing."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    (output_path / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prediction_rows = []
    for index, (expected, prediction, scores) in enumerate(
        zip(truth, predicted, probabilities, strict=True)
    ):
        prediction_rows.append(
            {
                "row_index": index,
                "expected_label": expected,
                "predicted_label": prediction,
                "scores": dict(zip(CANONICAL_LABELS, scores.tolist(), strict=True)),
            }
        )
    (output_path / "test_predictions.json").write_text(
        json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / "corpus_aware_metrics.json").write_text(
        json.dumps(corpus_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    """Fine-tune bert-base-uncased on train, model-select on validation, report test."""
    config = load_config()
    label_to_id = {label: index for index, label in enumerate(CANONICAL_LABELS)}
    train = load_processed_split(config, config.data.train_output_file)
    validation = load_processed_split(config, config.data.validation_output_file)
    test = load_processed_split(config, config.data.test_output_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CANONICAL_LABELS)
    )

    train_dataset = build_dataset(tokenizer, train, config, label_to_id)
    validation_dataset = build_dataset(tokenizer, validation, config, label_to_id)
    test_dataset = build_dataset(tokenizer, test, config, label_to_id)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=device == "cuda",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    prediction_output = trainer.predict(test_dataset)
    test_metrics = prediction_output.metrics
    probabilities = torch.softmax(
        torch.as_tensor(prediction_output.predictions), dim=1
    ).numpy()
    predicted_ids = np.argmax(probabilities, axis=1)
    predicted = [CANONICAL_LABELS[index] for index in predicted_ids]
    truth = list(test[config.data.label_column])
    corpus_metrics = evaluate_predictions_with_label_sets(
        predicted,
        probabilities,
        CANONICAL_LABELS,
        test,
        config,
        build_label_sets(config),
    )
    reported_metrics = {**test_metrics, **corpus_metrics}
    logger.info("Test metrics: %s", reported_metrics)
    persist_experiment_outputs(
        trainer,
        tokenizer,
        truth,
        predicted,
        probabilities,
        corpus_metrics,
        test_metrics,
    )
    log_to_mlflow(train_result.metrics, prediction_output.metrics, corpus_metrics)


if __name__ == "__main__":
    configure_logging()
    main()
