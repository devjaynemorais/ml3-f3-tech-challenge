"""Fine-tune a generic BERT classifier as a comparison baseline (P1.1).

Experimental: not wired into the servable Strategy/Factory or the API.
Reports macro F1 on the official test split so it can be compared against
the linear models in docs/model_card.md, and logs the run to MLflow under
the same experiment for side-by-side comparison.
"""

from __future__ import annotations

# Must load before pandas: on Windows, torch's native DLL fails to
# initialize if pandas claims the process's DLL search path first. The
# unconditional `from torch.utils.data import Dataset` below still gives a
# clear ImportError if torch truly isn't installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass

import json
import logging
import os
from pathlib import Path

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
from src.evaluation.evaluate import score_multilabel
from src.utils.config_loader import CANONICAL_LABELS, ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512
OUTPUT_DIR = "models/bert_experiment"


def resolve_device() -> str:
    """Resolve the training device from TRAINING_DEVICE (auto|cpu|cuda/gpu)."""
    requested = os.environ.get("TRAINING_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TRAINING_DEVICE=cuda mas nenhuma GPU CUDA foi detectada "
                "(torch.cuda.is_available() é False). Instale uma build do "
                "torch com suporte a CUDA ou use TRAINING_DEVICE=cpu/auto."
            )
        return "cuda"
    if requested not in ("auto", ""):
        logger.warning("TRAINING_DEVICE=%r invalido; usando auto-deteccao.", requested)
    return "cuda" if torch.cuda.is_available() else "cpu"


class AbstractsDataset(Dataset):
    """Tokenized abstracts with multi-hot label vectors, for the HF Trainer API."""

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
) -> AbstractsDataset:
    """Tokenize one processed split into a Trainer-ready multilabel Dataset."""
    texts = list(frame[config.data.text_column])
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    labels = frame[config.data.labels].to_numpy(dtype=np.float32)
    return AbstractsDataset(encodings, labels)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Macro precision/recall/F1 and subset accuracy for the HF Trainer callback.

    ``eval_pred.predictions`` are raw logits (``problem_type=
    "multi_label_classification"`` uses BCEWithLogitsLoss, not softmax), so
    predictions are thresholded on sigmoid probabilities, one independent
    decision per label — matching ``src/evaluation/evaluate.py``'s
    multilabel scoring, not single-label argmax.
    """
    probabilities = 1 / (1 + np.exp(-eval_pred.predictions))
    predictions = (probabilities >= 0.5).astype(int)
    labels = eval_pred.label_ids.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    accuracy = float((predictions == labels).all(axis=1).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def log_to_mlflow(
    train_metrics: dict, test_metrics: dict, multilabel_metrics: dict
) -> None:
    """Log this experimental run to the same MLflow experiment for comparison.

    Imports mlflow lazily: importing it before torch breaks torch's native
    DLL loading on Windows when both share a process (see export_onnx.py's
    equivalent note for the analogous skl2onnx/mlflow conflict).
    """
    import mlflow

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
            for key, value in multilabel_metrics.items():
                if isinstance(value, int | float):
                    mlflow.log_metric(f"test_{key}", value)
            mlflow.log_dict(multilabel_metrics, "corpus_aware_metrics.json")
    except Exception:
        logger.warning(
            "MLflow tracking unavailable at %s; skipping run log", tracking_uri
        )


def persist_experiment_outputs(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    truth_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    probabilities: np.ndarray,
    multilabel_metrics: dict,
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
    for index, (expected, predicted, scores) in enumerate(
        zip(truth_matrix, predicted_matrix, probabilities, strict=True)
    ):
        prediction_rows.append(
            {
                "row_index": index,
                "expected_labels": [
                    label
                    for label, hit in zip(CANONICAL_LABELS, expected, strict=True)
                    if hit
                ],
                "predicted_labels": [
                    label
                    for label, hit in zip(CANONICAL_LABELS, predicted, strict=True)
                    if hit
                ],
                "scores": dict(zip(CANONICAL_LABELS, scores.tolist(), strict=True)),
            }
        )
    (output_path / "test_predictions.json").write_text(
        json.dumps(prediction_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Filename kept for scripts/compare_experiments.py's _bert_result(), which
    # still reads "corpus_aware_metrics.json" — content is now the same
    # multilabel schema score_multilabel() produces everywhere else.
    (output_path / "corpus_aware_metrics.json").write_text(
        json.dumps(multilabel_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    """Fine-tune bert-base-uncased on train, model-select on validation, report test."""
    config = load_config()
    train = load_processed_split(config, config.data.train_output_file)
    validation = load_processed_split(config, config.data.validation_output_file)
    test = load_processed_split(config, config.data.test_output_file)

    device = resolve_device()
    logger.info("Using device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CANONICAL_LABELS),
        problem_type="multi_label_classification",
    )

    train_dataset = build_dataset(tokenizer, train, config)
    validation_dataset = build_dataset(tokenizer, validation, config)
    test_dataset = build_dataset(tokenizer, test, config)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=device == "cuda",
        use_cpu=device == "cpu",
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
    probabilities = torch.sigmoid(
        torch.as_tensor(prediction_output.predictions)
    ).numpy()
    predicted_matrix = (probabilities >= 0.5).astype(int)
    truth_matrix = test[config.data.labels].to_numpy(dtype=int)
    multilabel_metrics = score_multilabel(
        truth_matrix, predicted_matrix, CANONICAL_LABELS
    )
    reported_metrics = {**test_metrics, **multilabel_metrics}
    logger.info("Test metrics: %s", reported_metrics)
    persist_experiment_outputs(
        trainer,
        tokenizer,
        truth_matrix,
        predicted_matrix,
        probabilities,
        multilabel_metrics,
        test_metrics,
    )
    log_to_mlflow(train_result.metrics, prediction_output.metrics, multilabel_metrics)


if __name__ == "__main__":
    configure_logging()
    main()
