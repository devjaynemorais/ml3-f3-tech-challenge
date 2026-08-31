"""Train and evaluate every official experiment configuration in sequence.

`make train` only trains whatever ``model.type``/``features.type`` is
currently selected in ``config/config.yaml``. This script automates the
comparison documented in ``docs/metodologia_experimentos.md``: it overwrites
``features.type``/``model.type`` in ``config.yaml``, runs train + evaluate for
each sklearn combination (the same commands as ``make train``/``make
evaluate``), restores ``config.yaml``, and by default retrains the original
configuration once more so ``models/artifacts/`` — what
``src/serving/model_loader.py`` actually serves — ends up back on the
production model instead of whichever experiment happened to run last.

BERT does not go through ``config.yaml``: it is a separate script
(``scripts/finetune_bert.py``, own dependency group ``install-experiments``,
~12 min on GPU, much longer on CPU) that writes to ``models/bert_experiment/``
without touching ``models/artifacts/`` — so it is opt-in via ``--include-bert``
and never needs a restore step.

Uso:
    poetry run python -m scripts.compare_experiments
    poetry run python -m scripts.compare_experiments --include-bert
    poetry run python -m scripts.compare_experiments --experiments \
        logistic_regression_tfidf gradient_boosting_tfidf
    poetry run python -m scripts.compare_experiments --no-promote
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

from src.utils.config_loader import CONFIG_PATH, load_config

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_PATH = ROOT / "metrics" / "experiment_comparison.json"
BERT_OUTPUT_DIR = ROOT / "models" / "bert_experiment"

# Both keys sit on the line right after their section header in config.yaml,
# so an anchored "section:\n  type: " match is enough to tell them apart.
FEATURES_TYPE_RE = re.compile(r"(features:\n  type: )(\S+)")
MODEL_TYPE_RE = re.compile(r"(\nmodel:\n  type: )(\S+)")


class Experiment(NamedTuple):
    key: str
    label: str
    model_type: str
    features_type: str


# The 3 sklearn combinations from docs/metodologia_experimentos.md (BERT is
# the 4th official experiment but runs through --include-bert instead, since
# it bypasses config.yaml entirely).
EXPERIMENTS = [
    Experiment(
        "logistic_regression_tfidf",
        "Logistic Regression + TF-IDF",
        "logistic_regression",
        "tfidf",
    ),
    Experiment(
        "gradient_boosting_tfidf",
        "Gradient Boosting + TF-IDF",
        "gradient_boosting",
        "tfidf",
    ),
    Experiment(
        "logistic_regression_embeddings",
        "Logistic Regression + embeddings biomedicos",
        "logistic_regression",
        "embeddings",
    ),
]
EXPERIMENTS_BY_KEY = {experiment.key: experiment for experiment in EXPERIMENTS}


def _current_selection(text: str) -> tuple[str, str]:
    """Read the (model_type, features_type) currently active in config.yaml."""
    features_match = FEATURES_TYPE_RE.search(text)
    model_match = MODEL_TYPE_RE.search(text)
    if not features_match or not model_match:
        raise RuntimeError("Nao encontrei features.type/model.type em config.yaml")
    return model_match.group(2), features_match.group(2)


def _set_selection(text: str, model_type: str, features_type: str) -> str:
    """Return config.yaml text with features.type/model.type overwritten."""
    text, n_features = FEATURES_TYPE_RE.subn(rf"\g<1>{features_type}", text)
    if n_features != 1:
        raise RuntimeError("Substituicao de features.type falhou (0 ou 2+ matches)")
    text, n_model = MODEL_TYPE_RE.subn(rf"\g<1>{model_type}", text)
    if n_model != 1:
        raise RuntimeError("Substituicao de model.type falhou (0 ou 2+ matches)")
    return text


def _run(module: str) -> None:
    """Run one stage with the active venv interpreter, locally or in Docker."""
    print(f"  $ {sys.executable} -m {module}")
    # Windows consoles default to cp1252, which cannot encode the emoji
    # MLflow prints in its run-URL summary (e.g. "View run ... at: ...") —
    # without this the subprocess exits with UnicodeEncodeError even after
    # training/evaluating successfully.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run(
        [sys.executable, "-m", module],
        cwd=ROOT,
        check=True,
        env=env,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _train_and_evaluate(metadata_path: Path, metrics_path: Path) -> dict:
    """Run train + evaluate and collect the metrics they persisted."""
    _run("src.training.trainer")
    _run("src.evaluation.evaluate")
    metadata = _read_json(metadata_path)
    metrics = _read_json(metrics_path)
    return {
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "cv_macro_f1_mean": metadata.get("cv_macro_f1_mean"),
        "cv_macro_f1_std": metadata.get("cv_macro_f1_std"),
        "model_parameters": metadata.get("model_parameters"),
        "test": metrics.get("test", {}),
    }


def _bert_result() -> dict:
    """Normalize the HF Trainer + corpus-aware BERT outputs into the same shape."""
    raw_test_metrics = _read_json(BERT_OUTPUT_DIR / "test_metrics.json")
    corpus_metrics = _read_json(BERT_OUTPUT_DIR / "corpus_aware_metrics.json")
    return {
        "label": "BERT generico fine-tunado (bert-base-uncased)",
        "model_type": "bert-base-uncased",
        "features_type": "transformer",
        "test": {
            "accuracy": raw_test_metrics.get("test_accuracy"),
            "macro_avg": {
                "precision": raw_test_metrics.get("test_macro_precision"),
                "recall": raw_test_metrics.get("test_macro_recall"),
                "f1": raw_test_metrics.get("test_macro_f1"),
            },
            **corpus_metrics,
            "raw_hf_test_metrics": raw_test_metrics,
        },
    }


def _print_summary(results: dict[str, dict]) -> None:
    nan = float("nan")
    print(
        f"\n{'experimento':<46} {'cv_f1_macro':>12} {'test_f1_macro':>14} "
        f"{'test_acc':>9} {'ml_accuracy':>12}"
    )
    for result in results.values():
        cv = result.get("cv_macro_f1_mean")
        test = result.get("test", {})
        macro = test.get("macro_avg") or {}
        test_f1 = macro.get("f1")
        accuracy = test.get("accuracy")
        ml_accuracy = test.get("jaccard_samples")
        print(
            f"{result['label']:<46} "
            f"{cv if cv is not None else nan:>12.4f} "
            f"{test_f1 if test_f1 is not None else nan:>14.4f} "
            f"{accuracy if accuracy is not None else nan:>9.4f} "
            f"{ml_accuracy if ml_accuracy is not None else nan:>12.4f}"
        )
    print(
        "\ntest_f1_macro e ml_accuracy sao as metricas justas: cada abstract "
        "carrega seu conjunto completo de rotulos validos (agregado em "
        "aggregate_multilabel_split, src/data/make_dataset.py), entao "
        "test_f1_macro ja e F1-macro multirrotulo (macro_avg.f1) e ml_accuracy "
        "e a accuracy Jaccard por amostra (jaccard_samples; Godbole & Sarawagi, "
        "2004). test_acc (subset_accuracy) exige acerto simultaneo de todos os "
        "rotulos e costuma subestimar desempenho real — priorize test_f1_macro "
        "e ml_accuracy. Ver docs/model_card.md#avaliacao."
    )


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        # Windows consoles default to cp1252, which cannot encode the em
        # dashes in this module's own docstring/prints (same issue MLflow's
        # run-URL summary triggers elsewhere in this codebase).
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS_BY_KEY),
        choices=list(EXPERIMENTS_BY_KEY),
        help="Quais experimentos sklearn treinar/avaliar, em ordem (default: todos).",
    )
    parser.add_argument(
        "--include-bert",
        action="store_true",
        help=(
            "Tambem roda scripts.finetune_bert (BERT) ao final. Script "
            "separado (nao usa config.yaml), requer 'make install-experiments' "
            "e ~12 min em GPU (bem mais em CPU)."
        ),
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Nao roda 'make promote' (src.models.promote) ao final.",
    )
    args = parser.parse_args()

    config = load_config()
    metadata_path = config.artifacts.model_path / config.artifacts.metadata_file
    metrics_path = config.artifacts.metrics_path / config.artifacts.metrics_file

    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    original_selection = _current_selection(original_text)
    results: dict[str, dict] = {}
    last_selection: tuple[str, str] | None = None

    try:
        for key in args.experiments:
            experiment = EXPERIMENTS_BY_KEY[key]
            print(f"\n=== {experiment.label} ===")
            CONFIG_PATH.write_text(
                _set_selection(
                    original_text, experiment.model_type, experiment.features_type
                ),
                encoding="utf-8",
            )
            last_selection = (experiment.model_type, experiment.features_type)
            outcome = _train_and_evaluate(metadata_path, metrics_path)
            results[key] = {
                "label": experiment.label,
                "model_type": experiment.model_type,
                "features_type": experiment.features_type,
                **outcome,
            }
    finally:
        model_type, features_type = original_selection
        print(
            f"\nRestaurando config.yaml (model.type={model_type}, "
            f"features.type={features_type})"
        )
        CONFIG_PATH.write_text(original_text, encoding="utf-8")

    if last_selection is not None and last_selection != original_selection:
        print(
            "\nRetreinando a configuracao original para restaurar "
            "models/artifacts/ (o que src/serving/model_loader.py de fato "
            "serve) ao estado de producao..."
        )
        _train_and_evaluate(metadata_path, metrics_path)

    if args.include_bert:
        print("\n=== BERT (bert-base-uncased, fine-tuned) ===")
        _run("scripts.finetune_bert")
        results["bert"] = _bert_result()

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nComparacao salva em {COMPARISON_PATH}")
    _print_summary(results)

    if not args.no_promote:
        print("\n=== promote ===")
        _run("src.models.promote")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nFalhou: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
