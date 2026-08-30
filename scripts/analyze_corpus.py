"""Report the multi-label structure of the Medical Abstracts corpus.

Reproduces the diagnosis in docs/plano_melhoria_f1.md: the corpus is
multi-label flattened to one row per (text, label) pair, which caps the F1-
macro any single-label classifier can legitimately reach on the official
test split.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

from src.data.make_dataset import load_medical_corpus
from src.evaluation.evaluate import build_label_sets
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def label_count_distribution(label_sets: dict[str, set[str]]) -> dict[int, int]:
    """Count how many abstracts carry 1, 2, 3... valid labels."""
    counts: dict[int, int] = {}
    for labels in label_sets.values():
        counts[len(labels)] = counts.get(len(labels), 0) + 1
    return counts


def oracle_f1_ceiling(
    config: ExperimentConfig, label_sets: dict[str, set[str]], seed: int = 0
) -> float:
    """F1-macro of an oracle picking any one valid label per test abstract."""
    corpus = load_medical_corpus(config.data)
    texts = corpus.test[config.data.text_column]
    rng = np.random.default_rng(seed)
    picks = [rng.choice(sorted(label_sets[text])) for text in texts]
    return float(
        f1_score(corpus.test[config.data.label_column], picks, average="macro")
    )


def main() -> None:
    """Print the label-count distribution and the theoretical F1 ceiling."""
    config = load_config()
    label_sets = build_label_sets(config)
    logger.info("Abstracts by number of valid labels:")
    for count, n in sorted(label_count_distribution(label_sets).items()):
        logger.info("  %d label(s): %d", count, n)
    ceiling = oracle_f1_ceiling(config, label_sets)
    logger.info("Theoretical F1-macro ceiling (oracle, seed=0): %.4f", ceiling)


if __name__ == "__main__":
    configure_logging()
    main()
