"""Carrega o CSV bruto de laudos e separa em train/val/test."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


def load_raw_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    """Lê o CSV bruto e valida as colunas esperadas."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset bruto não encontrado em {path}. "
            "Rode `make dataset` (ou `python -m scripts.generate_synthetic_dataset`) "
            "para gerar um dataset sintético, ou coloque um dataset real "
            "(ver README, seção Dataset)."
        )
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {missing}")
    return df.dropna(subset=[text_col, label_col]).reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide em train/val/test com estratificação pela label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val[label_col],
        random_state=random_state,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def main() -> None:
    """Gera train.csv / val.csv / test.csv em data/processed/."""
    cfg = load_config()
    raw_path = Path(cfg["data"]["raw_path"]) / cfg["data"]["raw_file"]
    processed_path = Path(cfg["data"]["processed_path"])
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    df = load_raw_dataset(raw_path, text_col, label_col)
    train, val, test = split_dataset(
        df,
        label_col,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )

    processed_path.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed_path / "train.csv", index=False)
    val.to_csv(processed_path / "val.csv", index=False)
    test.to_csv(processed_path / "test.csv", index=False)
    logger.info(
        "Split salvo em %s — train=%d, val=%d, test=%d",
        processed_path,
        len(train),
        len(val),
        len(test),
    )


if __name__ == "__main__":
    from src.utils.logging_config import configure_logging

    configure_logging()
    main()
