"""Configuração central de logging para scripts e serviços do projeto."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configura o logging raiz com um formato consistente entre módulos."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
