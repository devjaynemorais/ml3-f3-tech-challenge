"""Utilitário de reprodutibilidade — fixa seeds de random/numpy."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fixa a seed do módulo random e do numpy."""
    random.seed(seed)
    np.random.seed(seed)
