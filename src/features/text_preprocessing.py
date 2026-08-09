"""Normalização leve de texto aplicada antes da vetorização TF-IDF."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Minúsculas, remove acentos e normaliza espaços em branco."""
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WHITESPACE_RE.sub(" ", text)
