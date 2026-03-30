"""Shared sentence-transformer helpers with simple model caching."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


@lru_cache(maxsize=4)
def get_embedding_model(model_name: str):
    """Load and cache a sentence-transformer model by name."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on environment setup
        raise ImportError(
            "sentence-transformers is required for semantic topic mapping and question clustering. "
            "Install the updated project dependencies before running the analyzer."
        ) from exc
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def encode_texts(texts: Iterable[str], model_name: str) -> np.ndarray:
    """Encode texts into normalized sentence embeddings."""
    model = get_embedding_model(model_name)
    values = list(texts)
    if not values:
        return np.empty((0, 0), dtype=float)
    return np.asarray(
        model.encode(values, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype=float,
    )
