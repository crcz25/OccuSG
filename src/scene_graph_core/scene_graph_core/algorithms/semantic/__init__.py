"""Reusable semantic representation algorithms."""

from .embedding import (
    accumulate_embedding,
    cosine_similarity,
    mean_embedding,
    normalize_embedding,
)

__all__ = [
    "accumulate_embedding",
    "cosine_similarity",
    "mean_embedding",
    "normalize_embedding",
]
