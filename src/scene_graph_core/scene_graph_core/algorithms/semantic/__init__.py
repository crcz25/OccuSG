"""Reusable semantic representation algorithms."""

from .embedding import (
    cosine_similarity,
    normalize_embedding,
    update_running_mean_embedding,
)

__all__ = [
    "cosine_similarity",
    "normalize_embedding",
    "update_running_mean_embedding",
]
