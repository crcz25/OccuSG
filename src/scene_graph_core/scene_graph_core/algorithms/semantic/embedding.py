"""Validation and online aggregation helpers for semantic embeddings.

This module intentionally has no ROS dependencies.  ROS-side managers can use
it to apply the same representation rules to messages from any perception
backend.
"""

from typing import Any, Optional, Sequence

import numpy as np

# Vectors shorter than this norm carry no reliable direction, so they cannot be
# normalized into a comparable representation.
_MIN_NORM = 1e-12


def _nonnegative_count(observation_count: Any) -> int:
    """Return a usable observation count, defaulting to 0 for invalid input."""
    try:
        return max(0, int(observation_count))
    except (TypeError, ValueError, OverflowError):
        return 0


def normalize_embedding(
    embedding: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    """Return a finite, non-zero, unit-norm 1-D embedding or ``None``."""
    if embedding is None:
        return None
    try:
        value = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        return None
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= _MIN_NORM:
        return None
    return (value / norm).astype(np.float32)


def cosine_similarity(
    left: Optional[Sequence[float]],
    right: Optional[Sequence[float]],
) -> Optional[float]:
    """Return cosine similarity, or ``None`` for invalid/incompatible vectors."""
    left_normalized = normalize_embedding(left)
    right_normalized = normalize_embedding(right)
    if left_normalized is None or right_normalized is None:
        return None
    if left_normalized.shape != right_normalized.shape:
        return None
    score = float(np.dot(left_normalized, right_normalized))
    return score if np.isfinite(score) else None


def running_mean_embedding(
    old_embedding: Optional[Sequence[float]],
    observation_count: int,
    new_embedding: Optional[Sequence[float]],
) -> tuple[Optional[np.ndarray], int]:
    """Add one valid observation to an embedding running mean.

    Invalid new observations leave the representation and count unchanged.
    A missing/invalid old representation is safely re-seeded from the new
    observation.  Dimension mismatches are treated as an incompatible old
    representation and also re-seed the running mean.

    Returns:
        Tuple of (updated unit-norm embedding, updated observation count).
    """
    new_normalized = normalize_embedding(new_embedding)
    if new_normalized is None:
        return normalize_embedding(old_embedding), _nonnegative_count(
            observation_count
        )

    count = _nonnegative_count(observation_count)
    old_normalized = normalize_embedding(old_embedding)
    if (
        old_normalized is None
        or count == 0
        or old_normalized.shape != new_normalized.shape
    ):
        return new_normalized, 1

    mean = (count * old_normalized.astype(np.float64) + new_normalized) / (count + 1)
    normalized_mean = normalize_embedding(mean)
    if normalized_mean is None:
        # Opposing observations can cancel out; keep the previous representation.
        return old_normalized, count
    return normalized_mean, count + 1
