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


def _finite_vector(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Return a finite 1-D float array, or ``None`` when unusable."""
    if values is None:
        return None
    try:
        value = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        return None
    return value


def accumulate_embedding(
    embedding_sum: Optional[Sequence[float]],
    observation_count: int,
    new_embedding: Optional[Sequence[float]],
) -> tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
    """Fold one observation into an embedding sum accumulator.

    The accumulator stores the raw sum of unit-norm observations, so the mean is
    exact at every step and never drifts through repeated re-normalization.

    Invalid new observations leave the accumulator and count unchanged. A missing,
    invalid, or dimensionally inconsistent accumulator is re-seeded from the new
    observation.

    Returns:
        Tuple of (embedding sum, observation count, normalized mean). The mean is
        ``None`` only when the accumulator holds no usable direction.
    """
    new_normalized = normalize_embedding(new_embedding)
    if new_normalized is None:
        current = _finite_vector(embedding_sum)
        return (
            current,
            _nonnegative_count(observation_count),
            normalize_embedding(current),
        )

    count = _nonnegative_count(observation_count)
    current = _finite_vector(embedding_sum)
    if current is None or count == 0 or current.shape != new_normalized.shape:
        updated = new_normalized.astype(np.float64)
        return updated, 1, normalize_embedding(updated)

    updated = current + new_normalized.astype(np.float64)
    return updated, count + 1, normalize_embedding(updated)


def mean_embedding(
    embedding_sum: Optional[Sequence[float]],
    observation_count: int,
) -> Optional[np.ndarray]:
    """Return the unit-norm mean of an embedding sum accumulator."""
    current = _finite_vector(embedding_sum)
    if current is None or _nonnegative_count(observation_count) <= 0:
        return None
    return normalize_embedding(current / float(_nonnegative_count(observation_count)))
