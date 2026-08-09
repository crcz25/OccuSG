#!/usr/bin/env python3
"""Unit tests for the reusable semantic embedding helpers."""

import math

import numpy as np
import pytest

from scene_graph_core.algorithms.semantic import (
    accumulate_embedding,
    cosine_similarity,
    mean_embedding,
    normalize_embedding,
)


# ========== normalize_embedding ==========


def test_normalize_embedding_returns_unit_norm_vector():
    normalized = normalize_embedding([3.0, 4.0])
    assert normalized is not None
    assert normalized.dtype == np.float32
    assert math.isclose(float(np.linalg.norm(normalized)), 1.0, rel_tol=1e-6)
    assert np.allclose(normalized, [0.6, 0.8], atol=1e-6)


def test_normalize_embedding_is_idempotent():
    once = normalize_embedding([1.0, 2.0, 2.0])
    twice = normalize_embedding(once)
    assert np.allclose(once, twice, atol=1e-6)


@pytest.mark.parametrize(
    "embedding",
    [
        None,
        [],
        [0.0, 0.0, 0.0],
        [1.0, float("nan")],
        [float("inf"), 1.0],
        [[1.0, 0.0], [0.0, 1.0]],  # not 1-D
        "not-an-embedding",
    ],
)
def test_normalize_embedding_rejects_invalid_input(embedding):
    assert normalize_embedding(embedding) is None


# ========== cosine_similarity ==========


def test_cosine_similarity_of_identical_directions_is_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [5.0, 0.0]), 1.0, rel_tol=1e-6)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-6)


def test_cosine_similarity_of_opposite_vectors_is_negative_one():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-2.0, 0.0]), -1.0, rel_tol=1e-6)


def test_cosine_similarity_rejects_dimension_mismatch():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) is None


@pytest.mark.parametrize(
    "left,right",
    [
        (None, [1.0, 0.0]),
        ([1.0, 0.0], None),
        ([0.0, 0.0], [1.0, 0.0]),
        ([float("nan"), 1.0], [1.0, 0.0]),
    ],
)
def test_cosine_similarity_rejects_invalid_input(left, right):
    assert cosine_similarity(left, right) is None


# ========== accumulate_embedding ==========


def test_accumulator_seeds_from_first_observation():
    total, count, mean = accumulate_embedding(None, 0, [0.0, 3.0])
    assert count == 1
    assert np.allclose(total, [0.0, 1.0], atol=1e-6)
    assert np.allclose(mean, [0.0, 1.0], atol=1e-6)


def test_accumulator_matches_the_sum_of_unit_observations():
    observations = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    total, count, mean = None, 0, None
    for observation in observations:
        total, count, mean = accumulate_embedding(total, count, observation)

    expected_sum = sum(
        (np.asarray(o, dtype=np.float64) / np.linalg.norm(o) for o in observations),
        np.zeros(2),
    )
    assert count == 3
    assert np.allclose(total, expected_sum, atol=1e-6)
    assert np.allclose(mean, expected_sum / np.linalg.norm(expected_sum), atol=1e-6)
    # The stored mean equals normalize(sum / count).
    assert np.allclose(mean, mean_embedding(total, count), atol=1e-6)


def test_accumulator_is_order_independent():
    forward, backward = (None, 0), (None, 0)
    observations = [[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [3.0, 1.0]]
    for observation in observations:
        forward = accumulate_embedding(forward[0], forward[1], observation)[:2]
    for observation in reversed(observations):
        backward = accumulate_embedding(backward[0], backward[1], observation)[:2]
    assert forward[1] == backward[1] == 4
    assert np.allclose(forward[0], backward[0], atol=1e-6)


def test_accumulator_converges_towards_repeated_observation():
    total, count, mean = accumulate_embedding(None, 0, [1.0, 0.0])
    target = [0.0, 1.0]
    for _ in range(200):
        total, count, mean = accumulate_embedding(total, count, target)

    assert count == 201
    assert cosine_similarity(mean, target) > 0.99


def test_accumulator_ignores_invalid_new_observation():
    total, count, mean = accumulate_embedding(None, 0, [1.0, 0.0])
    for invalid in (None, [], [0.0, 0.0], [float("nan"), 0.0], [float("inf"), 1.0]):
        total, count, mean = accumulate_embedding(total, count, invalid)
        assert count == 1
        assert np.allclose(mean, [1.0, 0.0], atol=1e-6)


def test_accumulator_reseeds_on_dimension_mismatch():
    total, count, mean = accumulate_embedding([1.0, 0.0], 7, [0.0, 0.0, 4.0])
    assert count == 1
    assert np.allclose(mean, [0.0, 0.0, 1.0], atol=1e-6)


def test_accumulator_reseeds_from_invalid_accumulator():
    total, count, mean = accumulate_embedding([float("nan"), 0.0], 4, [2.0, 0.0])
    assert count == 1
    assert np.allclose(mean, [1.0, 0.0], atol=1e-6)


def test_accumulator_tolerates_invalid_observation_count():
    total, count, mean = accumulate_embedding([1.0, 0.0], "not-a-count", [0.0, 1.0])
    assert count == 1
    assert np.allclose(mean, [0.0, 1.0], atol=1e-6)


def test_cancelling_observations_leave_no_direction():
    total, count, mean = accumulate_embedding(None, 0, [1.0, 0.0])
    total, count, mean = accumulate_embedding(total, count, [-1.0, 0.0])
    assert count == 2
    assert np.allclose(total, [0.0, 0.0], atol=1e-6)
    # The sum has no direction, so no unit-norm mean exists.
    assert mean is None


def test_accumulator_mean_is_unit_norm():
    total, count, mean = accumulate_embedding([1.0, 0.0], 2, [0.0, 1.0])
    assert math.isclose(float(np.linalg.norm(mean)), 1.0, rel_tol=1e-6)


def test_mean_embedding_requires_observations():
    assert mean_embedding([1.0, 0.0], 0) is None
    assert mean_embedding(None, 3) is None
    assert np.allclose(mean_embedding([2.0, 0.0], 2), [1.0, 0.0], atol=1e-6)
