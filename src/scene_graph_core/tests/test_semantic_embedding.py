#!/usr/bin/env python3
"""Unit tests for the reusable semantic embedding helpers."""

import math

import numpy as np
import pytest

from scene_graph_core.algorithms.semantic import (
    cosine_similarity,
    normalize_embedding,
    running_mean_embedding,
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


# ========== running_mean_embedding ==========


def test_running_mean_seeds_from_first_observation():
    mean, count = running_mean_embedding(None, 0, [0.0, 3.0])
    assert count == 1
    assert np.allclose(mean, [0.0, 1.0], atol=1e-6)


def test_running_mean_matches_incremental_formula():
    e_old = normalize_embedding([1.0, 0.0])
    e_new = normalize_embedding([0.0, 1.0])
    mean, count = running_mean_embedding(e_old, 3, e_new)

    expected = (3 * e_old.astype(np.float64) + e_new) / 4
    expected = expected / np.linalg.norm(expected)

    assert count == 4
    assert np.allclose(mean, expected, atol=1e-6)


def test_running_mean_converges_towards_repeated_observation():
    mean, count = normalize_embedding([1.0, 0.0]), 1
    target = normalize_embedding([0.0, 1.0])
    for _ in range(200):
        mean, count = running_mean_embedding(mean, count, target)

    assert count == 201
    assert cosine_similarity(mean, target) > 0.99


def test_running_mean_ignores_invalid_new_observation():
    e_old = normalize_embedding([1.0, 0.0])
    mean, count = running_mean_embedding(e_old, 5, [float("nan"), 0.0])
    assert count == 5
    assert np.allclose(mean, e_old, atol=1e-6)


def test_running_mean_ignores_missing_new_observation():
    e_old = normalize_embedding([1.0, 0.0])
    mean, count = running_mean_embedding(e_old, 2, None)
    assert count == 2
    assert np.allclose(mean, e_old, atol=1e-6)


def test_running_mean_reseeds_on_dimension_mismatch():
    mean, count = running_mean_embedding([1.0, 0.0], 7, [0.0, 0.0, 4.0])
    assert count == 1
    assert np.allclose(mean, [0.0, 0.0, 1.0], atol=1e-6)


def test_running_mean_reseeds_from_invalid_old_representation():
    mean, count = running_mean_embedding([0.0, 0.0], 4, [2.0, 0.0])
    assert count == 1
    assert np.allclose(mean, [1.0, 0.0], atol=1e-6)


def test_running_mean_keeps_previous_mean_when_observations_cancel():
    e_old = normalize_embedding([1.0, 0.0])
    mean, count = running_mean_embedding(e_old, 1, [-1.0, 0.0])
    assert count == 1
    assert np.allclose(mean, e_old, atol=1e-6)


def test_running_mean_tolerates_invalid_observation_count():
    mean, count = running_mean_embedding([1.0, 0.0], "not-a-count", [0.0, 1.0])
    assert count == 1
    assert np.allclose(mean, [0.0, 1.0], atol=1e-6)


def test_running_mean_output_is_unit_norm():
    mean, _ = running_mean_embedding([1.0, 0.0], 2, [0.0, 1.0])
    assert math.isclose(float(np.linalg.norm(mean)), 1.0, rel_tol=1e-6)
