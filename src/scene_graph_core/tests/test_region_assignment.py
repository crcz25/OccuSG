#!/usr/bin/env python3
"""Unit tests for point-in-region assignment with boundary tolerance."""

import pytest

from scene_graph_core.algorithms.spatial import (
    assign_region,
    distance_to_polygon_boundary,
    point_in_polygon,
)

# Unit square with its lower-left corner at the origin.
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
# Disjoint square to the right, 1 m away from SQUARE's right edge.
FAR_SQUARE = [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]
# Overlaps SQUARE over x in [0.8, 1.8].
OVERLAPPING = [(0.8, 0.0), (1.8, 0.0), (1.8, 1.0), (0.8, 1.0)]


# ========== primitives ==========


def test_point_in_polygon_detects_interior_and_exterior():
    assert point_in_polygon((0.5, 0.5), SQUARE)
    assert not point_in_polygon((1.5, 0.5), SQUARE)
    assert not point_in_polygon((-0.5, 0.5), SQUARE)


def test_point_in_polygon_rejects_degenerate_polygons():
    assert not point_in_polygon((0.5, 0.5), [])
    assert not point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 0.0)])
    assert not point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, float("nan")), (1.0, 1.0)])


def test_point_in_polygon_handles_a_closed_ring():
    closed = SQUARE + [SQUARE[0]]
    assert point_in_polygon((0.5, 0.5), closed)


def test_distance_to_boundary_measures_the_nearest_edge():
    assert distance_to_polygon_boundary((1.25, 0.5), SQUARE) == pytest.approx(0.25)
    # Interior points report their distance to the boundary, not zero.
    assert distance_to_polygon_boundary((0.5, 0.5), SQUARE) == pytest.approx(0.5)


# ========== 1. strictly contained ==========


def test_strictly_contained_object_is_assigned():
    result = assign_region((0.5, 0.5), [(7, SQUARE)], 0.10)
    assert result.region_id == 7
    assert result.reason == "contained"


def test_containment_wins_over_a_nearer_boundary():
    # The point is inside SQUARE but only 0.05 m from FAR_SQUARE-like neighbour.
    near_neighbour = [(0.55, 0.0), (1.5, 0.0), (1.5, 1.0), (0.55, 1.0)]
    result = assign_region((0.5, 0.5), [(1, SQUARE), (2, near_neighbour)], 0.10)
    assert result.region_id == 1
    assert result.reason == "contained"


# ========== 2. outside all rooms ==========


def test_object_outside_every_region_is_unassigned():
    result = assign_region((5.0, 5.0), [(1, SQUARE), (2, FAR_SQUARE)], 0.10)
    assert result.region_id is None
    assert result.reason == "outside_all_regions"


def test_object_just_beyond_tolerance_is_not_forced_into_the_nearest_room():
    result = assign_region((1.2, 0.5), [(1, SQUARE)], 0.10)
    assert result.region_id is None
    assert result.boundary_distance == pytest.approx(0.2)


def test_no_regions_leaves_the_object_unassigned():
    assert assign_region((0.5, 0.5), [], 0.10).region_id is None
    assert assign_region((0.5, 0.5), [(1, [])], 0.10).reason == "no_valid_regions"


# ========== 3. boundary tolerance ==========


def test_object_within_tolerance_attaches_to_the_nearest_boundary():
    result = assign_region((1.05, 0.5), [(1, SQUARE)], 0.10)
    assert result.region_id == 1
    assert result.reason == "boundary_tolerance"
    assert result.boundary_distance == pytest.approx(0.05)


def test_tolerance_is_inclusive_at_its_exact_value():
    assert assign_region((1.10, 0.5), [(1, SQUARE)], 0.10).region_id == 1


def test_zero_tolerance_only_accepts_containment():
    assert assign_region((1.001, 0.5), [(1, SQUARE)], 0.0).region_id is None
    assert assign_region((0.5, 0.5), [(1, SQUARE)], 0.0).region_id == 1


def test_negative_tolerance_is_clamped_to_zero():
    assert assign_region((1.05, 0.5), [(1, SQUARE)], -1.0).region_id is None


def test_boundary_tolerance_picks_the_closest_of_several_regions():
    # 0.05 m right of SQUARE, 0.95 m left of FAR_SQUARE.
    result = assign_region((1.05, 0.5), [(1, SQUARE), (2, FAR_SQUARE)], 0.10)
    assert result.region_id == 1
    assert result.reason == "boundary_tolerance"


# ========== 4. multi-region overlap tie-breaking ==========


def test_overlapping_containment_is_deterministic():
    point = (0.9, 0.5)
    forward = assign_region(point, [(1, SQUARE), (2, OVERLAPPING)], 0.10)
    reverse = assign_region(point, [(2, OVERLAPPING), (1, SQUARE)], 0.10)
    assert forward.region_id == reverse.region_id
    assert forward.candidate_ids == (1, 2)


def test_equal_boundary_distances_break_on_region_id():
    # Symmetric gap: the point sits exactly between two mirrored squares.
    left = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    right = [(1.1, 0.0), (2.1, 0.0), (2.1, 1.0), (1.1, 1.0)]
    result = assign_region((1.05, 0.5), [(9, left), (4, right)], 0.10)
    assert result.region_id == 4  # lower ID wins the exact tie


def test_candidate_ids_report_every_valid_region():
    result = assign_region((0.5, 0.5), [(3, SQUARE), (1, FAR_SQUARE)], 0.10)
    assert result.candidate_ids == (1, 3)


# ========== invalid input ==========


@pytest.mark.parametrize(
    "point", [(float("nan"), 0.0), (0.0, float("inf")), ("a", 0.0), ()]
)
def test_invalid_points_are_rejected(point):
    assert assign_region(point, [(1, SQUARE)], 0.10).region_id is None


def test_invalid_polygons_are_skipped_without_failing_the_query():
    result = assign_region((0.5, 0.5), [(1, [(0.0, 0.0)]), (2, SQUARE)], 0.10)
    assert result.region_id == 2
