"""Tests for region geometry preparation and overlap helpers."""

from pytest import approx

from scene_graph_ros.managers.region_util import (
    intersection_area,
    iou,
    point_in_polygon,
    prepare_region_geometry,
    repair_polygon,
    union_area,
)


def _rect(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def test_point_in_polygon_is_boundary_inclusive():
    """Verify polygon containment treats boundaries as inside."""
    polygon, _ = repair_polygon(_rect(0.0, 0.0, 2.0, 2.0))

    assert point_in_polygon((0.0, 1.0), polygon) is True
    assert point_in_polygon((2.0, 2.0), polygon) is True
    assert point_in_polygon((3.0, 1.0), polygon) is False


def test_overlap_metrics_match_expected_values():
    """Verify intersection, union, and IoU for overlapping rectangles."""
    polygon_a, _ = repair_polygon(_rect(0.0, 0.0, 2.0, 2.0))
    polygon_b, _ = repair_polygon(_rect(1.0, 0.0, 3.0, 2.0))

    assert intersection_area(polygon_a, polygon_b) == approx(2.0)
    assert union_area(polygon_a, polygon_b) == approx(6.0)
    assert iou(polygon_a, polygon_b) == approx(1.0 / 3.0)


def test_prepare_region_geometry_prefers_convex_hull_when_requested():
    """Verify convex-hull preference is honored during geometry preparation."""
    geometry = prepare_region_geometry(
        polygon_points=_rect(0.0, 0.0, 1.0, 1.0),
        convex_hull_points=_rect(0.0, 0.0, 2.0, 2.0),
        use_convex_hull=True,
    )

    assert geometry.is_valid is True
    assert geometry.source == "convex_hull"
    assert geometry.area == approx(4.0)


def test_buffer_zero_repair_keeps_largest_polygon_piece():
    """Verify invalid polygons are repaired to their largest polygon piece."""
    polygon, repaired = repair_polygon(
        (
            (0.0, 0.0),
            (6.0, 0.0),
            (1.0, 1.0),
            (6.0, 6.0),
            (0.0, 6.0),
            (2.0, 2.0),
        )
    )

    assert repaired is True
    assert polygon is not None
    assert polygon.area == approx(12.0)


def test_invalid_geometry_is_marked_for_skip():
    """Verify unusable region geometry reports a deterministic skip reason."""
    geometry = prepare_region_geometry(
        polygon_points=((0.0, 0.0), (1.0, 1.0)),
        convex_hull_points=(),
        use_convex_hull=False,
    )

    assert geometry.is_valid is False
    assert geometry.skip_reason == "invalid geometry"
