"""Point-in-region assignment with a boundary tolerance.

An object belongs to the region that strictly contains its representative
position. Boundary proximity is only a tie-breaker for points that fall outside
every region: the nearest boundary wins, and only within the configured
tolerance. Points farther than the tolerance stay unassigned — they are never
forced into the nearest region.

This module is deliberately free of ROS and of any particular polygon library:
callers pass simple ``(x, y)`` vertex sequences.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable, List, Optional, Sequence, Tuple

Point2D = Tuple[float, float]
Polygon2D = Sequence[Point2D]


@dataclass(frozen=True)
class RegionCandidate:
    """One region evaluated against a query point."""

    region_id: int
    contains: bool
    boundary_distance: float


@dataclass(frozen=True)
class RegionAssignment:
    """The resolved region for one query point."""

    region_id: Optional[int]
    reason: str
    boundary_distance: float = float("inf")
    candidate_ids: Tuple[int, ...] = ()


def _clean_polygon(polygon: Polygon2D) -> List[Point2D]:
    cleaned: List[Point2D] = []
    for vertex in polygon or ():
        try:
            x = float(vertex[0])
            y = float(vertex[1])
        except (IndexError, TypeError, ValueError):
            return []
        if not (isfinite(x) and isfinite(y)):
            return []
        cleaned.append((x, y))
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else []


def point_in_polygon(point: Point2D, polygon: Polygon2D) -> bool:
    """Return True when ``point`` is strictly inside ``polygon`` (ray casting)."""
    vertices = _clean_polygon(polygon)
    if not vertices:
        return False
    x, y = float(point[0]), float(point[1])
    if not (isfinite(x) and isfinite(y)):
        return False

    inside = False
    count = len(vertices)
    for index in range(count):
        x1, y1 = vertices[index]
        x2, y2 = vertices[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            # x coordinate where the edge crosses the horizontal ray through y.
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if crossing_x > x:
                inside = not inside
    return inside


def _distance_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / length_squared
    t = min(1.0, max(0.0, t))
    return hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def distance_to_polygon_boundary(point: Point2D, polygon: Polygon2D) -> float:
    """Return the shortest distance from ``point`` to the polygon boundary."""
    vertices = _clean_polygon(polygon)
    if not vertices:
        return float("inf")
    x, y = float(point[0]), float(point[1])
    if not (isfinite(x) and isfinite(y)):
        return float("inf")
    count = len(vertices)
    return min(
        _distance_to_segment((x, y), vertices[index], vertices[(index + 1) % count])
        for index in range(count)
    )


def evaluate_region(
    point: Point2D,
    region_id: int,
    polygon: Polygon2D,
) -> Optional[RegionCandidate]:
    """Evaluate one region, or return ``None`` when its polygon is unusable."""
    if not _clean_polygon(polygon):
        return None
    return RegionCandidate(
        region_id=int(region_id),
        contains=point_in_polygon(point, polygon),
        boundary_distance=distance_to_polygon_boundary(point, polygon),
    )


def assign_region(
    point: Point2D,
    regions: Iterable[Tuple[int, Polygon2D]],
    boundary_tolerance: float,
    distance_epsilon: float = 1e-9,
) -> RegionAssignment:
    """Assign ``point`` to at most one region.

    Args:
        point: Representative ``(x, y)`` position in the region coordinate frame.
        regions: ``(region_id, polygon)`` pairs; invalid polygons are skipped.
        boundary_tolerance: Maximum distance, in metres, at which a point outside
            every region may still be attached to the nearest boundary.
        distance_epsilon: Distances closer than this count as equal, after which
            the smaller region ID wins.

    Returns:
        A :class:`RegionAssignment`. ``region_id`` is ``None`` when the point is
        outside every region and beyond ``boundary_tolerance`` of all of them.
    """
    try:
        x = float(point[0])
        y = float(point[1])
    except (IndexError, TypeError, ValueError):
        return RegionAssignment(None, "invalid_point")
    if not (isfinite(x) and isfinite(y)):
        return RegionAssignment(None, "invalid_point")

    tolerance = float(boundary_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        tolerance = 0.0

    candidates: List[RegionCandidate] = []
    for region_id, polygon in regions:
        candidate = evaluate_region((x, y), region_id, polygon)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        return RegionAssignment(None, "no_valid_regions")

    candidate_ids = tuple(sorted(candidate.region_id for candidate in candidates))
    containing = [candidate for candidate in candidates if candidate.contains]
    if containing:
        # Overlapping regions: the deepest containment wins, then the lower ID.
        best = max(
            containing,
            key=lambda candidate: (
                round(candidate.boundary_distance / max(distance_epsilon, 1e-12)),
                -candidate.region_id,
            ),
        )
        return RegionAssignment(
            region_id=best.region_id,
            reason="contained",
            boundary_distance=best.boundary_distance,
            candidate_ids=candidate_ids,
        )

    nearest = min(
        candidates,
        key=lambda candidate: (
            round(candidate.boundary_distance / max(distance_epsilon, 1e-12)),
            candidate.region_id,
        ),
    )
    # Inclusive at the exact tolerance, with slack for floating-point distance.
    if nearest.boundary_distance <= tolerance + max(distance_epsilon, 0.0):
        return RegionAssignment(
            region_id=nearest.region_id,
            reason="boundary_tolerance",
            boundary_distance=nearest.boundary_distance,
            candidate_ids=candidate_ids,
        )

    return RegionAssignment(
        region_id=None,
        reason="outside_all_regions",
        boundary_distance=nearest.boundary_distance,
        candidate_ids=candidate_ids,
    )
