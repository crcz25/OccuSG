"""Shared helpers for preparing and querying tracker-region geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class PreparedGeometry:
    """Geometry chosen for region checks after repair and fallback."""

    polygon: Optional[Polygon]
    points: Tuple[Point2D, ...]
    area: float
    source: str
    repaired: bool
    skip_reason: Optional[str]

    @property
    def is_valid(self) -> bool:
        """Return True when a usable polygon exists."""
        return self.polygon is not None and not self.polygon.is_empty


def prepare_region_geometry(
    polygon_points: Sequence[Point2D],
    convex_hull_points: Sequence[Point2D],
    use_convex_hull: bool,
) -> PreparedGeometry:
    """Select and repair the geometry used for region evaluation."""
    candidates = (
        (
            ("convex_hull", convex_hull_points),
            ("polygon", polygon_points),
        )
        if use_convex_hull
        else (
            ("polygon", polygon_points),
            ("convex_hull", convex_hull_points),
        )
    )

    for source, points in candidates:
        polygon, repaired = repair_polygon(points)
        if polygon is not None:
            normalized_points = polygon_to_points(polygon)
            return PreparedGeometry(
                polygon=polygon,
                points=normalized_points,
                area=float(polygon.area),
                source=source,
                repaired=repaired,
                skip_reason=None,
            )

    return PreparedGeometry(
        polygon=None,
        points=(),
        area=0.0,
        source="convex_hull" if use_convex_hull else "polygon",
        repaired=False,
        skip_reason="invalid geometry",
    )


def repair_polygon(
    points: Sequence[Point2D],
) -> Tuple[Optional[Polygon], bool]:
    """Build a usable polygon, repairing invalid shapes with buffer(0)."""
    if len(points) < 3:
        return None, False

    try:
        geometry = Polygon(points)
    except Exception:
        return None, False

    repaired = False
    if geometry.is_empty:
        return None, repaired
    if not geometry.is_valid:
        try:
            geometry = geometry.buffer(0.0)
            repaired = True
        except Exception:
            return None, repaired

    polygon = _largest_polygon(geometry)
    if polygon is None or polygon.is_empty or polygon.area <= 0.0:
        return None, repaired
    return polygon, repaired


def point_in_polygon(point: Point2D, polygon: Optional[Polygon]) -> bool:
    """Return True when a point lies in or on the boundary of a polygon."""
    if polygon is None or polygon.is_empty:
        return False
    return bool(polygon.covers(Point(float(point[0]), float(point[1]))))


def intersection_area(
    polygon_a: Optional[Polygon],
    polygon_b: Optional[Polygon],
) -> float:
    """Return polygon intersection area."""
    if polygon_a is None or polygon_b is None:
        return 0.0
    return float(polygon_a.intersection(polygon_b).area)


def union_area(
    polygon_a: Optional[Polygon],
    polygon_b: Optional[Polygon],
) -> float:
    """Return polygon union area."""
    if polygon_a is None or polygon_b is None:
        return 0.0
    return float(polygon_a.union(polygon_b).area)


def iou(
    polygon_a: Optional[Polygon],
    polygon_b: Optional[Polygon],
) -> float:
    """Return polygon intersection-over-union."""
    union_value = union_area(polygon_a, polygon_b)
    if union_value <= 0.0:
        return 0.0
    return float(intersection_area(polygon_a, polygon_b) / union_value)


def polygon_to_points(polygon: Polygon) -> Tuple[Point2D, ...]:
    """Convert a Shapely polygon into a deterministic point tuple."""
    return tuple(
        (float(x), float(y))
        for x, y in list(polygon.exterior.coords)[:-1]
    )


def _largest_polygon(geometry: object) -> Optional[Polygon]:
    """Return the largest polygon from a repaired geometry result."""
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
        return max(polygons, key=lambda item: item.area) if polygons else None
    if isinstance(geometry, GeometryCollection):
        polygons = [
            item for item in geometry.geoms if isinstance(item, Polygon)
        ]
        return max(polygons, key=lambda item: item.area) if polygons else None
    return None
