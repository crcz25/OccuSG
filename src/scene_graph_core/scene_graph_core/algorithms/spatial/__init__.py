"""
Spatial algorithms - Spatial data structures and operations.

Spatial algorithms:
- SpatialIndex: KD-tree for fast spatial queries (nearest-neighbor, range)
- region_assignment: point-in-polygon region membership with boundary tolerance
- clustering: Spatial clustering (rooms, objects)
- ray_casting: Bresenham, visibility tests

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex

    index = SpatialIndex()
    index.insert(node)
    nearest_id, dist = index.query_nearest(position, node_type=NodeType.OBJECT)
    results = index.query_radius(position, radius=5.0)
"""

from .region_assignment import (
    RegionAssignment,
    RegionCandidate,
    assign_region,
    distance_to_polygon_boundary,
    point_in_polygon,
)
from .spatial_index import SpatialIndex, create_spatial_index

__all__ = [
    "RegionAssignment",
    "RegionCandidate",
    "SpatialIndex",
    "assign_region",
    "create_spatial_index",
    "distance_to_polygon_boundary",
    "point_in_polygon",
]
