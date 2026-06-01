"""
Spatial algorithms - Spatial data structures and operations.

Spatial algorithms:
- SpatialIndex: KD-tree for fast spatial queries (nearest-neighbor, range)
- clustering: Spatial clustering (rooms, objects)
- ray_casting: Bresenham, visibility tests

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex

    index = SpatialIndex()
    index.insert(node)
    nearest_id, dist = index.query_nearest(position, node_type=NodeType.OBJECT)
    results = index.query_radius(position, radius=5.0)
"""

from .spatial_index import SpatialIndex, create_spatial_index

__all__ = [
    "SpatialIndex",
    "create_spatial_index",
]
