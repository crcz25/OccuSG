"""
Algorithms module - Standalone algorithms.

Reusable algorithms used by foreground and background:
- spatial: Spatial algorithms (KD-tree, clustering, ray casting)
- semantic: Embedding validation, similarity, and online aggregation
- graph: Graph algorithms (shortest path, connected components)

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex
    from scene_graph_core.algorithms.semantic import cosine_similarity
"""

from .semantic import (
    cosine_similarity,
    normalize_embedding,
    update_running_mean_embedding,
)
from .spatial import (
    RegionAssignment,
    SpatialIndex,
    assign_region,
    create_spatial_index,
    distance_to_polygon_boundary,
    point_in_polygon,
)

__all__ = [
    "RegionAssignment",
    "SpatialIndex",
    "assign_region",
    "cosine_similarity",
    "create_spatial_index",
    "distance_to_polygon_boundary",
    "normalize_embedding",
    "point_in_polygon",
    "update_running_mean_embedding",
]
