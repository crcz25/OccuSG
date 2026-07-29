"""
Algorithms module - Standalone algorithms.

Reusable algorithms used by foreground and background:
- spatial: Spatial algorithms (KD-tree, clustering, ray casting)
- semantic: Embedding validation, similarity, online aggregation, class evidence
- graph: Graph algorithms (shortest path, connected components)

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex
    from scene_graph_core.algorithms.semantic import cosine_similarity
"""

from .semantic import (
    accumulate_class_evidence,
    accumulate_embedding,
    canonical_class_from_evidence,
    cosine_similarity,
    mean_embedding,
    normalize_embedding,
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
    "accumulate_class_evidence",
    "accumulate_embedding",
    "assign_region",
    "canonical_class_from_evidence",
    "cosine_similarity",
    "create_spatial_index",
    "distance_to_polygon_boundary",
    "mean_embedding",
    "normalize_embedding",
    "point_in_polygon",
]
