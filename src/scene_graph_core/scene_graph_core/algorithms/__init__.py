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

from .semantic import cosine_similarity, normalize_embedding, running_mean_embedding
from .spatial import SpatialIndex, create_spatial_index

__all__ = [
    "SpatialIndex",
    "create_spatial_index",
    "cosine_similarity",
    "normalize_embedding",
    "running_mean_embedding",
]
