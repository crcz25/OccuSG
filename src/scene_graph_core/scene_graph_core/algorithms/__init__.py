"""
Algorithms module - Standalone algorithms.

Reusable algorithms used by foreground and background:
- spatial: Spatial algorithms (KD-tree, clustering, ray casting)
- graph: Graph algorithms (shortest path, connected components)

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex
"""

from .spatial import SpatialIndex, create_spatial_index

__all__ = [
    "SpatialIndex",
    "create_spatial_index",
]
