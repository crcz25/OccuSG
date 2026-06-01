"""
Graph Interface module - Shared interface for concurrent foreground/background access.

This module provides thread-safe access to the scene graph:
- SceneGraphInterface: Main facade combining query/update operations
- QueryService: Read-only queries with explicit XY/XYZ semantics
- UpdateService: Write operations with thread-safe locking
- SerializationInterface: Save/load functionality
- GraphPatch: Atomic batch updates

The recommended way to use the scene graph is through SceneGraphInterface:

    from scene_graph_core.graph_interface import SceneGraphInterface, GraphPatch

    sg = SceneGraphInterface()
    node_id = sg.update.add_node(node)  # Write operation (UpdateService)
    node = sg.query.graph.get_node(node_id)  # Read operation (QueryService)

    # Atomic batch updates
    patch = GraphPatch()
    patch.add_node(node).add_edge(edge)
    sg.update.apply_patch(patch)
"""

from ..services import GraphPatch, QueryService, UpdateService
from .scene_graph_interface import SceneGraphInterface, create_scene_graph_interface
from .serialization import SerializationInterface

__all__ = [
    "SceneGraphInterface",
    "create_scene_graph_interface",
    "QueryService",
    "UpdateService",
    "SerializationInterface",
    "GraphPatch",
]
