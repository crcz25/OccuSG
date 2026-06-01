"""
Scene Graph Interface - Main facade for scene graph operations.

This is the primary entry point for all scene graph operations. It exposes
services for different concerns:
- query: Read-only operations (QueryService)
- update: Thread-safe write operations (UpdateService)
- serialize: Save/load operations (SerializationInterface)

The interface directly exposes the services layer without intermediate wrappers.

Usage:
    # Create interface
    sg = SceneGraphInterface()

    # Read operations (via query service or graph directly)
    node = sg.query.graph.get_node(node_id)  # Basic read
    closest_xy = sg.query.find_closest_node_xy(position, NodeType.ROOM)  # XY query

    # Write operations (via update service - thread-safe)
    node_id = sg.update.add_node(node)
    sg.update.remove_node(node_id)

    # Atomic batch updates
    patch = GraphPatch()
    patch.add_node(node).add_edge(edge)
    sg.update.apply_patch(patch)

    # Serialization
    sg.serialize.save("graph.json")
"""

from ..representation import SceneGraph
from ..services import GraphPatch, QueryService, UpdateService
from .serialization import SerializationInterface


class SceneGraphInterface:
    """
    Unified interface for scene graph access.

    This class serves as the main entry point, providing access to:
    - query: Read-only operations (QueryService)
    - update: Thread-safe write operations (UpdateService)
    - serialize: Save/load operations (SerializationInterface)

    Architecture (refactored):
        SceneGraphInterface
        ├── query (QueryService - read operations)
        │   └── graph (SceneGraph - direct access)
        ├── update (UpdateService - write operations)
        │   └── graph (SceneGraph - via lock)
        └── serialize (SerializationInterface)
            └── graph (SceneGraph)

    Thread Safety:
        - Read operations are concurrent-safe (no locks)
        - Write operations use RLock in UpdateService
        - All services share the same SceneGraph instance

    Example:
        >>> sg = SceneGraphInterface()
        >>>
        >>> # Read - basic operation (direct graph access)
        >>> node = sg.query.graph.get_node(node_id)
        >>>
        >>> # Read - spatial queries with explicit XY/XYZ semantics
        >>> closest_xy = sg.query.find_closest_node_xy(position, NodeType.ROOM)
        >>> closest_xyz = sg.query.find_closest_node_xyz(position, NodeType.OBJECT)
        >>>
        >>> # Read - business logic
        >>> objects = sg.query.get_objects_in_room(room_id)
        >>>
        >>> # Write - thread-safe
        >>> node_id = sg.update.add_node(node)
        >>>
        >>> # Write - atomic batch
        >>> patch = GraphPatch()
        >>> patch.add_node(node).add_edge(edge)
        >>> sg.update.apply_patch(patch)
        >>>
        >>> # Serialize
        >>> sg.serialize.save("graph.json")
    """

    def __init__(self):
        """Initialize the scene graph interface with all services."""
        # Core storage (shared by all services)
        self._graph = SceneGraph()

        # Services layer - directly exposed as public API
        self.query = QueryService(self._graph)
        self.update = UpdateService(self._graph)
        self.serialize = SerializationInterface(self._graph)

    # ========== Metadata Methods ==========
    # These don't duplicate functionality, they provide convenience

    def get_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        return self.query.graph.node_count()

    def get_edge_count(self) -> int:
        """Get total number of edges in the graph."""
        return self.query.graph.edge_count()

    def __repr__(self) -> str:
        """String representation of the interface."""
        return (
            f"SceneGraphInterface("
            f"nodes={self.get_node_count()}, "
            f"edges={self.get_edge_count()})"
        )


# ========== Factory Function for Dependency Injection ==========


def create_scene_graph_interface() -> SceneGraphInterface:
    """
    Factory function to create a SceneGraphInterface.

    This is the recommended way to create instances, as it:
    - Makes testing easier (can be mocked)
    - Makes dependencies explicit
    - Allows for future configuration options

    Returns:
        A new SceneGraphInterface instance

    Example:
        >>> from scene_graph_core.graph_interface import create_scene_graph_interface
        >>> sg = create_scene_graph_interface()
    """
    return SceneGraphInterface()
