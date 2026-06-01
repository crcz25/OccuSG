"""
QueryService - Read-only query operations with explicit XY vs XYZ semantics.

This service provides high-level read-only queries for the scene graph.
It focuses on clarity and separation of concerns:

- Explicit XY (2D) vs XYZ (3D) distance queries
- Type-based filtering
- Attribute-based queries
- Spatial indexing when available

Key Design Principles:
1. Read-only operations (no mutations)
2. Explicit naming for 2D vs 3D queries
3. Uses spatial indexing when available for O(log n) performance
4. No business logic about rooms/objects - just generic graph queries
5. Thread-safe: all public query methods are protected by a reentrant lock (RLock)
"""

import threading
from typing import List, Optional, Tuple

from ..representation import BaseNode, Edge, EdgeType, NodeType, SceneGraph
from ..representation.geometry import Point


class QueryService:
    """
    Read-only query service for scene graph operations.

    This service provides spatial and semantic queries without business logic.
    For basic graph operations (get_node, get_all_nodes), use graph directly.

    All spatial queries have explicit XY (2D) or XYZ (3D) variants.
    XY queries ignore the Z coordinate for room matching and navigation.

    Thread Safety:
        All public query methods are protected by an internal ``threading.RLock``.
        A reentrant lock is used so that deprecated convenience methods (which
        delegate to their XYZ counterparts) can re-acquire the lock without
        deadlocking.

    Usage:
        query = QueryService(graph)

        # Find nodes by type
        rooms = query.find_nodes_by_type(NodeType.ROOM)

        # Find closest room (2D distance only)
        closest = query.find_closest_node_xy(position, NodeType.ROOM)

        # Find nodes in radius (3D distance)
        nearby = query.find_nodes_by_position_xyz(position, max_range=5.0)

    Attributes:
        graph: The underlying SceneGraph (read-only access)
        _lock: Reentrant lock protecting all query operations
    """

    def __init__(self, graph: SceneGraph):
        """
        Initialize the query service.

        Args:
            graph: The SceneGraph to query
        """
        self._graph = graph
        self._lock = threading.RLock()

    @property
    def graph(self) -> SceneGraph:
        """
        Direct access to the underlying SceneGraph for basic operations.

        Use this for simple CRUD operations:
        - graph.get_node(node_id)
        - graph.get_all_nodes()
        - graph.has_node(node_id)

        Returns:
            The underlying SceneGraph instance
        """
        return self._graph

    # ========== Basic Graph Accessors ==========

    def get_node(self, node_id: int) -> Optional[BaseNode]:
        """Get node by ID, or None if it does not exist."""
        with self._lock:
            try:
                return self._graph.get_node(node_id)
            except KeyError:
                return None

    def get_all_nodes(self) -> List[BaseNode]:
        """Get a snapshot of all nodes."""
        with self._lock:
            return self._graph.get_all_nodes()

    def has_node(self, node_id: int) -> bool:
        """Check if node exists."""
        with self._lock:
            return self._graph.has_node(node_id)

    def has_edge(
        self, source_id: int, target_id: int, edge_type: Optional[EdgeType] = None
    ) -> bool:
        """Check if an edge exists (optionally filtered by type)."""
        with self._lock:
            return self._graph.has_edge(source_id, target_id, edge_type=edge_type)

    def get_all_edges(self, edge_type: Optional[EdgeType] = None) -> List[Edge]:
        """Get all edges, optionally filtered by edge type."""
        with self._lock:
            if edge_type is None:
                return self._graph.get_all_edges()
            return self._graph.get_edges_by_type(edge_type)

    def get_outgoing_edges(
        self, node_id: int, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        """Get outgoing edges for a node, optionally filtered by type."""
        with self._lock:
            if not self._graph.has_node(node_id):
                return []
            return self._graph.get_outgoing_edges(node_id, edge_type=edge_type)

    def get_incoming_edges(
        self, node_id: int, edge_type: Optional[EdgeType] = None
    ) -> List[Edge]:
        """Get incoming edges for a node, optionally filtered by type."""
        with self._lock:
            if not self._graph.has_node(node_id):
                return []
            return self._graph.get_incoming_edges(node_id, edge_type=edge_type)

    # ========== Type-Based Queries ==========

    def find_nodes_by_type(
        self, node_type: NodeType, layer: Optional[str] = None
    ) -> List[BaseNode]:
        """
        Find all nodes of a specific type.

        Args:
            node_type: Type of node to search for
            layer: Optional layer filter (currently unused, for future extension)

        Returns:
            List of nodes matching the type
        """
        with self._lock:
            return self._graph.get_nodes_by_type(node_type)

    # ========== Attribute-Based Queries ==========

    def find_nodes_by_attribute(self, attribute: str, value) -> List[BaseNode]:
        """
        Find all nodes with a specific attribute value.

        Args:
            attribute: Name of the attribute in node.attributes
            value: Value to match

        Returns:
            List of nodes matching the criteria
        """
        with self._lock:
            result = []
            for node in self._graph.get_all_nodes():
                if node.attributes and node.attributes.get(attribute) == value:
                    result.append(node)
            return result

    # ========== Spatial Queries (2D - XY only) ==========

    def find_closest_node_xy(
        self,
        position: Point,
        node_type: Optional[NodeType] = None,
        max_distance: Optional[float] = None,
    ) -> Optional[Tuple[BaseNode, float]]:
        """
        Find the closest node using 2D distance (XY plane only, Z ignored).

        This is the primary method for room matching and 2D navigation.

        Uses spatial indexing when available for O(log n) performance.

        Args:
            position: Position to search from (ROS Point, Z ignored)
            node_type: Optional type filter
            max_distance: Optional maximum distance (2D)

        Returns:
            Tuple of (node, distance) or None if no node found
        """
        with self._lock:
            spatial_index = self._graph.spatial_index
            if spatial_index is not None:
                result = spatial_index.query_nearest_xy(
                    position,
                    node_type=node_type,
                    k=1,
                    max_distance=max_distance,
                )
                if result is not None:
                    node_id, distance = result
                    try:
                        return (self._graph.get_node(node_id), distance)
                    except KeyError:
                        pass

            min_dist = float("inf")
            closest_node = None

            nodes_to_search = (
                self._graph.get_nodes_by_type(node_type)
                if node_type
                else self._graph.get_all_nodes()
            )

            for node in nodes_to_search:
                # 2D distance only (XY)
                dx = node.pose.position.x - position.x
                dy = node.pose.position.y - position.y
                dist = (dx**2 + dy**2) ** 0.5

                if dist < min_dist:
                    if max_distance is None or dist <= max_distance:
                        min_dist = dist
                        closest_node = node

            if closest_node is not None:
                return (closest_node, min_dist)
            return None

    def find_nodes_by_position_xy(
        self,
        position: Point,
        max_range: float,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Find all nodes within a 2D radius (XY plane only, Z ignored).

        This is useful for finding objects/rooms in the same floor.

        Args:
            position: Center position (ROS Point, Z ignored)
            max_range: Maximum 2D distance (meters)
            node_type: Optional type filter

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        with self._lock:
            spatial_index = self._graph.spatial_index
            if spatial_index is not None:
                results = spatial_index.query_radius_xy(
                    position,
                    max_range,
                    node_type=node_type,
                )
                nodes: List[Tuple[BaseNode, float]] = []
                for node_id, distance in results:
                    try:
                        nodes.append((self._graph.get_node(node_id), distance))
                    except KeyError:
                        continue
                return nodes

            result = []
            nodes_to_search = (
                self._graph.get_nodes_by_type(node_type)
                if node_type
                else self._graph.get_all_nodes()
            )

            for node in nodes_to_search:
                # 2D distance only (XY)
                dx = node.pose.position.x - position.x
                dy = node.pose.position.y - position.y
                dist = (dx**2 + dy**2) ** 0.5

                if dist <= max_range:
                    result.append((node, dist))

            # Sort by distance
            result.sort(key=lambda x: x[1])
            return result

    # ========== Spatial Queries (3D - XYZ) ==========

    def find_closest_node_xyz(
        self,
        position: Point,
        node_type: Optional[NodeType] = None,
        max_distance: Optional[float] = None,
    ) -> Optional[Tuple[BaseNode, float]]:
        """
        Find the closest node using 3D distance (XYZ).

        This is useful for 3D navigation or object detection.

        Uses spatial indexing when available for O(log n) performance.

        Args:
            position: Position to search from (ROS Point)
            node_type: Optional type filter
            max_distance: Optional maximum distance (3D)

        Returns:
            Tuple of (node, distance) or None if no node found
        """
        with self._lock:
            # Try spatial index first
            spatial_index = self._graph.spatial_index
            if spatial_index is not None:
                result = spatial_index.query_nearest(
                    position,
                    node_type=node_type,
                    k=1,
                    max_distance=max_distance,
                )
                if result is not None:
                    node_id, distance = result
                    try:
                        node = self._graph.get_node(node_id)
                        return (node, distance)
                    except KeyError:
                        pass  # Node was removed

            # Fallback: brute-force
            min_dist = float("inf")
            closest_node = None

            nodes_to_search = (
                self._graph.get_nodes_by_type(node_type)
                if node_type
                else self._graph.get_all_nodes()
            )

            for node in nodes_to_search:
                # 3D distance (XYZ)
                dx = node.pose.position.x - position.x
                dy = node.pose.position.y - position.y
                dz = node.pose.position.z - position.z
                dist = (dx**2 + dy**2 + dz**2) ** 0.5

                if dist < min_dist:
                    if max_distance is None or dist <= max_distance:
                        min_dist = dist
                        closest_node = node

            if closest_node is not None:
                return (closest_node, min_dist)
            return None

    def find_nodes_by_position_xyz(
        self,
        position: Point,
        max_range: float,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Find all nodes within a 3D radius (XYZ).

        Uses spatial indexing when available for O(k + log n) performance.

        Args:
            position: Center position (ROS Point)
            max_range: Maximum 3D distance (meters)
            node_type: Optional type filter

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        with self._lock:
            # Try spatial index first
            spatial_index = self._graph.spatial_index
            if spatial_index is not None:
                results = spatial_index.query_radius(
                    position, max_range, node_type=node_type
                )
                # Convert to (node, distance) tuples
                nodes = []
                for node_id, distance in results:
                    try:
                        node = self._graph.get_node(node_id)
                        nodes.append((node, distance))
                    except KeyError:
                        pass  # Node was removed
                return nodes

            # Fallback: brute-force
            result = []
            nodes_to_search = (
                self._graph.get_nodes_by_type(node_type)
                if node_type
                else self._graph.get_all_nodes()
            )

            for node in nodes_to_search:
                # 3D distance (XYZ)
                dx = node.pose.position.x - position.x
                dy = node.pose.position.y - position.y
                dz = node.pose.position.z - position.z
                dist = (dx**2 + dy**2 + dz**2) ** 0.5

                if dist <= max_range:
                    result.append((node, dist))

            # Sort by distance
            result.sort(key=lambda x: x[1])
            return result

    def find_k_nearest_nodes_xyz(
        self,
        position: Point,
        k: int,
        node_type: Optional[NodeType] = None,
        max_distance: Optional[float] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Find the k nearest nodes using 3D distance (XYZ).

        Uses spatial indexing when available for O(k log n) performance.

        Args:
            position: Position to search from (ROS Point)
            k: Number of nearest neighbors to return
            node_type: Optional type filter
            max_distance: Optional maximum distance

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        with self._lock:
            # Try spatial index first
            spatial_index = self._graph.spatial_index
            if spatial_index is not None:
                results = spatial_index.query_nearest(
                    position,
                    node_type=node_type,
                    k=k,
                    max_distance=max_distance,
                )
                if results is None:
                    return []
                if isinstance(results, tuple):
                    # Single result for k=1
                    results = [results]

                # Convert to (node, distance) tuples
                nodes = []
                for node_id, distance in results:
                    try:
                        node = self._graph.get_node(node_id)
                        nodes.append((node, distance))
                    except KeyError:
                        pass  # Node was removed
                return nodes

            # Fallback: brute-force
            nodes_to_search = (
                self._graph.get_nodes_by_type(node_type)
                if node_type
                else self._graph.get_all_nodes()
            )

            distances = []
            for node in nodes_to_search:
                dx = node.pose.position.x - position.x
                dy = node.pose.position.y - position.y
                dz = node.pose.position.z - position.z
                dist = (dx**2 + dy**2 + dz**2) ** 0.5

                if max_distance is None or dist <= max_distance:
                    distances.append((node, dist))

            # Sort by distance and return top k
            distances.sort(key=lambda x: x[1])
            return distances[:k]

    # ========== Edge Queries ==========

    def get_neighbors(self, node_id: int) -> List[int]:
        """
        Get IDs of all neighbor nodes (nodes connected by outgoing edges).

        Args:
            node_id: ID of the source node

        Returns:
            List of neighbor node IDs
        """
        with self._lock:
            return self._graph.get_neighbors(node_id)

    def has_path(self, source_id: int, target_id: int) -> bool:
        """
        Check if there is a path from source to target.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node

        Returns:
            True if a path exists, False otherwise
        """
        with self._lock:
            return self._graph.has_path(source_id, target_id)

    def get_objects_in_room(self, room_id: int) -> List[BaseNode]:
        """
        Get all objects connected to a specific room.

        Business logic: Finds all OBJECT nodes that have an edge from the given room.

        Args:
            room_id: ID of the room node

        Returns:
            List of object nodes in the room

        Raises:
            KeyError: If room does not exist
        """
        with self._lock:
            if not self._graph.has_node(room_id):
                raise KeyError(f"Room with ID {room_id} does not exist.")

            # Get all neighbors of the room
            neighbor_ids = self._graph.get_neighbors(room_id)

            # Filter for OBJECT nodes
            objects = []
            for node_id in neighbor_ids:
                node = self._graph.get_node(node_id)
                if node.node_type == NodeType.OBJECT:
                    objects.append(node)

            return objects

    # ========== Backward Compatibility Methods (Deprecated) ==========
    # These methods are provided for backward compatibility but are deprecated.
    # Use the explicit XY or XYZ methods instead.

    def find_closest_node(
        self,
        position: Point,
        node_type: Optional[NodeType] = None,
        max_distance: Optional[float] = None,
    ) -> Optional[Tuple[BaseNode, float]]:
        """
        Find the closest node using 3D distance (XYZ).

        DEPRECATED: Use find_closest_node_xyz() or find_closest_node_xy() for explicit semantics.

        This method defaults to 3D distance for backward compatibility.

        Args:
            position: Position to search from (ROS Point)
            node_type: Optional type filter
            max_distance: Optional maximum distance (3D)

        Returns:
            Tuple of (node, distance) or None if no node found
        """
        with self._lock:
            return self.find_closest_node_xyz(position, node_type, max_distance)

    def find_nodes_by_position(
        self,
        position: Point,
        max_range: float,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Find all nodes within a 3D radius (XYZ).

        DEPRECATED: Use find_nodes_by_position_xyz() or find_nodes_by_position_xy() for explicit semantics.

        This method defaults to 3D distance for backward compatibility.

        Args:
            position: Center position (ROS Point)
            max_range: Maximum 3D distance (meters)
            node_type: Optional type filter

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        with self._lock:
            return self.find_nodes_by_position_xyz(position, max_range, node_type)

    def find_k_nearest_nodes(
        self,
        position: Point,
        k: int,
        node_type: Optional[NodeType] = None,
        max_distance: Optional[float] = None,
    ) -> List[Tuple[BaseNode, float]]:
        """
        Find the k nearest nodes using 3D distance (XYZ).

        DEPRECATED: Use find_k_nearest_nodes_xyz() for explicit semantics.

        This method defaults to 3D distance for backward compatibility.

        Args:
            position: Position to search from (ROS Point)
            k: Number of nearest neighbors to return
            node_type: Optional type filter
            max_distance: Optional maximum distance

        Returns:
            List of (node, distance) tuples, sorted by distance
        """
        with self._lock:
            return self.find_k_nearest_nodes_xyz(position, k, node_type, max_distance)
