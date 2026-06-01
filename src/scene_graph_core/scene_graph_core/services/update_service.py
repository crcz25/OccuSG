"""
UpdateService - Thread-safe write operations for the scene graph.

This service provides high-level mutation operations with thread safety.
It wraps basic graph operations with locks and provides utility methods
for common update patterns.

Key Design Principles:
1. Thread-safe mutations with RLock
2. Spatial index synchronization
3. Edge management utilities
4. Atomic batch updates via GraphPatch
5. No domain logic about rooms/objects
"""

import threading
from typing import TYPE_CHECKING, Iterable, List, Optional

from ..representation import BaseNode, Edge, EdgeType, SceneGraph

if TYPE_CHECKING:
    from .graph_patch import GraphPatch


class UpdateService:
    """
    Thread-safe update service for scene graph mutations.

    All write operations are protected by a reentrant lock (RLock) to ensure
    thread safety when foreground and background processes modify the graph.

    This service focuses on generic graph mutations without business logic.
    Domain-specific logic (e.g., room assignment) should be in the ROS layer.

    Usage:
        update = UpdateService(graph)

        # Add node
        node_id = update.add_node(node)

        # Update node
        update.update_node(node_id, new_node)

        # Remove node (also removes incident edges)
        update.remove_node(node_id)

        # Edge operations
        update.add_edge(edge)
        update.remove_edge(source_id, target_id, edge_type)
        update.replace_edge_of_type(node_id, EdgeType.ROOM_CONTAINS, new_target_id)
    """

    def __init__(self, graph: SceneGraph):
        """
        Initialize the update service.

        Args:
            graph: The SceneGraph to update
        """
        self._graph = graph
        self._lock = threading.RLock()  # Reentrant lock for thread safety

    # ========== Node Operations ==========

    def add_node(self, node: BaseNode) -> int:
        """
        Add a node to the scene graph (thread-safe).

        The spatial index is automatically updated if enabled.

        Args:
            node: Node to add

        Returns:
            The assigned node ID

        Raises:
            TypeError: If node is not a BaseNode instance
            ValueError: If node ID already exists
        """
        with self._lock:
            return self._graph.add_node(node)

    def update_node(self, node_id: int, new_node: BaseNode) -> None:
        """
        Update an existing node (thread-safe).

        The spatial index is automatically updated if enabled.

        Args:
            node_id: ID of the node to update
            new_node: New node data

        Raises:
            KeyError: If node does not exist
            TypeError: If new_node is not a BaseNode instance
        """
        with self._lock:
            self._graph.update_node(node_id, new_node)

    def update_nodes(self, node_updates: Iterable[tuple[int, BaseNode]]) -> None:
        """Update multiple nodes in one lock acquisition."""
        with self._lock:
            self._graph.update_nodes(node_updates)

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the scene graph (thread-safe).

        All edges connected to this node are also removed.
        The spatial index is automatically updated if enabled.

        Args:
            node_id: ID of the node to remove

        Raises:
            KeyError: If node does not exist
        """
        with self._lock:
            self._graph.remove_node(node_id)

    def upsert_node(self, node: BaseNode) -> int:
        """
        Add or update a node (thread-safe).

        If node.id is set and exists, updates the node.
        Otherwise, adds a new node.

        Args:
            node: Node to upsert

        Returns:
            The node ID
        """
        with self._lock:
            if node.id is not None and self._graph.has_node(node.id):
                self._graph.update_node(node.id, node)
                return node.id
            else:
                return self._graph.add_node(node)

    # ========== Edge Operations ==========

    def add_edge(self, edge: Edge, is_structural: bool = True) -> int:
        """
        Add an edge to the scene graph (thread-safe).

        Args:
            edge: Edge to add
            is_structural: Whether this edge participates in the semantic hierarchy.
                          True = structural (hierarchical, must not create cycles)
                          False = relational (non-hierarchical, cycles allowed)

        Returns:
            The assigned edge ID

        Raises:
            TypeError: If edge is not an Edge instance
            ValueError: If source or target nodes don't exist, or structural edge creates cycle

        Note:
            - Structural edges (ROOM_CONTAINS, etc.) define the hierarchical backbone
            - Relational edges (NAVIGABLE_PATH, OBSERVATION_ANCHOR, etc.) are non-hierarchical
        """
        with self._lock:
            return self._graph.add_edge(edge, is_structural)

    def remove_edge(
        self, source_id: int, target_id: int, edge_type: Optional[EdgeType] = None
    ) -> bool:
        """
        Remove an edge from the scene graph (thread-safe).

        If edge_type is specified, only removes edges of that type.
        Otherwise, removes all edges between source and target.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Optional edge type filter

        Returns:
            True if any edge was removed, False otherwise
        """
        with self._lock:
            return (
                self._graph.remove_edges([(source_id, target_id, edge_type)]) > 0
            )

    def remove_edges_of_type(self, node_id: int, edge_type: EdgeType) -> int:
        """
        Remove all outgoing edges of a specific type from a node.

        Args:
            node_id: ID of the source node
            edge_type: Type of edges to remove

        Returns:
            Number of edges removed
        """
        with self._lock:
            if not self._graph.has_node(node_id):
                return 0

            edge_refs = [
                (edge.source_id, edge.target_id, edge_type)
                for edge in self._graph.get_outgoing_edges(node_id, edge_type=edge_type)
            ]
            return self._graph.remove_edges(edge_refs)

    def replace_edge_of_type(
        self, source_id: int, edge_type: EdgeType, new_target_id: int
    ) -> None:
        """
        Replace an edge of a specific type with a new target.

        This is useful for single-ownership relationships.
        It removes all existing outgoing edges of the specified type from the
        source node and creates a new edge to the new target.

        Args:
            source_id: ID of the source node
            edge_type: Type of edge to replace
            new_target_id: ID of the new target node

        Raises:
            KeyError: If source or target node does not exist
        """
        with self._lock:
            # Remove existing edges of this type
            self.remove_edges_of_type(source_id, edge_type)

            # Add new edge
            edge = Edge(
                source_id=source_id,
                target_id=new_target_id,
                type=edge_type,
            )
            self._graph.add_edge(edge)

    # ========== Batch Operations ==========

    def add_nodes(self, nodes: List[BaseNode]) -> List[int]:
        """
        Add multiple nodes (thread-safe, single lock acquisition).

        More efficient than calling add_node multiple times.

        Args:
            nodes: List of nodes to add

        Returns:
            List of assigned node IDs
        """
        with self._lock:
            return [self._graph.add_node(node) for node in nodes]

    def add_edges(
        self, edges: List[Edge] | List[tuple[Edge, bool]], is_structural: bool = True
    ) -> List[int]:
        """
        Add multiple edges in a single atomic operation (thread-safe).

        More efficient than calling add_edge multiple times.

        Args:
            edges: Either:
                   - List of Edge instances (all use is_structural parameter)
                   - List of (edge, is_structural) tuples for per-edge control
            is_structural: Default is_structural value when edges is List[Edge].
                          Ignored when edges contains tuples.

        Returns:
            List of assigned edge IDs in the same order

        Raises:
            TypeError: If any edge is not an Edge instance
            ValueError: If any edge validation fails

        Examples:
            # All edges are structural
            update.add_edges([edge1, edge2])

            # All edges are non-structural
            update.add_edges([edge1, edge2], is_structural=False)

            # Per-edge control
            update.add_edges([(edge1, True), (edge2, False)])
        """
        with self._lock:
            return self._graph.add_edges(edges, is_structural=is_structural)

    def remove_nodes(self, node_ids: List[int]) -> None:
        """
        Remove multiple nodes (thread-safe, single lock acquisition).

        All edges connected to these nodes are also removed.

        Args:
            node_ids: List of node IDs to remove
        """
        with self._lock:
            for node_id in node_ids:
                if self._graph.has_node(node_id):
                    self._graph.remove_node(node_id)

    # ========== Utility Operations ==========

    def reconnect_node(
        self, node_id: int, new_parent_id: int, edge_type: EdgeType
    ) -> None:
        """
        Reconnect a node to a new parent by replacing its edge.

        This is a convenience method that combines remove_edges_of_type
        and add_edge in a single operation.

        Args:
            node_id: ID of the node to reconnect
            new_parent_id: ID of the new parent node
            edge_type: Type of edge to create

        Raises:
            KeyError: If node or parent does not exist
        """
        self.replace_edge_of_type(node_id, edge_type, new_parent_id)

    def clear_graph(self) -> None:
        """
        Remove all nodes and edges from the graph (thread-safe).

        WARNING: This is a destructive operation!
        """
        with self._lock:
            for node in list(self._graph.get_all_nodes()):
                self._graph.remove_node(node.id)

    # ========== Atomic Batch Updates ==========

    def apply_patch(self, patch: "GraphPatch", validate: bool = True) -> None:
        """
        Apply a GraphPatch atomically.

        All operations in the patch are applied in a single lock acquisition,
        ensuring atomicity. Validation can optionally be performed before applying.

        Args:
            patch: GraphPatch describing the changes
            validate: If True, validate patch before applying (default: True)

        Raises:
            ValueError: If validation fails
        """
        with self._lock:
            # Validate if requested
            if validate:
                errors = patch.validate(self._graph)
                if errors:
                    raise ValueError(
                        f"Patch validation failed: {'; '.join(errors)}"
                    )

            # Apply in order: remove edges, remove nodes, add nodes, add edges, update attributes
            # This order ensures we don't have dangling edges

            # 1. Remove edges
            self._graph.remove_edges(
                [
                    (item.source_id, item.target_id, item.edge_type)
                    for item in patch.edges_to_remove
                ]
            )

            # 2. Remove nodes (also removes incident edges)
            for node_id in patch.nodes_to_remove:
                if self._graph.has_node(node_id):
                    self._graph.remove_node(node_id)

            # 3. Add nodes
            for node in patch.nodes_to_add:
                self._graph.add_node(node)

            # 4. Update nodes
            if patch.nodes_to_update:
                self._graph.update_nodes(patch.nodes_to_update.items())

            # 5. Add edges
            self._graph.add_edges(
                [
                    (item.edge, item.is_structural)
                    for item in patch.edges_to_add
                ]
            )

            # 6. Update node attributes
            for node_id, attributes in patch.node_attribute_updates.items():
                if self._graph.has_node(node_id):
                    node = self._graph.get_node(node_id)
                    if node.attributes is None:
                        node.attributes = {}
                    node.attributes.update(attributes)
                    self._graph.update_node(node_id, node)
