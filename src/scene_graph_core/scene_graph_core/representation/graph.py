"""
SceneGraph Storage Layer

Pure storage wrapper around networkx.DiGraph with no business logic.
Provides basic CRUD operations for nodes and edges.

Type-Scoped ID System:
---------------------
Each NodeType has its own ID range to ensure:
1. Sequential IDs within each type (0, 1, 2, 3...)
2. Globally unique IDs for edge integrity
3. Easy debugging (ID range reveals node type)

ID Ranges:
- AGENT: 0 - 999,999
- DYNAMIC_OBJECT: 1,000,000 - 1,999,999
- OBJECT: 2,000,000 - 2,999,999
- NAVIGATION: 3,000,000 - 3,999,999
- ROOM: 4,000,000 - 4,999,999

Spatial Indexing:
----------------
Maintains an optional SpatialIndex (KD-Tree) for efficient spatial queries.
The index is automatically synchronized on node CRUD operations.
"""

import threading
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from .edge import Edge, EdgeType
from .node import NODE_TYPE_ID_OFFSETS, BaseNode, NodeType

# Lazy import to avoid circular dependencies
if TYPE_CHECKING:
    from ..algorithms.spatial import SpatialIndex


class SceneGraph:
    """
    Pure storage layer for the scene graph with optional spatial indexing.

    Wraps networkx.DiGraph and provides type-safe access to nodes and edges.
    No business logic - just storage and retrieval.

    Spatial Indexing:
        When enable_spatial_index=True, maintains a KD-Tree based spatial index
        that is automatically synchronized on node CRUD operations. This enables
        O(log n) spatial queries instead of O(n) brute-force searches.

    Attributes:
        _graph: NetworkX directed graph for storage
        _type_index: Index mapping NodeType to set of node IDs for fast queries
        _type_counters: Type-scoped counters for node IDs (one counter per NodeType)
        edge_counter: Auto-incrementing counter for edge IDs
        _spatial_index: Optional KD-Tree index for spatial queries
    """

    def __init__(self, enable_spatial_index: bool = True):
        """
        Initialize an empty scene graph.

        Args:
            enable_spatial_index: If True, maintain a KD-Tree spatial index for
                                 O(log n) spatial queries. Default is True.
        """
        self._graph = nx.DiGraph()
        self._type_index: Dict[NodeType, Set[int]] = {}
        self._edge_type_index: Dict[EdgeType, Set[Tuple[int, int]]] = {}
        self._type_counters: Dict[NodeType, int] = {}  # Type-scoped ID counters
        self.edge_counter = 0
        self._lock = threading.RLock()  # Reentrant lock for thread safety

        # Spatial indexing (optional, enabled by default)
        self._enable_spatial_index = enable_spatial_index
        self._spatial_index: Optional["SpatialIndex"] = None

        if enable_spatial_index:
            self._init_spatial_index()

    # ========== Spatial Index Methods ==========

    def _init_spatial_index(self) -> None:
        """Initialize the spatial index (lazy import to avoid circular deps)."""
        try:
            from ..algorithms.spatial import SpatialIndex

            # Use rebuild_threshold=50 for balance between freshness and performance
            self._spatial_index = SpatialIndex(rebuild_threshold=50, use_3d=True)
        except ImportError as e:
            import warnings

            warnings.warn(
                f"Could not initialize spatial index: {e}. "
                "Spatial queries will fall back to O(n) brute-force."
            )
            self._enable_spatial_index = False
            self._spatial_index = None

    @property
    def spatial_index(self) -> Optional["SpatialIndex"]:
        """
        Access the spatial index for efficient spatial queries.

        Returns:
            SpatialIndex instance if enabled, None otherwise.

        Usage:
            # Get nearest room to position
            if graph.spatial_index:
                result = graph.spatial_index.query_nearest(position, NodeType.ROOM)
        """
        return self._spatial_index

    def _sync_spatial_insert(self, node: BaseNode) -> None:
        """Sync spatial index on node insert."""
        if self._spatial_index is not None and node.id is not None:
            try:
                self._spatial_index.insert(node)
            except Exception:
                pass  # Don't fail main operation if index update fails

    def _sync_spatial_update(self, node: BaseNode) -> None:
        """Sync spatial index on node update."""
        if self._spatial_index is not None and node.id is not None:
            try:
                if self._spatial_index.contains(node.id):
                    self._spatial_index.update(node)
                else:
                    self._spatial_index.insert(node)
            except Exception:
                pass  # Don't fail main operation if index update fails

    def _sync_spatial_remove(self, node_id: int) -> None:
        """Sync spatial index on node removal."""
        if self._spatial_index is not None:
            try:
                if self._spatial_index.contains(node_id):
                    self._spatial_index.remove(node_id)
            except Exception:
                pass  # Don't fail main operation if index update fails

    def _index_edge(self, edge: Edge) -> None:
        """Insert edge into fast type index."""
        self._edge_type_index.setdefault(edge.type, set()).add(
            (edge.source_id, edge.target_id)
        )

    def _deindex_edge(self, edge: Edge) -> None:
        """Remove edge from fast type index."""
        keyed_edges = self._edge_type_index.get(edge.type)
        if keyed_edges is None:
            return
        keyed_edges.discard((edge.source_id, edge.target_id))
        if not keyed_edges:
            self._edge_type_index.pop(edge.type, None)

    # ========== Node Operations ==========

    def has_node(self, node_id: int) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._graph

    def add_node(self, node: BaseNode) -> int:
        """
        Add a node to the scene graph with type-scoped ID generation.

        Uses type-based ID ranges to ensure:
        1. Sequential numbering within each type (0, 1, 2, 3...)
        2. Globally unique IDs (via type-specific offsets)
        3. Edge integrity (edges can safely reference nodes by ID)

        Examples:
            - First Room: ID = 4,000,000 (4M + 0)
            - Second Room: ID = 4,000,001 (4M + 1)
            - First Object: ID = 2,000,000 (2M + 0)
            - Second Object: ID = 2,000,001 (2M + 1)

        Args:
            node: Node to add

        Returns:
            The assigned globally-unique node ID

        Raises:
            TypeError: If node is not a BaseNode instance
            ValueError: If node type is None or ID already exists
        """
        if not isinstance(node, BaseNode):
            raise TypeError("Expected a BaseNode instance.")

        if node.node_type is None:
            raise ValueError("Node must have a node_type set before adding to graph.")

        if node.node_type not in NODE_TYPE_ID_OFFSETS:
            raise ValueError(f"Unknown node type: {node.node_type}")

        with self._lock:
            # Get the ID offset for this node type
            id_offset = NODE_TYPE_ID_OFFSETS[node.node_type]

            # Assign type-scoped sequential ID if not set
            if node.id is None:
                # Get or initialize counter for this node type
                if node.node_type not in self._type_counters:
                    self._type_counters[node.node_type] = 0

                # Sequential counter (0, 1, 2, 3...)
                type_scoped_id = self._type_counters[node.node_type]
                self._type_counters[node.node_type] += 1

                # Create globally unique ID by adding offset
                global_id = id_offset + type_scoped_id
                node.id = global_id
            else:
                # Node already has an ID - validate it's in the correct range
                global_id = node.id
                if global_id < id_offset or global_id >= id_offset + 1_000_000:
                    raise ValueError(
                        f"Node ID {global_id} is outside valid range for type {node.node_type} "
                        f"(expected {id_offset} to {id_offset + 999_999})"
                    )

                # Update counter to track highest ID used
                type_scoped_id = global_id - id_offset
                current_counter = self._type_counters.get(node.node_type, 0)
                self._type_counters[node.node_type] = max(
                    current_counter, type_scoped_id + 1
                )

            # Check for duplicate IDs
            if global_id in self._graph:
                raise ValueError(
                    f"Node with ID {global_id} already exists in the graph."
                )

            # Add to graph and type index
            self._graph.add_node(global_id, node=node)
            self._type_index.setdefault(node.node_type, set()).add(global_id)

            # Sync spatial index
            self._sync_spatial_insert(node)

            return global_id

    def get_node(self, node_id: int) -> BaseNode:
        """
        Retrieve a node by its ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            The node object

        Raises:
            KeyError: If node does not exist
        """
        if node_id not in self._graph:
            raise KeyError(f"Node with ID {node_id} does not exist.")
        return self._graph.nodes[node_id]["node"]

    def get_all_nodes(self) -> List[BaseNode]:
        """Get all nodes in the scene graph."""
        with self._lock:
            return [data["node"] for _, data in self._graph.nodes(data=True)]

    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        """
        Get all nodes of a specific type.

        Uses type index for O(k) performance where k is number of nodes of that type.

        Args:
            node_type: Type of nodes to retrieve

        Returns:
            List of nodes of the specified type
        """
        with self._lock:
            ids = self._type_index.get(node_type, set())
            return [self._graph.nodes[nid]["node"] for nid in ids]

    def get_node_ids_by_type(self, node_type: NodeType) -> List[int]:
        """Get node IDs for one type without materializing node objects."""
        with self._lock:
            return list(self._type_index.get(node_type, ()))

    def update_nodes(self, node_updates: Iterable[Tuple[int, BaseNode]]) -> None:
        """Update multiple nodes in one lock acquisition."""
        with self._lock:
            for node_id, new_node in node_updates:
                if node_id not in self._graph:
                    raise KeyError(f"Node with ID {node_id} does not exist.")
                if not isinstance(new_node, BaseNode):
                    raise TypeError("Expected a BaseNode instance.")

                old_type = self._graph.nodes[node_id]["node"].node_type
                new_type = new_node.node_type
                if old_type != new_type and new_type is not None:
                    if old_type is not None:
                        self._type_index[old_type].discard(node_id)
                        if not self._type_index[old_type]:
                            del self._type_index[old_type]
                    self._type_index.setdefault(new_type, set()).add(node_id)

                new_node.id = node_id
                self._graph.nodes[node_id]["node"] = new_node
                self._sync_spatial_update(new_node)

    def update_node(self, node_id: int, new_node: BaseNode) -> None:
        """
        Update an existing node with a new Node instance.

        Args:
            node_id: ID of the node to update
            new_node: New node data

        Raises:
            KeyError: If node does not exist
            TypeError: If new_node is not a BaseNode instance
        """
        if node_id not in self._graph:
            raise KeyError(f"Node with ID {node_id} does not exist.")

        if not isinstance(new_node, BaseNode):
            raise TypeError("Expected a BaseNode instance.")

        with self._lock:
            # Update type index if type changed
            old_type = self._graph.nodes[node_id]["node"].node_type
            new_type = new_node.node_type
            if old_type != new_type and new_type is not None:
                if old_type is not None:
                    self._type_index[old_type].discard(node_id)
                    if not self._type_index[old_type]:
                        del self._type_index[old_type]
                self._type_index.setdefault(new_type, set()).add(node_id)

            # Ensure ID remains the same
            new_node.id = node_id
            self._graph.nodes[node_id]["node"] = new_node

            # Sync spatial index
            self._sync_spatial_update(new_node)

    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the scene graph.

        Args:
            node_id: ID of the node to remove

        Raises:
            KeyError: If node does not exist
        """
        if node_id not in self._graph:
            raise KeyError(f"Node with ID {node_id} does not exist.")

        with self._lock:
            # Track incident edges so edge-type index stays consistent.
            incident_edges = [
                self._graph.edges[source_id, target_id]["edge"]
                for source_id, target_id in self._graph.in_edges(node_id)
            ] + [
                self._graph.edges[source_id, target_id]["edge"]
                for source_id, target_id in self._graph.out_edges(node_id)
            ]
            for edge in incident_edges:
                self._deindex_edge(edge)

            # Update type index
            node_type = self._graph.nodes[node_id]["node"].node_type
            if node_type is not None and node_type in self._type_index:
                self._type_index[node_type].discard(node_id)
                if not self._type_index[node_type]:
                    del self._type_index[node_type]

            # Sync spatial index (must be done before removing from graph)
            self._sync_spatial_remove(node_id)

            # Remove from graph (also removes connected edges)
            self._graph.remove_node(node_id)

    # ========== Edge Operations ==========

    def has_edge(
        self, source_id: int, target_id: int, edge_type: Optional["EdgeType"] = None
    ) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Optional EdgeType to check for a specific edge type.
                      If None, checks for any edge between the nodes.

        Returns:
            True if an edge exists (optionally of the specified type), False otherwise
        """
        if not self._graph.has_edge(source_id, target_id):
            return False

        if edge_type is None:
            return True

        # Check if the edge has the specified type
        edge = self._graph.edges[source_id, target_id]["edge"]
        return edge.type == edge_type

    def add_edge(self, edge: Edge, is_structural: bool = True) -> int:
        """
        Add an edge to the scene graph.

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
        if not isinstance(edge, Edge):
            raise TypeError("Expected an Edge instance.")

        with self._lock:
            # Validate nodes exist
            if not self.has_node(edge.source_id):
                raise ValueError(f"Source node {edge.source_id} does not exist.")
            if not self.has_node(edge.target_id):
                raise ValueError(f"Target node {edge.target_id} does not exist.")

            # No self-loops
            if edge.source_id == edge.target_id:
                raise ValueError(
                    f"Self-loop edges are not allowed: {edge.source_id} -> {edge.target_id}"
                )

            # Duplicate-edge check: if an edge with the same (source, target, type)
            # already exists, return its existing ID silently rather than adding a
            # second parallel edge.  NetworkX overwrites the edge data on duplicate
            # add_edge calls, so without this check the old edge ID is silently
            # lost and the edge_counter drifts out of sync.
            existing_edge_to_replace: Optional[Edge] = None
            if self._graph.has_edge(edge.source_id, edge.target_id):
                existing_edge: Edge = self._graph.edges[edge.source_id, edge.target_id][
                    "edge"
                ]
                if existing_edge.type == edge.type:
                    # Identical edge already present — nothing to do.
                    return (
                        existing_edge.id
                        if existing_edge.id is not None
                        else self.edge_counter
                    )
                # Edge exists for same endpoint pair but with different type.
                # NetworkX will overwrite it if this add succeeds.
                existing_edge_to_replace = existing_edge

            # Check for cycles ONLY for structural edges
            # Relational edges (like NAVIGABLE_PATH) can form cycles
            if is_structural and nx.has_path(
                self._graph, edge.target_id, edge.source_id
            ):
                raise ValueError(
                    f"Adding this structural edge would create a cycle: {edge.source_id} -> {edge.target_id}"
                )

            # Assign ID and add to graph
            if edge.id is None:
                edge.id = self.edge_counter
                self.edge_counter += 1
            else:
                self.edge_counter = max(self.edge_counter, edge.id + 1)

            edge.is_structural = is_structural
            if existing_edge_to_replace is not None:
                self._deindex_edge(existing_edge_to_replace)
            self._graph.add_edge(edge.source_id, edge.target_id, edge=edge)
            self._index_edge(edge)

            return edge.id

    def get_edge(self, source_id: int, target_id: int) -> Edge:
        """
        Retrieve an edge by its source and target node IDs.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node

        Returns:
            The edge object

        Raises:
            KeyError: If edge does not exist
        """
        if not self.has_edge(source_id, target_id):
            raise KeyError(f"Edge from {source_id} to {target_id} does not exist.")
        return self._graph.edges[source_id, target_id]["edge"]

    def get_all_edges(self) -> List[Edge]:
        """Get all edges in the scene graph."""
        # Make a copy to avoid RuntimeError if edges are modified during iteration
        # Use lock to prevent modifications during the snapshot operation
        with self._lock:
            edges_snapshot = list(self._graph.edges(data=True))
            return [data["edge"] for _, _, data in edges_snapshot]

    def get_edges_by_type(self, edge_type: EdgeType) -> List[Edge]:
        """Get all edges of a specific type."""
        with self._lock:
            keyed_edges = list(self._edge_type_index.get(edge_type, set()))
            edges: List[Edge] = []
            for source_id, target_id in keyed_edges:
                if not self._graph.has_edge(source_id, target_id):
                    continue
                edge = self._graph.edges[source_id, target_id]["edge"]
                if edge.type == edge_type:
                    edges.append(edge)
            return edges

    def get_edge_pairs_by_type(self, edge_type: EdgeType) -> Set[Tuple[int, int]]:
        """Get endpoint pairs for one edge type."""
        with self._lock:
            return set(self._edge_type_index.get(edge_type, set()))

    def add_edges(
        self, edge_items: Iterable[Tuple[Edge, bool] | Edge], is_structural: bool = True
    ) -> List[int]:
        """Add multiple edges in one lock acquisition."""
        edge_ids: List[int] = []
        with self._lock:
            for item in edge_items:
                if isinstance(item, tuple):
                    edge, edge_is_structural = item
                else:
                    edge = item
                    edge_is_structural = is_structural
                edge_ids.append(self.add_edge(edge, edge_is_structural))
        return edge_ids

    def update_edge(self, source_id: int, target_id: int, new_edge: Edge) -> None:
        """
        Update an existing edge with a new Edge instance.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            new_edge: New edge data

        Raises:
            KeyError: If edge does not exist
            TypeError: If new_edge is not an Edge instance
        """
        if not self.has_edge(source_id, target_id):
            raise KeyError(f"Edge from {source_id} to {target_id} does not exist.")

        if not isinstance(new_edge, Edge):
            raise TypeError("Expected an Edge instance.")

        with self._lock:
            old_edge = self._graph.edges[source_id, target_id]["edge"]
            if old_edge.type != new_edge.type:
                self._deindex_edge(old_edge)
            self._graph.edges[source_id, target_id]["edge"] = new_edge
            self._index_edge(new_edge)

    def remove_edge(self, source_id: int, target_id: int) -> None:
        """
        Remove an edge from the scene graph.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node

        Raises:
            KeyError: If edge does not exist
        """
        if not self.has_edge(source_id, target_id):
            raise KeyError(f"Edge from {source_id} to {target_id} does not exist.")
        with self._lock:
            edge = self._graph.edges[source_id, target_id]["edge"]
            self._deindex_edge(edge)
            self._graph.remove_edge(source_id, target_id)

    def remove_edges(
        self,
        edge_refs: Iterable[Tuple[int, int, Optional[EdgeType]]],
    ) -> int:
        """Remove multiple edges with optional type filters."""
        removed_count = 0
        with self._lock:
            for source_id, target_id, edge_type in edge_refs:
                if not self.has_edge(source_id, target_id, edge_type=edge_type):
                    continue
                edge = self._graph.edges[source_id, target_id]["edge"]
                if edge_type is not None and edge.type != edge_type:
                    continue
                self._deindex_edge(edge)
                self._graph.remove_edge(source_id, target_id)
                removed_count += 1
        return removed_count

    # ========== Graph Queries ==========

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get IDs of all nodes connected to the given node (successors)."""
        if not self.has_node(node_id):
            raise KeyError(f"Node with ID {node_id} does not exist.")
        return list(self._graph.successors(node_id))

    def get_predecessors(self, node_id: int) -> List[int]:
        """Get IDs of all nodes that connect to the given node."""
        if not self.has_node(node_id):
            raise KeyError(f"Node with ID {node_id} does not exist.")
        return list(self._graph.predecessors(node_id))

    def has_path(self, source_id: int, target_id: int) -> bool:
        """Check if there is a path from source to target."""
        if not self.has_node(source_id) or not self.has_node(target_id):
            return False
        return nx.has_path(self._graph, source_id, target_id)

    def node_count(self) -> int:
        """Get the total number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get the total number of edges in the graph."""
        return self._graph.number_of_edges()

    def get_outgoing_edges(
        self,
        node_id: int,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """
        Get all outgoing edges from a node.

        Args:
            node_id: ID of the source node

        Returns:
            List of Edge objects going out from this node

        Raises:
            KeyError: If node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node with ID {node_id} does not exist.")

        edges = []
        if edge_type is None:
            for _, target_id in self._graph.out_edges(node_id):
                edges.append(self._graph.edges[node_id, target_id]["edge"])
            return edges

        for source_id, target_id in self._edge_type_index.get(edge_type, ()):
            if source_id != node_id or not self._graph.has_edge(source_id, target_id):
                continue
            edge = self._graph.edges[source_id, target_id]["edge"]
            if edge.type == edge_type:
                edges.append(edge)
        return edges

    def get_incoming_edges(
        self,
        node_id: int,
        edge_type: Optional[EdgeType] = None,
    ) -> List[Edge]:
        """
        Get all incoming edges to a node.

        Args:
            node_id: ID of the target node

        Returns:
            List of Edge objects coming into this node

        Raises:
            KeyError: If node does not exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node with ID {node_id} does not exist.")

        edges = []
        if edge_type is None:
            for source_id, _ in self._graph.in_edges(node_id):
                edges.append(self._graph.edges[source_id, node_id]["edge"])
            return edges

        for source_id, target_id in self._edge_type_index.get(edge_type, ()):
            if target_id != node_id or not self._graph.has_edge(source_id, target_id):
                continue
            edge = self._graph.edges[source_id, target_id]["edge"]
            if edge.type == edge_type:
                edges.append(edge)
        return edges

    # ========== Direct NetworkX Access ==========

    @property
    def nx_graph(self) -> nx.DiGraph:
        """
        Direct access to the underlying NetworkX graph.

        WARNING: Use with caution! Direct manipulation bypasses type indexing
        and other internal structures. Prefer using the provided methods.

        Returns:
            The underlying NetworkX DiGraph
        """
        return self._graph
