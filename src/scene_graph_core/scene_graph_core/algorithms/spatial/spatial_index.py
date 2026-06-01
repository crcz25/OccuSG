"""
Spatial Index - KD-Tree based spatial indexing for scene graph nodes.

This module provides efficient spatial queries (nearest-neighbor, range queries)
using scipy's cKDTree implementation. It is designed to:
1. Support per-NodeType indexing for type-specific queries
2. Stay synchronized with the SceneGraph on node CRUD operations
3. Handle dynamic updates efficiently (rebuild strategy with batching)
4. Be thread-safe for concurrent access

Design Decisions:
-----------------
- Uses scipy.spatial.cKDTree for O(log n) queries vs O(n) brute-force
- Maintains separate indexes per NodeType for type-filtered queries
- Uses rebuild strategy (not incremental) since cKDTree is immutable
- Batches updates to minimize rebuild frequency
- Thread-safe via RLock for concurrent foreground/background access

Usage:
    from scene_graph_core.algorithms.spatial import SpatialIndex

    index = SpatialIndex()

    # Insert nodes
    index.insert(node)

    # Query nearest
    nearest_id, distance = index.query_nearest(position, node_type=NodeType.OBJECT)

    # Query radius
    ids = index.query_radius(position, radius=5.0, node_type=NodeType.ROOM)

    # Update/remove
    index.update(node)
    index.remove(node_id)
"""

import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

from ...representation import BaseNode, NodeType
from ...representation.geometry import Point


def _is_point_like(position: Any) -> bool:
    """Return whether an object exposes x/y/z point fields."""
    return all(hasattr(position, field_name) for field_name in ("x", "y", "z"))


class SpatialIndex:
    """
    KD-Tree based spatial index for efficient node queries.

    Maintains separate KD-Trees per NodeType for type-filtered queries.
    Uses a deferred rebuild strategy to batch updates and minimize rebuilds.

    Thread-safe for concurrent access from foreground/background nodes.

    Attributes:
        _nodes: Dict mapping node_id -> BaseNode (source of truth)
        _type_nodes: Dict mapping NodeType -> Set[node_id]
        _trees: Dict mapping NodeType -> cKDTree (lazy-built)
        _tree_data: Dict mapping NodeType -> (positions array, node_ids list)
        _dirty_types: Set of NodeTypes that need tree rebuild
        _lock: RLock for thread safety
        _rebuild_threshold: Number of changes before auto-rebuild
    """

    def __init__(self, rebuild_threshold: int = 100, use_3d: bool = True):
        """
        Initialize the spatial index.

        Args:
            rebuild_threshold: Number of accumulated changes before auto-rebuild.
                              Set to 0 for immediate rebuilds (slower but always fresh).
            use_3d: If True, use 3D coordinates (x,y,z). If False, use 2D (x,y).
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is required for spatial indexing. "
                "Install with: pip install scipy>=1.7"
            )

        self._nodes: Dict[int, BaseNode] = {}
        self._type_nodes: Dict[NodeType, Set[int]] = {}
        self._trees: Dict[NodeType, Optional[cKDTree]] = {}
        self._tree_data: Dict[NodeType, Tuple[np.ndarray, List[int]]] = {}
        self._trees_xy: Dict[NodeType, Optional[cKDTree]] = {}
        self._tree_data_xy: Dict[NodeType, Tuple[np.ndarray, List[int]]] = {}
        self._dirty_types: Set[NodeType] = set()
        self._change_count: Dict[NodeType, int] = {}
        self._lock = threading.RLock()
        self._rebuild_threshold = rebuild_threshold
        self._use_3d = use_3d

        # Global tree (all nodes regardless of type)
        self._global_tree: Optional[cKDTree] = None
        self._global_tree_data: Optional[Tuple[np.ndarray, List[int]]] = None
        self._global_tree_xy: Optional[cKDTree] = None
        self._global_tree_data_xy: Optional[Tuple[np.ndarray, List[int]]] = None
        self._global_dirty = False

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    def insert(self, node: BaseNode) -> None:
        """
        Insert a node into the spatial index.

        Args:
            node: Node to insert (must have id and node_type set)

        Raises:
            ValueError: If node.id or node.node_type is None
        """
        if node.id is None:
            raise ValueError("Node must have an id set before insertion")
        if node.node_type is None:
            raise ValueError("Node must have a node_type set before insertion")

        with self._lock:
            # Store node reference
            self._nodes[node.id] = node

            # Add to type index
            if node.node_type not in self._type_nodes:
                self._type_nodes[node.node_type] = set()
            self._type_nodes[node.node_type].add(node.id)

            # Mark tree as dirty
            self._dirty_types.add(node.node_type)
            self._global_dirty = True

            # Track change count for auto-rebuild
            self._change_count[node.node_type] = (
                self._change_count.get(node.node_type, 0) + 1
            )

            # Auto-rebuild if threshold reached
            if self._rebuild_threshold > 0:
                if self._change_count.get(node.node_type, 0) >= self._rebuild_threshold:
                    self._rebuild_tree(node.node_type)

    def update(self, node: BaseNode) -> None:
        """
        Update a node's position in the spatial index.

        If the node's type changed, it will be moved to the correct index.

        Args:
            node: Node with updated position (must have id set)

        Raises:
            ValueError: If node.id is None
            KeyError: If node not found in index
        """
        if node.id is None:
            raise ValueError("Node must have an id set")

        with self._lock:
            if node.id not in self._nodes:
                raise KeyError(f"Node {node.id} not found in spatial index")

            old_node = self._nodes[node.id]
            old_type = old_node.node_type
            new_type = node.node_type

            # Update stored node
            self._nodes[node.id] = node

            # Handle type change
            if old_type != new_type and new_type is not None:
                if old_type is not None and old_type in self._type_nodes:
                    self._type_nodes[old_type].discard(node.id)
                    self._dirty_types.add(old_type)

                if new_type not in self._type_nodes:
                    self._type_nodes[new_type] = set()
                self._type_nodes[new_type].add(node.id)
                self._dirty_types.add(new_type)
            elif new_type is not None:
                # Same type, just mark as dirty
                self._dirty_types.add(new_type)

            self._global_dirty = True

            # Track change count
            if new_type is not None:
                self._change_count[new_type] = self._change_count.get(new_type, 0) + 1

                # Auto-rebuild if threshold reached
                if self._rebuild_threshold > 0:
                    if self._change_count.get(new_type, 0) >= self._rebuild_threshold:
                        self._rebuild_tree(new_type)

    def remove(self, node_id: int) -> None:
        """
        Remove a node from the spatial index.

        Args:
            node_id: ID of node to remove

        Raises:
            KeyError: If node not found
        """
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node {node_id} not found in spatial index")

            node = self._nodes[node_id]
            node_type = node.node_type

            # Remove from storage
            del self._nodes[node_id]

            # Remove from type index
            if node_type in self._type_nodes:
                self._type_nodes[node_type].discard(node_id)
                self._dirty_types.add(node_type)

            self._global_dirty = True

            # Track change count
            if node_type:
                self._change_count[node_type] = self._change_count.get(node_type, 0) + 1

    def contains(self, node_id: int) -> bool:
        """Check if a node exists in the index."""
        with self._lock:
            return node_id in self._nodes

    def clear(self) -> None:
        """Clear all nodes from the index."""
        with self._lock:
            self._nodes.clear()
            self._type_nodes.clear()
            self._trees.clear()
            self._tree_data.clear()
            self._trees_xy.clear()
            self._tree_data_xy.clear()
            self._dirty_types.clear()
            self._change_count.clear()
            self._global_tree = None
            self._global_tree_data = None
            self._global_tree_xy = None
            self._global_tree_data_xy = None
            self._global_dirty = False

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query_nearest(
        self,
        position: Union[Point, Tuple[float, float, float], np.ndarray],
        node_type: Optional[NodeType] = None,
        k: int = 1,
        max_distance: Optional[float] = None,
    ) -> Union[
        Optional[Tuple[int, float]],  # k=1: (node_id, distance) or None
        List[Tuple[int, float]],  # k>1: [(node_id, distance), ...]
    ]:
        """
        Find the k nearest nodes to a position.

        Args:
            position: Query position (Point, tuple, or numpy array)
            node_type: If specified, only search nodes of this type
            k: Number of nearest neighbors to return (default: 1)
            max_distance: Maximum distance to search (optional)

        Returns:
            If k=1: Tuple of (node_id, distance) or None if no nodes found
            If k>1: List of (node_id, distance) tuples, sorted by distance

        Performance:
            O(log n) for k=1, O(k log n) for k>1 (vs O(n) brute-force)
        """
        coords = self._normalize_position(position)

        with self._lock:
            # Get appropriate tree
            if node_type is not None:
                self._ensure_tree(node_type)
                tree = self._trees.get(node_type)
                tree_data = self._tree_data.get(node_type)
            else:
                self._ensure_global_tree()
                tree = self._global_tree
                tree_data = self._global_tree_data

            if tree is None or tree_data is None:
                return None if k == 1 else []

            positions, node_ids = tree_data
            if len(node_ids) == 0:
                return None if k == 1 else []

            # Adjust k if we have fewer nodes
            actual_k = min(k, len(node_ids))

            # Query tree
            if max_distance is not None:
                distances, indices = tree.query(
                    coords, k=actual_k, distance_upper_bound=max_distance
                )
            else:
                distances, indices = tree.query(coords, k=actual_k)

            # Handle single result
            if actual_k == 1:
                if np.isinf(distances):
                    return None
                return (node_ids[indices], float(distances))

            # Handle multiple results
            results = []
            # Ensure distances and indices are arrays for iteration
            if actual_k == 1:
                distances = [distances]
                indices = [indices]

            for dist, idx in zip(distances, indices):
                if not np.isinf(dist) and idx < len(node_ids):
                    results.append((node_ids[idx], float(dist)))

            return results

    def query_radius(
        self,
        position: Union[Point, Tuple[float, float, float], np.ndarray],
        radius: float,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find all nodes within a radius of a position.

        Args:
            position: Query position (Point, tuple, or numpy array)
            radius: Search radius in meters
            node_type: If specified, only search nodes of this type

        Returns:
            List of (node_id, distance) tuples for all nodes within radius,
            sorted by distance (closest first)

        Performance:
            O(k + log n) where k is number of results (vs O(n) brute-force)
        """
        coords = self._normalize_position(position)

        with self._lock:
            # Get appropriate tree
            if node_type is not None:
                self._ensure_tree(node_type)
                tree = self._trees.get(node_type)
                tree_data = self._tree_data.get(node_type)
            else:
                self._ensure_global_tree()
                tree = self._global_tree
                tree_data = self._global_tree_data

            if tree is None or tree_data is None:
                return []

            positions, node_ids = tree_data
            if len(node_ids) == 0:
                return []

            # Query tree for all points within radius
            indices = tree.query_ball_point(coords, radius)

            # Calculate distances and create results
            results = []
            for idx in indices:
                node_id = node_ids[idx]
                node_pos = positions[idx]
                dist = np.linalg.norm(coords - node_pos)
                results.append((node_id, float(dist)))

            # Sort by distance
            results.sort(key=lambda x: x[1])
            return results

    def query_nearest_xy(
        self,
        position: Union[Point, Tuple[float, float, float], np.ndarray],
        node_type: Optional[NodeType] = None,
        k: int = 1,
        max_distance: Optional[float] = None,
    ) -> Union[Optional[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find the nearest nodes using XY distance only."""
        coords = self._normalize_position_xy(position)

        with self._lock:
            if node_type is not None:
                self._ensure_tree(node_type)
                tree = self._trees_xy.get(node_type)
                tree_data = self._tree_data_xy.get(node_type)
            else:
                self._ensure_global_tree()
                tree = self._global_tree_xy
                tree_data = self._global_tree_data_xy

            return self._query_nearest_from_tree(
                coords,
                tree,
                tree_data,
                k=k,
                max_distance=max_distance,
            )

    def query_radius_xy(
        self,
        position: Union[Point, Tuple[float, float, float], np.ndarray],
        radius: float,
        node_type: Optional[NodeType] = None,
    ) -> List[Tuple[int, float]]:
        """Find all nodes within an XY radius."""
        coords = self._normalize_position_xy(position)

        with self._lock:
            if node_type is not None:
                self._ensure_tree(node_type)
                tree = self._trees_xy.get(node_type)
                tree_data = self._tree_data_xy.get(node_type)
            else:
                self._ensure_global_tree()
                tree = self._global_tree_xy
                tree_data = self._global_tree_data_xy

            return self._query_radius_from_tree(coords, radius, tree, tree_data)

    def query_k_nearest_with_filter(
        self,
        position: Union[Point, Tuple[float, float, float], np.ndarray],
        k: int,
        filter_fn,
        node_type: Optional[NodeType] = None,
        search_expansion: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Find k nearest nodes that pass a filter function.

        Useful for queries like "find nearest 3 ROOM nodes with >5 objects".

        Args:
            position: Query position
            k: Number of results to return
            filter_fn: Function(BaseNode) -> bool to filter candidates
            node_type: If specified, only search nodes of this type
            search_expansion: Multiplier for initial search (to find enough matches)

        Returns:
            List of (node_id, distance) tuples for filtered nodes
        """
        coords = self._normalize_position(position)

        with self._lock:
            # Get appropriate tree
            if node_type is not None:
                self._ensure_tree(node_type)
                tree = self._trees.get(node_type)
                tree_data = self._tree_data.get(node_type)
            else:
                self._ensure_global_tree()
                tree = self._global_tree
                tree_data = self._global_tree_data

            if tree is None or tree_data is None:
                return []

            positions, node_ids = tree_data
            if len(node_ids) == 0:
                return []

            results = []
            search_k = min(k * search_expansion, len(node_ids))

            while len(results) < k and search_k <= len(node_ids):
                # Query more candidates
                distances, indices = tree.query(coords, k=search_k)

                # Ensure arrays for iteration
                if search_k == 1:
                    distances = [distances]
                    indices = [indices]

                # Filter candidates
                results = []
                for dist, idx in zip(distances, indices):
                    if np.isinf(dist) or idx >= len(node_ids):
                        continue
                    node_id = node_ids[idx]
                    node = self._nodes.get(node_id)
                    if node and filter_fn(node):
                        results.append((node_id, float(dist)))
                        if len(results) >= k:
                            break

                # Expand search if not enough results
                if len(results) < k:
                    search_k = min(search_k * 2, len(node_ids))
                    if search_k >= len(node_ids):
                        break  # Can't expand further

            return results[:k]

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def bulk_insert(self, nodes: List[BaseNode]) -> None:
        """
        Insert multiple nodes efficiently (single rebuild).

        More efficient than calling insert() multiple times.

        Args:
            nodes: List of nodes to insert
        """
        with self._lock:
            affected_types = set()

            for node in nodes:
                if node.id is None or node.node_type is None:
                    continue

                self._nodes[node.id] = node

                if node.node_type not in self._type_nodes:
                    self._type_nodes[node.node_type] = set()
                self._type_nodes[node.node_type].add(node.id)

                affected_types.add(node.node_type)

            # Mark affected types as dirty
            self._dirty_types.update(affected_types)
            self._global_dirty = True

            # Rebuild affected trees
            for node_type in affected_types:
                self._rebuild_tree(node_type)
            self._rebuild_global_tree()

    def rebuild_all(self) -> None:
        """Force rebuild of all trees. Call after batch operations."""
        with self._lock:
            for node_type in self._type_nodes.keys():
                self._rebuild_tree(node_type)
            self._rebuild_global_tree()
            self._dirty_types.clear()
            self._change_count.clear()

    # =========================================================================
    # Statistics and Debugging
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get index statistics for debugging/monitoring."""
        with self._lock:
            return {
                "total_nodes": len(self._nodes),
                "nodes_by_type": {
                    t.value: len(ids) for t, ids in self._type_nodes.items()
                },
                "dirty_types": [t.value for t in self._dirty_types],
                "global_dirty": self._global_dirty,
                "change_counts": {t.value: c for t, c in self._change_count.items()},
                "rebuild_threshold": self._rebuild_threshold,
                "use_3d": self._use_3d,
            }

    def __len__(self) -> int:
        """Return total number of indexed nodes."""
        with self._lock:
            return len(self._nodes)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SpatialIndex(nodes={stats['total_nodes']}, "
            f"types={list(stats['nodes_by_type'].keys())})"
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _normalize_position(
        self, position: Union[Point, Tuple[float, float, float], np.ndarray]
    ) -> np.ndarray:
        """Convert position to numpy array."""
        if _is_point_like(position):
            if self._use_3d:
                return np.array([position.x, position.y, position.z])
            return np.array([position.x, position.y])
        elif isinstance(position, (tuple, list)):
            if self._use_3d:
                return np.array(position[:3])
            return np.array(position[:2])
        elif isinstance(position, np.ndarray):
            if self._use_3d:
                return position[:3].copy()
            return position[:2].copy()
        else:
            raise TypeError(f"Unsupported position type: {type(position)}")

    def _normalize_position_xy(
        self, position: Union[Point, Tuple[float, float, float], np.ndarray]
    ) -> np.ndarray:
        """Convert position to a 2D XY numpy array."""
        if _is_point_like(position):
            return np.array([position.x, position.y])
        if isinstance(position, (tuple, list)):
            return np.array(position[:2])
        if isinstance(position, np.ndarray):
            return position[:2].copy()
        raise TypeError(f"Unsupported position type: {type(position)}")

    def _get_node_position(self, node: BaseNode) -> np.ndarray:
        """Extract position from node."""
        if self._use_3d:
            return np.array(
                [
                    node.pose.position.x,
                    node.pose.position.y,
                    node.pose.position.z,
                ]
            )
        return np.array(
            [
                node.pose.position.x,
                node.pose.position.y,
            ]
        )

    def _get_node_position_xy(self, node: BaseNode) -> np.ndarray:
        """Extract XY position from node."""
        return np.array([node.pose.position.x, node.pose.position.y])

    def _ensure_tree(self, node_type: NodeType) -> None:
        """Ensure tree for node_type is up-to-date."""
        if node_type in self._dirty_types or node_type not in self._trees:
            self._rebuild_tree(node_type)

    def _ensure_global_tree(self) -> None:
        """Ensure global tree is up-to-date."""
        if self._global_dirty or self._global_tree is None:
            self._rebuild_global_tree()

    def _rebuild_tree(self, node_type: NodeType) -> None:
        """Rebuild KD-tree for a specific node type."""
        node_ids = list(self._type_nodes.get(node_type, set()))

        if not node_ids:
            self._trees[node_type] = None
            self._tree_data[node_type] = (np.array([]), [])
            self._trees_xy[node_type] = None
            self._tree_data_xy[node_type] = (np.array([]), [])
            self._dirty_types.discard(node_type)
            self._change_count[node_type] = 0
            return

        # Build position array
        positions = []
        positions_xy = []
        valid_ids = []
        for node_id in node_ids:
            node = self._nodes.get(node_id)
            if node:
                positions.append(self._get_node_position(node))
                positions_xy.append(self._get_node_position_xy(node))
                valid_ids.append(node_id)

        if not positions:
            self._trees[node_type] = None
            self._tree_data[node_type] = (np.array([]), [])
            self._trees_xy[node_type] = None
            self._tree_data_xy[node_type] = (np.array([]), [])
            self._dirty_types.discard(node_type)
            self._change_count[node_type] = 0
            return

        positions_array = np.array(positions)
        positions_array_xy = np.array(positions_xy)

        # Build tree
        self._trees[node_type] = cKDTree(positions_array)
        self._tree_data[node_type] = (positions_array, valid_ids)
        self._trees_xy[node_type] = cKDTree(positions_array_xy)
        self._tree_data_xy[node_type] = (positions_array_xy, valid_ids)
        self._dirty_types.discard(node_type)
        self._change_count[node_type] = 0

    def _rebuild_global_tree(self) -> None:
        """Rebuild global KD-tree (all nodes)."""
        if not self._nodes:
            self._global_tree = None
            self._global_tree_data = None
            self._global_tree_xy = None
            self._global_tree_data_xy = None
            self._global_dirty = False
            return

        positions = []
        positions_xy = []
        node_ids = []

        for node_id, node in self._nodes.items():
            positions.append(self._get_node_position(node))
            positions_xy.append(self._get_node_position_xy(node))
            node_ids.append(node_id)

        positions_array = np.array(positions)
        positions_array_xy = np.array(positions_xy)

        self._global_tree = cKDTree(positions_array)
        self._global_tree_data = (positions_array, node_ids)
        self._global_tree_xy = cKDTree(positions_array_xy)
        self._global_tree_data_xy = (positions_array_xy, node_ids)
        self._global_dirty = False

    def _query_nearest_from_tree(
        self,
        coords: np.ndarray,
        tree: Optional[cKDTree],
        tree_data: Optional[Tuple[np.ndarray, List[int]]],
        *,
        k: int,
        max_distance: Optional[float],
    ) -> Union[Optional[Tuple[int, float]], List[Tuple[int, float]]]:
        """Shared nearest-neighbor query helper."""
        if tree is None or tree_data is None:
            return None if k == 1 else []

        _, node_ids = tree_data
        if len(node_ids) == 0:
            return None if k == 1 else []

        actual_k = min(k, len(node_ids))
        if max_distance is not None:
            distances, indices = tree.query(
                coords, k=actual_k, distance_upper_bound=max_distance
            )
        else:
            distances, indices = tree.query(coords, k=actual_k)

        if actual_k == 1:
            if np.isinf(distances):
                return None
            return (node_ids[indices], float(distances))

        results = []
        if actual_k == 1:
            distances = [distances]
            indices = [indices]

        for dist, idx in zip(distances, indices):
            if not np.isinf(dist) and idx < len(node_ids):
                results.append((node_ids[idx], float(dist)))
        return results

    def _query_radius_from_tree(
        self,
        coords: np.ndarray,
        radius: float,
        tree: Optional[cKDTree],
        tree_data: Optional[Tuple[np.ndarray, List[int]]],
    ) -> List[Tuple[int, float]]:
        """Shared radius query helper."""
        if tree is None or tree_data is None:
            return []

        positions, node_ids = tree_data
        if len(node_ids) == 0:
            return []

        indices = tree.query_ball_point(coords, radius)
        results = []
        for idx in indices:
            node_id = node_ids[idx]
            node_pos = positions[idx]
            dist = np.linalg.norm(coords - node_pos)
            results.append((node_id, float(dist)))
        results.sort(key=lambda x: x[1])
        return results


# Convenience function for creating index with default settings
def create_spatial_index(
    rebuild_threshold: int = 100,
    use_3d: bool = True,
) -> SpatialIndex:
    """
    Factory function to create a SpatialIndex instance.

    Args:
        rebuild_threshold: Changes before auto-rebuild (0 for immediate)
        use_3d: Use 3D coordinates (True) or 2D (False)

    Returns:
        Configured SpatialIndex instance
    """
    return SpatialIndex(rebuild_threshold=rebuild_threshold, use_3d=use_3d)
