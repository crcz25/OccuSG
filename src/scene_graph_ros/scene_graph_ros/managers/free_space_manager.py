"""Free-space manager for grid-derived navigation nodes."""

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.time import Time

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import Edge, EdgeType, NavNode, NodeType


class FreeSpaceNodeManager:
    """Creates NAVIGATION nodes from occupancy-grid free-space blocks."""

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        cell_stride_cells: int = 10,
        min_free_cell_count: int = 50,
        z_offset: float = 4.0,
        nearest_link_max_distance_m: float = 1.0,
        navigation_connectivity: int = 8,
        enable_debug_logging: bool = True,
        debug_log_interval: int = 10,
    ):
        self.sg = sg_interface
        self.logger = logger

        self.z_offset = float(z_offset)
        self.nearest_link_max_distance_m = max(0.0, float(nearest_link_max_distance_m))
        self.cell_stride = max(1, int(cell_stride_cells))
        self.min_free_cell_count = max(1, int(min_free_cell_count))
        self.navigation_connectivity = (
            4 if int(navigation_connectivity) == 4 else 8
        )

        self.enable_debug_logging = bool(enable_debug_logging)
        self.debug_log_interval = int(max(1, debug_log_interval))
        self.update_counter = 0

        self.stats = {
            "total_nav_nodes_created": 0,
            "total_nav_nodes_updated": 0,
            "total_nav_nodes_deleted": 0,
            "total_nav_edges_created": 0,
        }

        self.cell_key_resolution: Optional[float] = None
        self.current_grid: Optional[np.ndarray] = None
        self.current_grid_info: Optional[dict] = None

        self.nav_node_cache: Dict[Tuple[int, int], NavNode] = {}
        self.block_to_node_id: Dict[Tuple[int, int], int] = {}
        self.dirty_blocks: Set[Tuple[int, int]] = set()
        self.block_free_cell_counts: Dict[Tuple[int, int], int] = {}
        self.previous_block_free_cell_counts: Dict[Tuple[int, int], int] = {}
        self.nav_edge_pairs: Set[Tuple[int, int]] = set()
        self.block_centroid_cache: Dict[
            Tuple[Tuple[int, int], float, float, float], Tuple[float, float]
        ] = {}
        self.block_bounds_cache: Dict[
            Tuple[Tuple[int, int], float, float, float],
            Tuple[float, float, float, float],
        ] = {}
        self.block_free_cell_cache: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        self._grid_geometry_signature: Optional[Tuple[float, float, float]] = None
        self._logged_geometry_sample_signature: Optional[
            Tuple[Tuple[int, int], float, float, float]
        ] = None

        self.pending_object_ids: Set[int] = set()
        self.object_to_block: Dict[int, Tuple[int, int]] = {}
        self.object_block_index: Dict[Tuple[int, int], Set[int]] = {}
        self.object_nearest_nav: Dict[int, int] = {}
        self.pending_full_relink = False

        self.nearest_link_distance_epsilon = 0.05

        self.logger.debug("FreeSpaceNodeManager initialized:")
        self.logger.debug(f"  - cell_stride: {self.cell_stride} cells")
        self.logger.debug(f"  - min_free_cell_count: {self.min_free_cell_count}")
        self.logger.debug(f"  - z_offset: {self.z_offset}m")
        self.logger.debug(
            f"  - nearest_link_max_distance_m: {self.nearest_link_max_distance_m}m"
        )
        self.logger.debug(
            f"  - navigation_connectivity: {self.navigation_connectivity}"
        )
        self.logger.debug(f"  - debug_logging: {self.enable_debug_logging}")

    def _log_update_stats(self, stats: Dict[str, int], extra_info: str = "") -> None:
        if not self.enable_debug_logging:
            return

        self.update_counter += 1
        if self.update_counter % self.debug_log_interval != 0:
            return

        self.logger.debug(f"=== FreeSpaceNodeManager Update #{self.update_counter} ===")
        self.logger.debug(
            f"  Nav nodes: total={stats.get('total_nav_nodes', 0)}, "
            f"new={stats.get('new_nav_nodes', 0)}, "
            f"deleted={stats.get('deleted_nav_nodes', 0)}"
        )
        self.logger.debug(
            f"  Totals: created={self.stats['total_nav_nodes_created']}, "
            f"updated={self.stats['total_nav_nodes_updated']}, "
            f"edges={self.stats['total_nav_edges_created']}"
        )
        if extra_info:
            self.logger.debug(f"  {extra_info}")

    def has_processed_map_snapshot(self) -> bool:
        """Return True once at least one occupancy grid has been processed."""
        return self.cell_key_resolution is not None and self.current_grid_info is not None

    def has_pending_nearest_link_work(self) -> bool:
        """Return True when maintenance has nearest-link work to do."""
        return bool(self.pending_object_ids) or self.pending_full_relink

    def queue_object_ids_for_nearest_link(self, object_ids: Iterable[int]) -> int:
        """Queue object IDs for later nearest-link reconciliation."""
        queued = 0
        for raw_object_id in object_ids:
            try:
                object_id = int(raw_object_id)
            except (TypeError, ValueError):
                continue

            if object_id not in self.pending_object_ids:
                queued += 1
            self.pending_object_ids.add(object_id)

            if not self.has_processed_map_snapshot():
                continue
            obj_node = self.sg.query.get_node(object_id)
            if obj_node is not None and obj_node.node_type == NodeType.OBJECT:
                self._index_object_node(obj_node)

        return queued

    def drain_queued_object_ids(self) -> Set[int]:
        """Get queued object IDs and clear the pending set."""
        object_ids = set(self.pending_object_ids)
        self.pending_object_ids.clear()
        return object_ids

    def rebuild_object_block_index(self) -> int:
        """Rebuild the object-to-block index from the current graph state."""
        self.object_to_block.clear()
        self.object_block_index.clear()

        if not self.has_processed_map_snapshot():
            return 0

        indexed_count = 0
        for obj_node in self.sg.query.find_nodes_by_type(NodeType.OBJECT):
            if self._index_object_node(obj_node) is not None:
                indexed_count += 1
        return indexed_count

    def _get_neighbor_block_offsets(
        self,
        connectivity: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """Return the local block-neighborhood offsets for the requested connectivity."""
        resolved_connectivity = (
            self.navigation_connectivity
            if connectivity is None
            else (4 if int(connectivity) == 4 else 8)
        )
        if resolved_connectivity == 4:
            return (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            )

        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    def _get_canonical_neighbor_block_offsets(
        self,
        connectivity: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """Return a forward-only subset of neighbor offsets for unique undirected pairs."""
        return tuple(
            (dx, dy)
            for dx, dy in self._get_neighbor_block_offsets(connectivity)
            if dx > 0 or (dx == 0 and dy > 0)
        )

    def _neighbor_block_ids(
        self,
        block_id: Tuple[int, int],
        connectivity: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """Return adjacency candidates for one block."""
        block_i, block_j = block_id
        return tuple(
            (block_i + dx, block_j + dy)
            for dx, dy in self._get_neighbor_block_offsets(connectivity)
        )

    def _canonical_neighbor_block_ids(
        self,
        block_id: Tuple[int, int],
        connectivity: Optional[int] = None,
    ) -> Tuple[Tuple[int, int], ...]:
        """Return forward-only adjacency candidates for one block."""
        block_i, block_j = block_id
        return tuple(
            (block_i + dx, block_j + dy)
            for dx, dy in self._get_canonical_neighbor_block_offsets(connectivity)
        )

    def _remove_object_from_block_index(self, object_id: int) -> None:
        """Remove one object from the cached block index."""
        block_id = self.object_to_block.pop(object_id, None)
        if block_id is None:
            return

        block_members = self.object_block_index.get(block_id)
        if block_members is None:
            return

        block_members.discard(object_id)
        if not block_members:
            self.object_block_index.pop(block_id, None)

    def _index_object_node(self, obj_node) -> Optional[Tuple[int, int]]:
        """Track the coarse free-space block that contains the object pose."""
        if (
            obj_node is None
            or obj_node.id is None
            or obj_node.node_type != NodeType.OBJECT
            or not self.has_processed_map_snapshot()
        ):
            return None

        block_id = self._world_to_block_id(
            float(obj_node.pose.position.x),
            float(obj_node.pose.position.y),
        )
        if block_id is None:
            return None

        object_id = int(obj_node.id)
        old_block_id = self.object_to_block.get(object_id)
        if old_block_id == block_id:
            return block_id

        self._remove_object_from_block_index(object_id)
        self.object_to_block[object_id] = block_id
        self.object_block_index.setdefault(block_id, set()).add(object_id)
        return block_id

    def _world_to_block_id(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Convert a world-space XY position into the coarse block key."""
        if self.cell_key_resolution is None or self.cell_key_resolution <= 0.0:
            return None

        block_size_m = float(self.cell_stride) * float(self.cell_key_resolution)
        if block_size_m <= 0.0:
            return None

        return (
            int(math.floor(float(x) / block_size_m)),
            int(math.floor(float(y) / block_size_m)),
        )

    def _grid_cell_to_block_id(
        self,
        grid_cell_x: int,
        grid_cell_y: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[int, int]:
        """Convert one occupancy-grid cell index into the coarse block key."""
        block_size_m = float(self.cell_stride) * float(resolution)
        cell_center_x = float(origin_x) + (float(grid_cell_x) + 0.5) * float(resolution)
        cell_center_y = float(origin_y) + (float(grid_cell_y) + 0.5) * float(resolution)
        return (
            int(math.floor(cell_center_x / block_size_m)),
            int(math.floor(cell_center_y / block_size_m)),
        )

    def _make_block_geometry_cache_key(
        self,
        block_id: Tuple[int, int],
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[Tuple[int, int], float, float, float]:
        """Return the cache key for one block under one grid geometry."""
        return (
            block_id,
            float(resolution),
            float(origin_x),
            float(origin_y),
        )

    def _clear_block_geometry_caches(self) -> None:
        """Drop cached block geometry derived from the previous grid signature."""
        self.block_centroid_cache.clear()
        self.block_bounds_cache.clear()
        self._grid_geometry_signature = None
        self._logged_geometry_sample_signature = None

    def _iter_candidate_block_ids(
        self,
        block_id: Tuple[int, int],
        radius_m: float,
    ):
        """Yield candidate block IDs within the requested metric radius."""
        if block_id is None:
            return

        block_size_m = self.cell_stride * max(self.cell_key_resolution or 0.0, 1e-9)
        if radius_m <= 0.0:
            block_radius = 0
        else:
            block_radius = int(math.ceil(radius_m / block_size_m))

        block_x, block_y = block_id
        for dx in range(-block_radius, block_radius + 1):
            for dy in range(-block_radius, block_radius + 1):
                yield (block_x + dx, block_y + dy)

    def _get_object_ids_near_blocks(
        self,
        block_ids: Iterable[Tuple[int, int]],
        radius_m: float,
    ) -> Set[int]:
        """Return object IDs indexed near the provided free-space blocks."""
        affected_object_ids: Set[int] = set()
        for block_id in block_ids:
            for candidate_block_id in self._iter_candidate_block_ids(block_id, radius_m):
                affected_object_ids.update(
                    self.object_block_index.get(candidate_block_id, ())
                )
        return affected_object_ids

    def process_occupancy_grid_update(
        self,
        free_space_data: OccupancyGrid,
        odom_data: Optional[Odometry],
        frame_id: str = "odom",
    ) -> Dict[str, object]:
        """Create or refresh NAVIGATION nodes from the latest occupancy grid."""
        del frame_id

        if free_space_data is None:
            self.logger.debug("No free space data received yet")
            return {
                "total_nav_nodes": 0,
                "new_nav_nodes": 0,
                "deleted_nav_nodes": 0,
                "created_block_ids": set(),
                "removed_linked_object_ids": set(),
                "full_rescan_required": False,
            }

        width = free_space_data.info.width
        height = free_space_data.info.height
        resolution = free_space_data.info.resolution
        origin_x = free_space_data.info.origin.position.x
        origin_y = free_space_data.info.origin.position.y

        previous_grid_info = (
            dict(self.current_grid_info) if self.current_grid_info is not None else None
        )
        previous_resolution = self.cell_key_resolution

        current_grid = np.array(free_space_data.data, dtype=np.int8).reshape(
            (height, width)
        )
        grid_info = {
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
        }

        self.current_grid = current_grid
        self.current_grid_info = grid_info
        self.block_free_cell_cache.clear()

        if odom_data is not None:
            ros_time = Time.from_msg(odom_data.header.stamp)
        else:
            ros_time = Time.from_msg(free_space_data.header.stamp)
        timestamp = ros_time.nanoseconds / 1e9

        full_rescan_required = False
        resolution_changed = (
            previous_resolution is not None
            and abs(float(previous_resolution) - float(resolution)) > 0.0001
        )
        origin_changed = (
            previous_grid_info is not None
            and (
                abs(float(previous_grid_info["origin_x"]) - float(origin_x)) > 1e-6
                or abs(float(previous_grid_info["origin_y"]) - float(origin_y)) > 1e-6
            )
        )

        if self.cell_key_resolution is None:
            self.cell_key_resolution = resolution
        else:
            self.cell_key_resolution = resolution
        if resolution_changed:
            old_resolution = (
                float(previous_resolution)
                if previous_resolution is not None
                else float(resolution)
            )
            old_origin_x = (
                float(previous_grid_info["origin_x"])
                if previous_grid_info is not None
                else float(origin_x)
            )
            old_origin_y = (
                float(previous_grid_info["origin_y"])
                if previous_grid_info is not None
                else float(origin_y)
            )
            self.logger.debug(
                "[FreeSpaceNodeManager] Grid resolution changed "
                f"resolution={old_resolution:.4f}->{float(resolution):.4f} "
                f"origin=({old_origin_x:.4f}, {old_origin_y:.4f})->"
                f"({float(origin_x):.4f}, {float(origin_y):.4f})"
            )
            self._clear_block_geometry_caches()
            self.object_to_block.clear()
            self.object_block_index.clear()
            self.object_nearest_nav.clear()
            self.previous_block_free_cell_counts.clear()
            self.pending_full_relink = True
            full_rescan_required = True
        elif origin_changed:
            old_origin_x = float(previous_grid_info["origin_x"])
            old_origin_y = float(previous_grid_info["origin_y"])
            self.logger.debug(
                "[FreeSpaceNodeManager] Grid origin changed; rebuilding "
                "world-aligned navigation blocks "
                f"origin=({old_origin_x:.4f}, {old_origin_y:.4f})->"
                f"({float(origin_x):.4f}, {float(origin_y):.4f})"
            )
            self._clear_block_geometry_caches()
            self.object_to_block.clear()
            self.object_block_index.clear()
            self.object_nearest_nav.clear()
            self.previous_block_free_cell_counts.clear()
            self.pending_full_relink = True
            full_rescan_required = True

        previous_counts = (
            {}
            if full_rescan_required
            else dict(self.previous_block_free_cell_counts)
        )

        update_stats = self._process_cell_blocks(
            current_grid,
            grid_info,
            timestamp,
            previous_counts=previous_counts,
            prune_stale_nodes=full_rescan_required,
        )
        created_block_ids = set(update_stats["created_block_ids"])
        removed_linked_object_ids = set(update_stats["removed_linked_object_ids"])
        changed_block_ids = set(update_stats["changed_block_ids"])
        self.previous_block_free_cell_counts = dict(self.block_free_cell_counts)

        if created_block_ids and not full_rescan_required:
            nearby_object_ids = self._get_object_ids_near_blocks(
                created_block_ids,
                self.nearest_link_max_distance_m,
            )
            self.pending_object_ids.update(nearby_object_ids)

        if changed_block_ids and not full_rescan_required:
            self.pending_object_ids.update(
                self._get_object_ids_near_blocks(
                    changed_block_ids,
                    self.nearest_link_max_distance_m,
                )
            )

        if removed_linked_object_ids:
            self.pending_object_ids.update(removed_linked_object_ids)

        result_stats = {
            "total_nav_nodes": len(self.block_to_node_id),
            "new_nav_nodes": int(update_stats["new_nav_nodes"]),
            "deleted_nav_nodes": int(update_stats["deleted_nav_nodes"]),
            "created_block_ids": created_block_ids,
            "changed_block_ids": changed_block_ids,
            "removed_linked_object_ids": removed_linked_object_ids,
            "full_rescan_required": full_rescan_required,
        }

        extra_info = (
            f"Grid: {width}x{height}, origin=({origin_x:.2f},{origin_y:.2f}), "
            f"res={resolution}m"
        )
        self._log_update_stats(result_stats, extra_info)
        return result_stats

    def _process_cell_blocks(
        self,
        grid: np.ndarray,
        grid_info: dict,
        timestamp: float,
        *,
        previous_counts: Optional[Dict[Tuple[int, int], int]] = None,
        prune_stale_nodes: bool = False,
    ) -> Dict[str, object]:
        """Refresh the active set of free-space blocks."""
        new_blocks: Set[Tuple[int, int]] = set()
        removed_linked_object_ids: Set[int] = set()
        previous_counts = previous_counts or {}
        self.block_free_cell_counts = self._compute_block_free_cell_counts(grid, grid_info)
        count_changed_blocks = {
            block_id
            for block_id in set(previous_counts).union(self.block_free_cell_counts)
            if previous_counts.get(block_id) != self.block_free_cell_counts.get(block_id)
        }
        self.dirty_blocks = set(count_changed_blocks)

        existing_blocks = set(self.block_to_node_id.keys())
        blocks_to_refresh = set(existing_blocks).union(self.block_free_cell_counts)
        for block_id in blocks_to_refresh:
            node_id = self._create_or_update_nav_node(block_id, grid_info, timestamp)
            if node_id is not None and block_id not in existing_blocks:
                new_blocks.add(block_id)

        blocks_to_remove = []
        if prune_stale_nodes:
            for block_id in existing_blocks:
                if (
                    block_id not in self.block_free_cell_counts
                    or self.block_free_cell_counts[block_id] < self.min_free_cell_count
                ):
                    blocks_to_remove.append(block_id)

        for block_id in blocks_to_remove:
            node_id = self.block_to_node_id.get(block_id)
            if node_id is None:
                continue

            incoming_nearest_edges = self.sg.query.get_incoming_edges(
                node_id,
                edge_type=EdgeType.NEAREST_FREE_SPACE,
            )
            removed_linked_object_ids.update(
                int(edge.source_id) for edge in incoming_nearest_edges
            )

            try:
                self.sg.update.remove_node(node_id)
                self.stats["total_nav_nodes_deleted"] += 1
            except Exception as exc:
                self.logger.debug(f"Failed to remove nav block {block_id}: {exc}")
            self.block_to_node_id.pop(block_id, None)
            self.nav_node_cache.pop(block_id, None)
            self.nav_edge_pairs = {
                pair
                for pair in self.nav_edge_pairs
                if node_id not in pair
            }

        edge_dirty_blocks = set(existing_blocks).union(self.block_free_cell_counts)
        if blocks_to_remove:
            edge_dirty_blocks.update(blocks_to_remove)
        self._create_navigation_edges(edge_dirty_blocks, blocks_to_remove)

        self.logger.debug(
            f"Navigation block processing: {len(self.block_free_cell_counts)} blocks, "
            f"{len(self.block_to_node_id)} active, {len(new_blocks)} new, "
            f"{len(blocks_to_remove)} removed"
        )

        return {
            "new_nav_nodes": len(new_blocks),
            "deleted_nav_nodes": len(blocks_to_remove),
            "created_block_ids": new_blocks,
            "changed_block_ids": set(count_changed_blocks),
            "removed_linked_object_ids": removed_linked_object_ids,
        }

    def _compute_block_free_cell_counts(
        self,
        grid: np.ndarray,
        grid_info: dict,
    ) -> Dict[Tuple[int, int], int]:
        """Aggregate free-cell counts per coarse block for one grid."""
        free_rows, free_cols = np.nonzero(grid == 0)
        if free_rows.size == 0:
            self._grid_geometry_signature = (0.0, 0.0, 0.0)
            return {}

        resolution = float(grid_info["resolution"])
        origin_x = float(grid_info["origin_x"])
        origin_y = float(grid_info["origin_y"])

        geometry_signature = (resolution, 0.0, 0.0)
        if self._grid_geometry_signature != geometry_signature:
            self._clear_block_geometry_caches()
            self._grid_geometry_signature = geometry_signature

        block_size_m = float(self.cell_stride) * resolution
        cell_center_x = origin_x + (free_cols.astype(np.float64) + 0.5) * resolution
        cell_center_y = origin_y + (free_rows.astype(np.float64) + 0.5) * resolution
        block_x = np.floor_divide(cell_center_x, block_size_m).astype(np.int64)
        block_y = np.floor_divide(cell_center_y, block_size_m).astype(np.int64)

        block_pairs = np.stack((block_x, block_y), axis=1)
        unique_pairs, inverse_indices, counts = np.unique(
            block_pairs,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        block_counts: Dict[Tuple[int, int], int] = {}
        for idx, pair in enumerate(unique_pairs):
            block_id = (int(pair[0]), int(pair[1]))
            block_counts[block_id] = int(counts[idx])

        if unique_pairs.size > 0 and self.enable_debug_logging:
            sample_block_id = (int(unique_pairs[0][0]), int(unique_pairs[0][1]))
            sample_cache_key = self._make_block_geometry_cache_key(
                sample_block_id,
                resolution,
                0.0,
                0.0,
            )
            if sample_cache_key not in self.block_centroid_cache:
                self._get_block_centroid(sample_block_id, resolution, 0.0, 0.0)
            if self._logged_geometry_sample_signature != sample_cache_key:
                sample_centroid = self.block_centroid_cache.get(sample_cache_key)
                if sample_centroid is not None:
                    self.logger.debug(
                        "[FreeSpaceNodeManager] Sample block geometry "
                        f"block_id={sample_block_id} origin=({float(origin_x):.3f}, "
                        f"{float(origin_y):.3f}) centroid="
                        f"({sample_centroid[0]:.3f}, {sample_centroid[1]:.3f})"
                    )
                    self._logged_geometry_sample_signature = sample_cache_key

        return block_counts

    def _get_block_bounds(
        self,
        block_id: Tuple[int, int],
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[float, float, float, float]:
        """Return world-space bounds for one free-space block."""
        cache_key = self._make_block_geometry_cache_key(
            block_id,
            resolution,
            origin_x,
            origin_y,
        )
        cached = self.block_bounds_cache.get(cache_key)
        if cached is not None:
            return cached

        del origin_x, origin_y
        block_i, block_j = block_id
        block_size_m = float(self.cell_stride) * float(resolution)

        min_x = float(block_i) * block_size_m
        max_x = float(block_i + 1) * block_size_m
        min_y = float(block_j) * block_size_m
        max_y = float(block_j + 1) * block_size_m

        bounds = (min_x, max_x, min_y, max_y)
        self.block_bounds_cache[cache_key] = bounds
        return bounds

    def _get_block_centroid(
        self,
        block_id: Tuple[int, int],
        resolution: float,
        origin_x: float,
        origin_y: float,
    ) -> Tuple[float, float]:
        """Return the world-space centroid for one free-space block."""
        cache_key = self._make_block_geometry_cache_key(
            block_id,
            resolution,
            origin_x,
            origin_y,
        )
        cached = self.block_centroid_cache.get(cache_key)
        if cached is not None:
            return cached
        del origin_x, origin_y
        block_i, block_j = block_id
        block_size_m = float(self.cell_stride) * float(resolution)
        centroid = (
            (float(block_i) + 0.5) * block_size_m,
            (float(block_j) + 0.5) * block_size_m,
        )
        self.block_centroid_cache[cache_key] = centroid
        return centroid

    def _build_nav_node_attributes(
        self,
        block_id: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
        free_cell_count: int,
    ) -> dict:
        min_x, max_x, min_y, max_y = bounds
        return {
            "grid_block": {"x": int(block_id[0]), "y": int(block_id[1])},
            "bounds": {
                "min_x": float(min_x),
                "max_x": float(max_x),
                "min_y": float(min_y),
                "max_y": float(max_y),
            },
            "free_cell_count": int(free_cell_count),
            "meets_minimum_free_cells": bool(
                int(free_cell_count) >= int(self.min_free_cell_count)
            ),
        }

    def _create_or_update_nav_node(
        self,
        block_id: Tuple[int, int],
        grid_info: dict,
        timestamp: float,
    ) -> Optional[int]:
        """Create or refresh the NAVIGATION node for one free-space block."""
        free_cell_count = self.block_free_cell_counts.get(block_id, 0)
        if (
            block_id not in self.block_to_node_id
            and free_cell_count < self.min_free_cell_count
        ):
            return None

        resolution = grid_info["resolution"]
        bounds = self._get_block_bounds(block_id, resolution, 0.0, 0.0)
        centroid_x, centroid_y = self._get_block_centroid(
            block_id,
            resolution,
            0.0,
            0.0,
        )
        attributes = self._build_nav_node_attributes(block_id, bounds, free_cell_count)

        if block_id in self.block_to_node_id:
            node_id = self.block_to_node_id[block_id]
            node = self.nav_node_cache.get(block_id)
            if node is None:
                graph_node = self.sg.query.get_node(node_id)
                if graph_node is None:
                    self.block_to_node_id.pop(block_id, None)
                    return self._create_or_update_nav_node(block_id, grid_info, timestamp)
                node = graph_node

            node.pose.position.x = centroid_x
            node.pose.position.y = centroid_y
            node.pose.position.z = self.z_offset
            node.pose.orientation.w = 1.0
            node.last_seen = timestamp
            node.attributes = attributes

            self.sg.update.update_node(node_id, node)
            self.nav_node_cache[block_id] = node
            self.stats["total_nav_nodes_updated"] += 1
            return node_id

        node = NavNode()
        node.pose.position.x = centroid_x
        node.pose.position.y = centroid_y
        node.pose.position.z = self.z_offset
        node.pose.orientation.w = 1.0
        node.created_at = timestamp
        node.last_seen = timestamp
        node.attributes = attributes

        node_id = self.sg.update.add_node(node)
        self.nav_node_cache[block_id] = node
        self.block_to_node_id[block_id] = node_id
        self.stats["total_nav_nodes_created"] += 1

        self.logger.debug(
            f"Created NAVIGATION block {block_id} (node {node_id}) "
            f"with {free_cell_count} cells at ({centroid_x:.2f}, {centroid_y:.2f})"
        )
        return node_id

    def _get_free_cells_for_block(
        self,
        block_id: Tuple[int, int],
    ) -> Set[Tuple[int, int]]:
        """Return current free occupancy-grid cells belonging to one nav block."""
        cached = self.block_free_cell_cache.get(block_id)
        if cached is not None:
            return cached

        free_cells: Set[Tuple[int, int]] = set()
        if self.current_grid is None or self.current_grid_info is None:
            return free_cells

        resolution = float(self.current_grid_info["resolution"])
        origin_x = float(self.current_grid_info["origin_x"])
        origin_y = float(self.current_grid_info["origin_y"])
        width = int(self.current_grid_info["width"])
        height = int(self.current_grid_info["height"])
        if resolution <= 0.0 or width <= 0 or height <= 0:
            return free_cells

        min_x, max_x, min_y, max_y = self._get_block_bounds(
            block_id,
            resolution,
            origin_x,
            origin_y,
        )
        min_col = max(0, int(math.floor((min_x - origin_x) / resolution - 0.5)) - 1)
        max_col = min(width, int(math.ceil((max_x - origin_x) / resolution - 0.5)) + 2)
        min_row = max(0, int(math.floor((min_y - origin_y) / resolution - 0.5)) - 1)
        max_row = min(height, int(math.ceil((max_y - origin_y) / resolution - 0.5)) + 2)

        for grid_y in range(min_row, max_row):
            for grid_x in range(min_col, max_col):
                if self.current_grid[grid_y, grid_x] != 0:
                    continue
                if (
                    self._grid_cell_to_block_id(
                        grid_x,
                        grid_y,
                        resolution,
                        origin_x,
                        origin_y,
                    )
                    == block_id
                ):
                    free_cells.add((grid_x, grid_y))

        self.block_free_cell_cache[block_id] = free_cells
        return free_cells

    def _get_representative_free_cell(
        self,
        block_id: Tuple[int, int],
        free_cells: Set[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """Pick the free grid cell closest to the NAV node's block centroid."""
        if not free_cells or self.current_grid_info is None:
            return None

        resolution = float(self.current_grid_info["resolution"])
        origin_x = float(self.current_grid_info["origin_x"])
        origin_y = float(self.current_grid_info["origin_y"])
        centroid_x, centroid_y = self._get_block_centroid(
            block_id,
            resolution,
            origin_x,
            origin_y,
        )

        return min(
            free_cells,
            key=lambda cell: (
                (
                    origin_x + (float(cell[0]) + 0.5) * resolution - centroid_x
                )
                ** 2
                + (
                    origin_y + (float(cell[1]) + 0.5) * resolution - centroid_y
                )
                ** 2,
                cell[1],
                cell[0],
            ),
        )

    def _get_neighbor_cell_offsets(self) -> Tuple[Tuple[int, int], ...]:
        """Return grid-cell neighbor offsets matching nav connectivity settings."""
        if self.navigation_connectivity == 4:
            return (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
            )
        return (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    def _has_connected_free_path_between_blocks(
        self,
        block_a: Tuple[int, int],
        block_b: Tuple[int, int],
    ) -> bool:
        """Return True when the current grid connects the nav-block centroids."""
        free_cells_a = self._get_free_cells_for_block(block_a)
        free_cells_b = self._get_free_cells_for_block(block_b)
        start_cell = self._get_representative_free_cell(block_a, free_cells_a)
        target_cell = self._get_representative_free_cell(block_b, free_cells_b)
        if start_cell is None or target_cell is None:
            return False

        allowed_cells = free_cells_a | free_cells_b
        if start_cell not in allowed_cells or target_cell not in allowed_cells:
            return False

        neighbor_offsets = self._get_neighbor_cell_offsets()
        frontier = [start_cell]
        visited = {start_cell}
        while frontier:
            cell_x, cell_y = frontier.pop()
            if (cell_x, cell_y) == target_cell:
                return True

            for dx, dy in neighbor_offsets:
                neighbor = (cell_x + dx, cell_y + dy)
                if neighbor in visited or neighbor not in allowed_cells:
                    continue
                visited.add(neighbor)
                frontier.append(neighbor)

        return False

    def _is_navigation_connection_traversable(
        self,
        block_a: Tuple[int, int],
        block_b: Tuple[int, int],
        active_blocks: Set[Tuple[int, int]],
    ) -> Tuple[bool, str]:
        """Validate one local NAVIGATION block-to-block connection."""
        if block_a == block_b:
            return False, "self_loop"

        if not self._has_connected_free_path_between_blocks(block_a, block_b):
            return False, "blocked"

        del active_blocks
        return True, "accepted"

    def _sync_navigation_edge_cache(
        self,
        block_to_node: Dict[Tuple[int, int], int],
    ) -> None:
        """Synchronize cached NAVIGABLE_PATH pairs with graph state."""
        managed_node_ids = {int(node_id) for node_id in block_to_node.values()}
        graph_pairs: Set[Tuple[int, int]] = set()
        for edge in self.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH):
            source_id = int(edge.source_id)
            target_id = int(edge.target_id)
            if source_id in managed_node_ids and target_id in managed_node_ids:
                graph_pairs.add((source_id, target_id))

        self.nav_edge_pairs = {
            (int(source_id), int(target_id))
            for source_id, target_id in self.nav_edge_pairs | graph_pairs
            if int(source_id) in managed_node_ids and int(target_id) in managed_node_ids
        }

    def _create_navigation_edges(
        self,
        dirty_blocks: Optional[Iterable[Tuple[int, int]]] = None,
        removed_blocks: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> int:
        """Reconcile NAVIGABLE_PATH edges between neighboring NAVIGATION nodes."""
        edges_added = 0
        edges_removed = 0
        candidate_pairs_checked = 0
        candidate_pairs_accepted = 0
        edges_blocked = 0
        diagonal_corner_cut_rejections = 0
        block_to_node = dict(self.block_to_node_id)
        if not block_to_node:
            self.nav_edge_pairs.clear()
            return 0
        self._sync_navigation_edge_cache(block_to_node)

        active_blocks = set(block_to_node)
        dirty_blocks = set(dirty_blocks or block_to_node.keys())
        removed_blocks = set(removed_blocks or ())
        candidate_blocks = set(dirty_blocks)
        candidate_blocks.update(removed_blocks)
        for block_id in tuple(candidate_blocks):
            candidate_blocks.update(
                self._neighbor_block_ids(
                    block_id,
                    connectivity=self.navigation_connectivity,
                )
            )

        desired_block_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        affected_existing_pairs: Set[Tuple[int, int]] = set()
        affected_node_ids = {
            int(node_id)
            for block_id, node_id in block_to_node.items()
            if block_id in candidate_blocks
        }

        for block_id in candidate_blocks:
            if block_id not in active_blocks:
                continue

            for neighbor_block in self._canonical_neighbor_block_ids(
                block_id,
                connectivity=self.navigation_connectivity,
            ):
                if neighbor_block not in active_blocks:
                    continue

                candidate_pairs_checked += 1
                is_traversable, reason = self._is_navigation_connection_traversable(
                    block_id,
                    neighbor_block,
                    active_blocks,
                )
                if not is_traversable:
                    if reason == "corner_cut":
                        diagonal_corner_cut_rejections += 1
                    else:
                        edges_blocked += 1
                    continue

                candidate_pairs_accepted += 1
                desired_block_pairs.add(tuple(sorted((block_id, neighbor_block))))

        desired_pairs: Set[Tuple[int, int]] = set()
        for block_a, block_b in desired_block_pairs:
            node_a = block_to_node.get(block_a)
            node_b = block_to_node.get(block_b)
            if node_a is None or node_b is None or node_a == node_b:
                continue
            desired_pairs.add((int(node_a), int(node_b)))
            desired_pairs.add((int(node_b), int(node_a)))

        for source_id, target_id in tuple(self.nav_edge_pairs):
            if source_id in affected_node_ids or target_id in affected_node_ids:
                affected_existing_pairs.add((source_id, target_id))

        for source_id, target_id in affected_existing_pairs - desired_pairs:
            if self.sg.update.remove_edge(source_id, target_id, EdgeType.NAVIGABLE_PATH):
                edges_removed += 1
                self.nav_edge_pairs.discard((source_id, target_id))

        for source_id, target_id in desired_pairs - self.nav_edge_pairs:
            try:
                self.sg.update.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=target_id,
                        type=EdgeType.NAVIGABLE_PATH,
                    ),
                    is_structural=False,
                )
                edges_added += 1
                self.nav_edge_pairs.add((source_id, target_id))
            except Exception:
                continue

        if (
            candidate_pairs_checked > 0
            or edges_added > 0
            or edges_removed > 0
            or edges_blocked > 0
            or diagonal_corner_cut_rejections > 0
        ):
            self.stats["total_nav_edges_created"] += edges_added
            self.logger.debug(
                f"Navigation adjacency: {candidate_pairs_checked} candidate pairs, "
                f"{candidate_pairs_accepted} accepted, {edges_blocked} blocked, "
                f"{diagonal_corner_cut_rejections} diagonal corner-cut rejections, "
                f"{edges_added} edges added, {edges_removed} removed, "
                f"{len(self.block_to_node_id)} nav nodes"
            )

        return edges_added

    def get_statistics(self) -> Dict[str, int]:
        """Return manager statistics for monitoring."""
        return {
            **self.stats,
            "total_nav_nodes": len(self.block_to_node_id),
        }

    def _remove_nearest_link_if_invalid(
        self,
        object_id: int,
        keep_target_id: Optional[int] = None,
    ) -> bool:
        """Remove stale nearest-link edges while optionally keeping one target."""
        changed = False
        outgoing_edges = self.sg.query.get_outgoing_edges(
            object_id, edge_type=EdgeType.NEAREST_FREE_SPACE
        )
        for edge in outgoing_edges:
            if keep_target_id is not None and edge.target_id == keep_target_id:
                continue
            if self.sg.update.remove_edge(
                edge.source_id,
                edge.target_id,
                EdgeType.NEAREST_FREE_SPACE,
            ):
                changed = True
        if keep_target_id is None:
            self.object_nearest_nav.pop(int(object_id), None)
        return changed

    def _set_nearest_link(self, object_id: int, nav_id: int, distance: float) -> bool:
        """Ensure the object has exactly one nearest-link edge to ``nav_id``."""
        changed = self._remove_nearest_link_if_invalid(object_id, keep_target_id=nav_id)
        retained_edges = self.sg.query.get_outgoing_edges(
            object_id,
            edge_type=EdgeType.NEAREST_FREE_SPACE,
        )
        retained_edge = next(
            (edge for edge in retained_edges if edge.target_id == nav_id),
            None,
        )
        if retained_edge is not None:
            prev_distance = retained_edge.attributes.get("distance")
            if isinstance(prev_distance, (float, int)) and (
                abs(float(prev_distance) - float(distance))
                <= self.nearest_link_distance_epsilon
            ):
                return changed
            self.sg.update.remove_edge(object_id, nav_id, EdgeType.NEAREST_FREE_SPACE)
            changed = True

        self.sg.update.add_edge(
            Edge(
                source_id=object_id,
                target_id=nav_id,
                type=EdgeType.NEAREST_FREE_SPACE,
                attributes={"distance": float(distance)},
            ),
            is_structural=False,
        )
        self.object_nearest_nav[int(object_id)] = int(nav_id)
        return True

    def _resolve_edge_distance(self, edge: Edge, obj_node) -> Optional[float]:
        """Resolve the current object-to-nav distance from edge metadata or geometry."""
        prev_distance = edge.attributes.get("distance") if edge.attributes else None
        if isinstance(prev_distance, (float, int)):
            return float(prev_distance)

        target_node = self.sg.query.get_node(edge.target_id)
        if target_node is None:
            return None

        return float(
            math.hypot(
                float(target_node.pose.position.x) - float(obj_node.pose.position.x),
                float(target_node.pose.position.y) - float(obj_node.pose.position.y),
            )
        )

    def _get_best_nav_candidate_for_object(self, obj_node) -> Optional[Tuple[int, float]]:
        """Return the closest nav candidate within the configured update radius."""
        if (
            obj_node is None
            or obj_node.id is None
            or not self.has_processed_map_snapshot()
            or not self.block_to_node_id
        ):
            return None

        object_block_id = self._index_object_node(obj_node)
        if object_block_id is None:
            return None

        obj_x = float(obj_node.pose.position.x)
        obj_y = float(obj_node.pose.position.y)
        resolution = float(self.cell_key_resolution)

        best_nav_id: Optional[int] = None
        best_distance = float("inf")

        for candidate_block_id in self._iter_candidate_block_ids(
            object_block_id,
            self.nearest_link_max_distance_m,
        ):
            if (
                self.block_free_cell_counts.get(candidate_block_id, 0)
                < self.min_free_cell_count
            ):
                continue
            nav_id = self.block_to_node_id.get(candidate_block_id)
            if nav_id is None:
                continue

            nav_x, nav_y = self._get_block_centroid(
                candidate_block_id,
                resolution,
                self.current_grid_info["origin_x"],
                self.current_grid_info["origin_y"],
            )
            distance = float(math.hypot(nav_x - obj_x, nav_y - obj_y))
            if distance > self.nearest_link_max_distance_m:
                continue

            if (
                distance + self.nearest_link_distance_epsilon < best_distance
                or (
                    abs(distance - best_distance) <= self.nearest_link_distance_epsilon
                    and (best_nav_id is None or nav_id < best_nav_id)
                )
            ):
                best_nav_id = nav_id
                best_distance = distance

        if best_nav_id is None:
            return None
        return (best_nav_id, best_distance)

    def _is_nav_node_currently_qualifying(self, nav_id: int) -> bool:
        """Return True when the nav node's block still has enough free cells."""
        for block_id, node_id in self.block_to_node_id.items():
            if int(node_id) != int(nav_id):
                continue
            return (
                self.block_free_cell_counts.get(block_id, 0)
                >= self.min_free_cell_count
            )
        return False

    def _reconcile_nearest_link_for_object(self, obj_node) -> bool:
        """Repair or improve one object's nearest free-space link."""
        if obj_node is None or obj_node.id is None:
            return False

        object_id = int(obj_node.id)
        best_candidate = self._get_best_nav_candidate_for_object(obj_node)
        current_edges = self.sg.query.get_outgoing_edges(
            object_id,
            edge_type=EdgeType.NEAREST_FREE_SPACE,
        )

        if not current_edges:
            if best_candidate is None:
                return False
            return self._set_nearest_link(object_id, best_candidate[0], best_candidate[1])

        current_edge_distances: List[Tuple[Edge, Optional[float]]] = [
            (edge, self._resolve_edge_distance(edge, obj_node)) for edge in current_edges
        ]
        current_edge, current_distance = min(
            current_edge_distances,
            key=lambda item: (
                item[1] is None,
                float("inf") if item[1] is None else item[1],
                item[0].target_id,
            ),
        )

        if best_candidate is None:
            return self._remove_nearest_link_if_invalid(object_id)

        best_nav_id, best_distance = best_candidate
        if not self._is_nav_node_currently_qualifying(current_edge.target_id):
            return self._set_nearest_link(object_id, best_nav_id, best_distance)

        if current_distance is None:
            return self._set_nearest_link(object_id, best_nav_id, best_distance)

        if current_edge.target_id == best_nav_id:
            return self._set_nearest_link(object_id, best_nav_id, best_distance)

        if best_distance + self.nearest_link_distance_epsilon < current_distance:
            return self._set_nearest_link(object_id, best_nav_id, best_distance)

        return self._set_nearest_link(object_id, current_edge.target_id, current_distance)

    def try_initial_nearest_link(self, obj_node) -> bool:
        """Best-effort callback fast path for newly created objects."""
        if obj_node is None or obj_node.id is None or not self.has_processed_map_snapshot():
            return False

        object_id = int(obj_node.id)
        self._index_object_node(obj_node)
        changed = self._reconcile_nearest_link_for_object(obj_node)
        self.pending_object_ids.discard(object_id)
        return changed

    def update_nearest_freespace_links_for_objects(
        self,
        object_nodes: Optional[list] = None,
    ) -> int:
        """Refresh nearest NAVIGATION links for the provided object subset."""
        if not self.has_processed_map_snapshot():
            self.logger.debug(
                "Nearest free space update skipped: no processed map snapshot"
            )
            return 0

        if object_nodes is None:
            object_nodes = self.sg.query.find_nodes_by_type(NodeType.OBJECT)
        if not object_nodes:
            return 0

        updated_count = 0
        for obj_node in object_nodes:
            if obj_node is None or obj_node.id is None:
                continue

            try:
                if self._reconcile_nearest_link_for_object(obj_node):
                    updated_count += 1
                self.pending_object_ids.discard(int(obj_node.id))
            except Exception as exc:
                self.logger.warning(
                    f"Failed to update nearest free space for object {obj_node.id}: {exc}"
                )

        if updated_count > 0:
            self.logger.debug(
                f"Nearest free space update: {updated_count} objects updated"
            )
        return updated_count
