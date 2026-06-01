"""
Object Manager - Manages object detection nodes and observation links.

This manager handles the creation and tracking of OBJECT nodes in the OBJECT layer,
representing detected objects in the environment. Observation edges are created
during occupancy-grid-based line-of-sight updates.

Features:
- Spatial merging to prevent duplicate object nodes
- Detection confidence tracking and metadata
- OBSERVATION_ANCHOR edge creation to poses via LoS updates
- Thread-safe scene graph updates
- Line-of-sight validation using occupancy-grid raycasting

Usage:
    obj_mgr = ObjectNodeManager(sg_interface, logger, ...)
    obj_mgr.process_detections_update(detections_msg, tf_buffer)
"""

import math
import traceback
from collections import Counter
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from vision_msgs.msg import Detection3DArray

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import (
    BaseNode,
    Edge,
    EdgeType,
    NodeType,
    ObjectNode,
)
from scene_graph_core.services import GraphPatch


class ObjectNodeManager:
    """
    Manages OBJECT nodes representing detected objects in the OBJECT layer.

    This class processes 3D object detections and creates object nodes with
    spatial merging to prevent duplicates. It also performs occupancy-grid
    line-of-sight checks and writes observation edges to visible objects.
    """

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        # Object detection parameters
        spatial_merge_threshold: float = 0.75,  # meters
        # Line-of-sight parameters
        max_los_range: float = 10.0,  # meters — maximum sensing range for LoS
        los_fov_deg: float = 360.0,  # degrees — field of view (360 = omnidirectional)
        los_unknown_is_occupied: bool = False,  # treat unknown grid cells as obstacles
        # Debug logging
        enable_debug_logging: bool = True,
        debug_log_interval: int = 10,  # Log every N detections
    ):
        """
        Initialize the ObjectNodeManager.

        Args:
            sg_interface: Shared scene graph interface for thread-safe access
            logger: ROS logger for debug/info messages
            spatial_merge_threshold: Distance threshold (m) for merging nearby detections
            max_los_range: Maximum sensing range (m) for line-of-sight checks
            los_fov_deg: Field-of-view angle (degrees) centred on robot heading;
                         360 means full omnidirectional visibility
            los_unknown_is_occupied: If True, unknown occupancy grid cells (-1)
                                     are treated as obstacles when raycasting
            enable_debug_logging: Enable detailed debug logging
            debug_log_interval: Log detailed stats every N updates
        """
        self.sg = sg_interface
        self.logger = logger

        # Configuration
        self.spatial_merge_threshold = spatial_merge_threshold
        self.max_los_range = max_los_range
        self.los_fov_deg = los_fov_deg
        self.los_unknown_is_occupied = los_unknown_is_occupied

        # Debug logging
        self.enable_debug_logging = enable_debug_logging
        self.debug_log_interval = debug_log_interval
        self.update_counter = 0

        # Occupancy grid for Bresenham raycasting
        self.occupancy_grid = None
        self.grid_resolution: Optional[float] = None
        self.grid_origin: Optional[Tuple[float, float]] = None
        self.grid_width: Optional[int] = None
        self.grid_height: Optional[int] = None

        # Statistics (for monitoring)
        self.stats = {
            "total_detection_messages": 0,
            "total_detections_received": 0,
            "total_detections_accepted": 0,
            "total_detections_rejected": 0,
            "rejected_by_reason": {},
            "total_objects_created": 0,
            "total_objects_updated": 0,
            "total_objects_merged": 0,
            "total_observation_edges_created": 0,
            "last_detection_stamp_sec": None,
            "last_object_create_stamp_sec": None,
            "last_object_update_stamp_sec": None,
        }
        self._rejected_by_reason = Counter()

        self.logger.debug("ObjectNodeManager initialized:")
        self.logger.debug(
            f"  - spatial_merge_threshold: {self.spatial_merge_threshold}m"
        )
        self.logger.debug(f"  - max_los_range: {self.max_los_range}m")
        self.logger.debug(f"  - los_fov_deg: {self.los_fov_deg}°")
        self.logger.debug(
            f"  - los_unknown_is_occupied: {self.los_unknown_is_occupied}"
        )
        self.logger.debug(f"  - debug_logging: {self.enable_debug_logging}")

    def _initialize_pose_attributes(self, pose_node: BaseNode) -> None:
        """Initialize tracking attributes for a pose node if not present.

        Args:
            pose_node: Pose node to initialize attributes for
        """
        if pose_node.attributes is None:
            pose_node.attributes = {}

        # Initialize object_in_los tracking
        if "object_in_los" not in pose_node.attributes:
            pose_node.attributes["object_in_los"] = []

    def _log_update_stats(self, objects_processed: int, new_objects: int):
        """Log detailed update statistics if debug logging is enabled."""
        if not self.enable_debug_logging:
            return

        self.update_counter += 1
        if self.update_counter % self.debug_log_interval != 0:
            return

        self.logger.debug(f"=== ObjectNodeManager Update #{self.update_counter} ===")
        self.logger.debug(
            f"  Total objects created: {self.stats['total_objects_created']}"
        )
        self.logger.debug(
            f"  Total objects updated: {self.stats['total_objects_updated']}"
        )
        self.logger.debug(
            f"  Total objects merged: {self.stats['total_objects_merged']}"
        )
        self.logger.debug(
            f"  Total observation edges: {self.stats['total_observation_edges_created']}"
        )
        self.logger.debug(
            f"  Last batch: {objects_processed} processed, {new_objects} new"
        )

    def update_occupancy_grid(self, grid_msg: OccupancyGrid):
        """
        Update stored occupancy grid for raycast-based LoS checks.

        Args:
            grid_msg: OccupancyGrid message
        """
        # Store grid data as 2D numpy array
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.int8).reshape((height, width))

        self.occupancy_grid = data
        self.grid_resolution = grid_msg.info.resolution
        self.grid_origin = (
            grid_msg.info.origin.position.x,
            grid_msg.info.origin.position.y,
        )
        self.grid_width = width
        self.grid_height = height

        self.logger.debug(
            f"Updated occupancy grid: {width}x{height}, res={self.grid_resolution:.3f}m"
        )

    def process_detections_update(
        self,
        detections_msg: Detection3DArray,
        tf_buffer,
        fixed_frame_id: str = "world",
        transform_stamped=None,
    ) -> Dict[str, int]:
        """
        Process 3D object detections and create/update OBJECT nodes.

        For each detection:
        1. Transform detection to world frame
        2. Check for spatial merge with existing objects
        3. Create new object or update existing one
        4. Report newly created object IDs for derived nearest-link maintenance

        Args:
            detections_msg: Detection3DArray with 3D bounding boxes
            tf_buffer: TF2 buffer for coordinate transforms
            fixed_frame_id: Target frame for object poses (default: "world")
            transform_stamped: Optional pre-fetched transform from detection frame
                to fixed frame. Supplying this avoids TF lookups inside SG critical sections.

        Returns:
            Dictionary with update statistics
        """
        if detections_msg is None or len(detections_msg.detections) == 0:
            self.stats["total_detection_messages"] += 1
            return {
                "objects_processed": 0,
                "accepted_detections": 0,
                "rejected_detections": 0,
                "rejected_by_reason": {},
                "new_objects": 0,
                "updated_objects": 0,
                "new_object_ids": [],
                "updated_object_ids": [],
            }

        # Get detection frame from message
        det_frame = detections_msg.header.frame_id
        detection_count = len(detections_msg.detections)
        self.stats["total_detection_messages"] += 1
        self.stats["total_detections_received"] += detection_count

        if transform_stamped is None and det_frame and det_frame != fixed_frame_id:
            # Get transform from detection frame to world frame
            try:
                transform_stamped = tf_buffer.lookup_transform(
                    fixed_frame_id,  # target frame
                    det_frame,  # source frame (camera/sensor frame)
                    Time(),  # Get latest available transform
                    Duration(seconds=1),  # timeout
                )
            except Exception as e:
                self.logger.warn(
                    f"Could not get transform from {det_frame} to {fixed_frame_id}: {e}",
                    throttle_duration_sec=5.0,
                )
                self._record_detection_rejection(
                    "tf_lookup_failed", detection_count
                )
                self.stats["total_detections_rejected"] += detection_count
                return {
                    "objects_processed": 0,
                    "accepted_detections": 0,
                    "rejected_detections": detection_count,
                    "rejected_by_reason": {
                        "tf_lookup_failed": detection_count,
                    },
                    "new_objects": 0,
                    "updated_objects": 0,
                    "new_object_ids": [],
                    "updated_object_ids": [],
                }

        # Convert ROS Time to float timestamp
        ros_time = Time.from_msg(detections_msg.header.stamp)
        timestamp = ros_time.nanoseconds / 1e9
        self.stats["last_detection_stamp_sec"] = timestamp

        # Process each detection
        new_objects = 0
        updated_objects = 0
        accepted_detections = 0
        rejected_detections = 0
        rejected_by_reason = Counter()
        new_object_ids: List[int] = []
        updated_object_ids: List[int] = []

        for detection_index, detection in enumerate(detections_msg.detections):
            obj_node, is_new, reason = self._process_single_detection(
                detection, timestamp, transform_stamped, detection_index=detection_index
            )

            if obj_node is not None:
                accepted_detections += 1
                if is_new:
                    new_objects += 1
                    if obj_node.id is not None:
                        new_object_ids.append(int(obj_node.id))
                else:
                    updated_objects += 1
                    if obj_node.id is not None:
                        updated_object_ids.append(int(obj_node.id))

                # Existing objects are not re-queued here because object poses are
                # currently static after creation. If object poses become mutable in
                # the runtime, moved IDs should be reported for maintenance here.
            else:
                rejected_detections += 1
                rejected_by_reason[str(reason or "unknown")] += 1

        # Build result stats
        objects_processed = len(detections_msg.detections)
        self.stats["total_detections_accepted"] += accepted_detections
        self.stats["total_detections_rejected"] += rejected_detections
        for reason, count in rejected_by_reason.items():
            self._record_detection_rejection(reason, count)
        self.stats["rejected_by_reason"] = dict(self._rejected_by_reason)

        result_stats = {
            "objects_processed": objects_processed,
            "accepted_detections": accepted_detections,
            "rejected_detections": rejected_detections,
            "rejected_by_reason": dict(rejected_by_reason),
            "new_objects": new_objects,
            "updated_objects": updated_objects,
            "new_object_ids": new_object_ids,
            "updated_object_ids": updated_object_ids,
            "object_count": len(self.sg.query.find_nodes_by_type(NodeType.OBJECT)),
        }

        self._log_update_stats(objects_processed, new_objects)

        return result_stats

    def _record_detection_rejection(self, reason: str, count: int = 1) -> None:
        """Track detection rejection reasons for runtime diagnostics."""
        reason = str(reason or "unknown")
        count = max(0, int(count))
        if count <= 0:
            return
        self._rejected_by_reason[reason] += count
        self.stats["rejected_by_reason"] = dict(self._rejected_by_reason)

    def _is_finite_pose(self, pose: Pose) -> bool:
        """Return True if a pose contains finite position and orientation values."""
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return all(math.isfinite(float(value)) for value in values)

    def _is_finite_bbox_size(self, detection) -> bool:
        """Return True when bbox dimensions are finite."""
        size = detection.bbox.size
        return all(
            math.isfinite(float(value))
            for value in (size.x, size.y, size.z)
        )

    def _detection_label_and_score(self, detection) -> tuple[Optional[str], Optional[float]]:
        """Extract the most useful class label and score from a Detection3D."""
        label = str(getattr(detection, "id", "") or "").strip()
        score = None
        if detection.results:
            hypothesis = detection.results[0].hypothesis
            if not label:
                label = str(getattr(hypothesis, "class_id", "") or "").strip()
            try:
                score = float(hypothesis.score)
            except (TypeError, ValueError):
                score = None
        return (label or None), score

    def _process_single_detection(
        self,
        detection,
        timestamp: float,
        transform_stamped,
        detection_index: int = 0,
    ) -> Tuple[Optional[BaseNode], bool, Optional[str]]:
        """
        Process a single 3D object detection.

        Creates or updates an OBJECT node in the scene graph.
        Uses spatial merging to prevent duplicates.

        Args:
            detection: Single Detection3D message
            timestamp: Timestamp of the detection
            transform_stamped: TF transform from detection frame to world frame

        Returns:
            Tuple of (object_node, is_new) where is_new indicates if this is a new object
        """
        class_label, detection_score = self._detection_label_and_score(detection)
        if detection.results and detection_score is not None and not math.isfinite(detection_score):
            self.logger.warning(
                "[object_detection] rejected detection "
                f"idx={detection_index} label={class_label!r}: nonfinite_score",
                throttle_duration_sec=5.0,
            )
            return None, False, "nonfinite_score"

        if not self._is_finite_bbox_size(detection):
            self.logger.warning(
                "[object_detection] rejected detection "
                f"idx={detection_index} label={class_label!r}: nonfinite_bbox_size",
                throttle_duration_sec=5.0,
            )
            return None, False, "nonfinite_bbox_size"

        # Create pose in detection frame
        det_ps = Pose()
        det_ps.position = detection.bbox.center.position
        det_ps.orientation = detection.bbox.center.orientation
        if not self._is_finite_pose(det_ps):
            self.logger.warning(
                "[object_detection] rejected detection "
                f"idx={detection_index} label={class_label!r}: nonfinite_bbox_center",
                throttle_duration_sec=5.0,
            )
            return None, False, "nonfinite_bbox_center"

        # Transform to world/map frame
        if transform_stamped is None:
            world_ps = det_ps
        else:
            try:
                world_ps = do_transform_pose(det_ps, transform_stamped)
            except Exception as e:
                self.logger.warn(
                    "[object_detection] rejected detection "
                    f"idx={detection_index} label={class_label!r}: "
                    f"transform_failed: {e}",
                    throttle_duration_sec=5.0,
                )
                return None, False, "transform_failed"

        if world_ps is None or not self._is_finite_pose(world_ps):
            self.logger.warn(
                "[object_detection] rejected detection "
                f"idx={detection_index} label={class_label!r}: invalid_world_pose",
                throttle_duration_sec=5.0,
            )
            return None, False, "invalid_world_pose"

        # Create OBJECT node in world frame using ObjectNode subclass
        node = ObjectNode()  # ID will be assigned by graph
        node.pose.position.x = world_ps.position.x
        node.pose.position.y = world_ps.position.y
        node.pose.position.z = world_ps.position.z
        node.pose.orientation.x = world_ps.orientation.x
        node.pose.orientation.y = world_ps.orientation.y
        node.pose.orientation.z = world_ps.orientation.z
        node.pose.orientation.w = world_ps.orientation.w
        node.created_at = timestamp
        node.last_seen = timestamp

        # Add detection metadata
        if detection.results:
            node.attributes = {
                "class_name": class_label,  # Human-readable class name
                "class_id": detection.results[0].hypothesis.class_id,
                "detection_score": detection_score,
            }
        elif class_label:
            node.attributes = {"class_name": class_label}

        # Use spatial merge to find existing node or add new one (prevents duplicates)
        try:
            merged_node, is_new = self._find_or_add(
                node, range_m=self.spatial_merge_threshold
            )
        except Exception as exc:
            self.logger.error(
                "[object_detection] rejected detection "
                f"idx={detection_index} label={class_label!r}: graph_mutation_failed: "
                f"{exc}\n{traceback.format_exc()}"
            )
            return None, False, "graph_mutation_failed"

        # Update tracking
        if is_new:
            self.stats["total_objects_created"] += 1
            self.stats["last_object_create_stamp_sec"] = timestamp
            self.logger.debug(
                "[object_detection] accepted detection "
                f"idx={detection_index} label={class_label!r} "
                f"score={detection_score} action=create object_id={merged_node.id} "
                f"world=({merged_node.pose.position.x:.2f}, "
                f"{merged_node.pose.position.y:.2f}, "
                f"{merged_node.pose.position.z:.2f})"
            )
        else:
            # Update last_seen timestamp for existing node
            merged_node.last_seen = timestamp
            self.stats["total_objects_updated"] += 1
            self.stats["last_object_update_stamp_sec"] = timestamp
            self.logger.debug(
                "[object_detection] accepted detection "
                f"idx={detection_index} label={class_label!r} "
                f"score={detection_score} action=update object_id={merged_node.id} "
                f"world=({world_ps.position.x:.2f}, "
                f"{world_ps.position.y:.2f}, "
                f"{world_ps.position.z:.2f})"
            )

        if merged_node.attributes is None:
            merged_node.attributes = {}
        merged_node.attributes["object_id"] = merged_node.id
        self.sg.update.update_node(merged_node.id, merged_node)

        return merged_node, is_new, None

    def _get_normalized_class_name(self, node: Optional[BaseNode]) -> Optional[str]:
        """
        Return a normalized semantic class label for class-aware object merges.

        Labels are normalized to stripped lowercase strings. Missing or invalid
        labels return ``None`` so unlabeled detections do not merge by accident.
        """
        if node is None:
            return None

        attrs = node.attributes
        if not isinstance(attrs, Mapping):
            return None

        class_name = attrs.get("class_name")
        if class_name is None:
            return None

        normalized = str(class_name).strip().lower()
        if not normalized:
            return None

        return normalized

    def _find_or_add(
        self, node: BaseNode, range_m: float = 0.75
    ) -> Tuple[BaseNode, bool]:
        """
        Find existing node within range or add new node to graph.

        Performs class-aware spatial merging to prevent duplicate object nodes.
        Objects merge only when they are in the same layer, are spatially close,
        and share the same normalized ``attributes['class_name']``. Unlabeled
        objects are kept distinct instead of guessing semantic compatibility.
        A second safeguard suppresses creating a new object when another object
        already occupies essentially the same pose, even if the class label
        flickers to a different value.

        Args:
            node: Candidate node to add
            range_m: Search radius in meters

        Returns:
            Tuple of (found_or_added_node, is_new) where is_new=True if node was added
        """
        # Check if a spatially close node exists in the scene graph
        # Pass node_type for faster spatial index query (only searches nodes of same type)
        # Returns list of (node, distance) tuples
        candidates_with_dist = self.sg.query.find_nodes_by_position_xyz(
            node.pose.position, range_m, node_type=node.node_type
        )

        incoming_class_name = self._get_normalized_class_name(node)
        same_pose_range_m = min(range_m, 0.15)
        best_candidate: Optional[BaseNode] = None
        best_candidate_key: Optional[Tuple[float, int]] = None
        best_same_pose_candidate: Optional[BaseNode] = None
        best_same_pose_key: Optional[Tuple[float, int]] = None

        for candidate_node, distance in candidates_with_dist:
            if candidate_node.layer != node.layer:
                continue

            candidate_id = (
                int(candidate_node.id) if candidate_node.id is not None else -1
            )
            candidate_key = (float(distance), candidate_id)

            # Do not create a second object on top of an existing one just because
            # the detector briefly changed class labels at the same pose.
            if float(distance) <= same_pose_range_m:
                if (
                    best_same_pose_key is None
                    or candidate_key < best_same_pose_key
                ):
                    best_same_pose_candidate = candidate_node
                    best_same_pose_key = candidate_key

            candidate_class_name = self._get_normalized_class_name(candidate_node)

            # Only merge when each object has a usable semantic class label and
            # the normalized labels match exactly.
            if incoming_class_name is None or candidate_class_name is None:
                continue
            if candidate_class_name != incoming_class_name:
                continue

            if best_candidate_key is None or candidate_key < best_candidate_key:
                best_candidate = candidate_node
                best_candidate_key = candidate_key

        if best_candidate is not None:
            self.stats["total_objects_merged"] += 1
            return best_candidate, False

        if best_same_pose_candidate is not None:
            return best_same_pose_candidate, False

        # No existing node found, add new one
        node_id = self.sg.update.add_node(node)
        node.id = node_id  # Update node with assigned ID
        return node, True

    def _create_observation_edge(self, pose_node: BaseNode, obj_node: BaseNode) -> bool:
        """
        Create OBSERVATION_ANCHOR edge from pose to object.

        Args:
            pose_node: Pose node (observer)
            obj_node: Object node (observed)

        Returns:
            True if edge was created, False otherwise
        """
        if pose_node is None or obj_node is None:
            return False

        if pose_node.id is None or obj_node.id is None:
            self.logger.warn("Cannot create observation edge: node IDs are None")
            return False

        # Check if edge already exists to avoid duplicates
        if self.sg.query.has_edge(
            pose_node.id, obj_node.id, edge_type=EdgeType.OBSERVATION_ANCHOR
        ):
            self.logger.debug(
                f"OBSERVATION_ANCHOR edge already exists: pose {pose_node.id} -> object {obj_node.id}"
            )
            return False

        # Create edge (relational, not structural)
        edge = Edge(
            source_id=pose_node.id,
            target_id=obj_node.id,
            type=EdgeType.OBSERVATION_ANCHOR,
        )
        self.sg.update.add_edge(edge, is_structural=False)
        self.stats["total_observation_edges_created"] += 1

        self.logger.debug(
            f"Created OBSERVATION_ANCHOR edge: pose {pose_node.id} -> object {obj_node.id}"
        )

        return True

    def _create_observation_edges_batch(
        self,
        pose_node: BaseNode,
        object_nodes: List[BaseNode],
        existing_target_ids: Optional[Set[int]] = None,
    ) -> int:
        """Create missing OBSERVATION_ANCHOR edges from one pose in a single patch."""
        if pose_node is None or pose_node.id is None or not object_nodes:
            return 0

        existing_target_ids = existing_target_ids or set()
        patch = GraphPatch()
        added_count = 0
        pose_id = int(pose_node.id)
        for obj_node in object_nodes:
            if obj_node is None or obj_node.id is None:
                continue
            object_id = int(obj_node.id)
            if object_id in existing_target_ids:
                continue
            patch.add_edge(
                Edge(
                    source_id=pose_id,
                    target_id=object_id,
                    type=EdgeType.OBSERVATION_ANCHOR,
                    is_structural=False,
                ),
                is_structural=False,
            )
            existing_target_ids.add(object_id)
            added_count += 1

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)
            self.stats["total_observation_edges_created"] += added_count

        return added_count

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.

        Args:
            x, y: World coordinates (meters)

        Returns:
            Tuple of (grid_x, grid_y) cell indices
        """
        if self.occupancy_grid is None:
            return None, None

        grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
        grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)

        return grid_x, grid_y

    def _is_grid_cell_valid(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid cell is within bounds.

        Args:
            grid_x, grid_y: Grid cell indices

        Returns:
            True if cell is within grid bounds
        """
        if self.occupancy_grid is None:
            return False

        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height

    def _is_grid_cell_occupied(
        self, grid_x: int, grid_y: int, unknown_is_occupied: bool = True
    ) -> bool:
        """
        Check if grid cell is occupied or unknown.

        Args:
            grid_x, grid_y: Grid cell indices
            unknown_is_occupied: Treat unknown cells (-1) as occupied

        Returns:
            True if cell is occupied or (optionally) unknown
        """
        if not self._is_grid_cell_valid(grid_x, grid_y):
            return True  # Out of bounds = occupied

        cell_value = self.occupancy_grid[grid_y, grid_x]

        # OccupancyGrid values: -1=unknown, 0=free, 100=occupied
        if cell_value >= 50:  # Occupied threshold
            return True
        if unknown_is_occupied and cell_value < 0:  # Unknown
            return True

        return False

    def _raycast_grid_bresenham(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        unknown_is_occupied: bool = False,
    ) -> bool:
        """
        Raycast from (x0,y0) to (x1,y1) in world coordinates using Bresenham.

        Early-exit if any occupied cell is encountered along the ray.

        Args:
            x0, y0: Start position in world coordinates (pose)
            x1, y1: End position in world coordinates (object)
            unknown_is_occupied: Treat unknown cells as occupied (stricter)

        Returns:
            True if ray is clear (not occluded), False if occluded
        """
        if self.occupancy_grid is None:
            # No grid available - fall back to permissive (not occluded)
            return True

        # Convert to grid coordinates
        gx0, gy0 = self._world_to_grid(x0, y0)
        gx1, gy1 = self._world_to_grid(x1, y1)

        if gx0 is None or gx1 is None:
            # Conversion failed
            return True

        # Bresenham's line algorithm
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        gx, gy = gx0, gy0

        # March along the ray
        while True:
            # Check current cell (skip start and end cells)
            if not (gx == gx0 and gy == gy0) and not (gx == gx1 and gy == gy1):
                if self._is_grid_cell_occupied(gx, gy, unknown_is_occupied):
                    # Hit an occupied cell - ray is occluded
                    return False

            # Reached end
            if gx == gx1 and gy == gy1:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx += sx
            if e2 < dx:
                err += dx
                gy += sy

        # Ray is clear
        return True

    def get_los_candidates(
        self, pose_node: BaseNode, radius: Optional[float] = None
    ) -> List[BaseNode]:
        """
        Get candidate object IDs for line-of-sight test from the scene graph.

        Uses spatial query to find objects within radius of the pose position.

        Args:
            pose_node: Current pose node
            radius: Search radius in metres (defaults to max_los_range)

        Returns:
            Object nodes within radius
        """
        if radius is None:
            radius = self.max_los_range

        # Spatial query: Find all OBJECT nodes within radius
        # Returns list of (node, distance) tuples
        nearby_nodes_with_dist = self.sg.query.find_nodes_by_position_xyz(
            pose_node.pose.position,
            max_range=radius,
            node_type=NodeType.OBJECT,
        )

        candidate_nodes: List[BaseNode] = [
            node for node, _ in nearby_nodes_with_dist if node.id is not None
        ]

        self.logger.debug(
            f"LoS candidates from spatial query: {len(candidate_nodes)} objects within {radius:.1f}m"
        )

        return candidate_nodes

    def compute_line_of_sight_for_pose(
        self,
        pose_node: BaseNode,
        max_los_range: Optional[float] = None,
        fov_deg: Optional[float] = None,
        unknown_is_occupied: Optional[bool] = None,
    ) -> List[int]:
        """
        Compute which objects are visible from *pose_node* using Bresenham raycasting
        on the occupancy grid.

        Algorithm
        ---------
        1. Spatial query — collect all OBJECT nodes within *max_los_range*.
        2. Distance filter — discard objects beyond *max_los_range*.
        3. FOV filter — discard objects outside the robot's field of view
           (centred on its heading; 360° means omnidirectional).
        4. Bresenham raycast — march from the robot grid cell to the object
           grid cell along a straight line. The ray is **blocked** by any grid
           cell whose value is ≥ 50 (occupied). Unknown cells (value == -1)
           block the ray when *unknown_is_occupied* is True.
        5. If the ray is clear → object is visible.
        6. Persist visible IDs into ``pose_node.attributes["object_in_los"]``
           and update the node in the scene graph.

        Assumptions
        -----------
        * The occupancy grid and the pose/object positions share the same 2-D
          coordinate frame (x-forward, y-left).
        * Only the x-y plane is considered (3-D objects are projected to 2-D).
        * When no occupancy grid is available the method returns an empty list
          without raising an error.

        Args:
            pose_node:            Current robot pose node.
            max_los_range:        Maximum sensing range in metres.
                                  Defaults to ``self.max_los_range``.
            fov_deg:              Field-of-view angle in degrees, centred on the
                                  robot heading.  360 means fully omnidirectional.
                                  Defaults to ``self.los_fov_deg``.
            unknown_is_occupied:  If True, unknown grid cells (-1) are treated as
                                  obstacles during raycasting.
                                  Defaults to ``self.los_unknown_is_occupied``.

        Returns:
            List of object IDs confirmed visible from *pose_node*.
        """
        if pose_node is None or pose_node.id is None:
            return []

        # Resolve defaults
        if max_los_range is None:
            max_los_range = self.max_los_range
        if fov_deg is None:
            fov_deg = self.los_fov_deg
        if unknown_is_occupied is None:
            unknown_is_occupied = self.los_unknown_is_occupied

        # Fetch latest pose state from the graph
        current_pose = self.sg.query.get_node(pose_node.id)
        if current_pose is None:
            self.logger.warn(f"LoS: pose node {pose_node.id} not found in graph")
            return []

        if self.occupancy_grid is None:
            self.logger.debug("LoS skipped: no occupancy grid available")
            self._persist_pose_los(current_pose, set())
            return []

        # Collect candidates within range
        candidate_nodes = self.get_los_candidates(current_pose, radius=max_los_range)
        if not candidate_nodes:
            self.logger.debug(
                f"LoS pose {current_pose.id}: no candidates within {max_los_range:.1f}m"
            )
            self._persist_pose_los(current_pose, set())
            return []

        # Precompute FOV half-angle and robot heading (only needed for <360° FOV)
        half_fov_rad: Optional[float] = None
        robot_yaw: float = 0.0
        if fov_deg < 360.0:
            half_fov_rad = np.deg2rad(fov_deg / 2.0)
            q = current_pose.pose.orientation
            robot_yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2),
            )

        rx = float(current_pose.pose.position.x)
        ry = float(current_pose.pose.position.y)
        raycast = self._raycast_grid_bresenham

        visible_ids: Set[int] = set()
        visible_nodes: List[BaseNode] = []
        occluded_count = 0
        fov_filtered = 0
        existing_observation_targets = {
            int(edge.target_id)
            for edge in self.sg.query.get_outgoing_edges(
                int(current_pose.id),
                edge_type=EdgeType.OBSERVATION_ANCHOR,
            )
        }

        for obj_node in candidate_nodes:
            obj_id = int(obj_node.id)

            ox = float(obj_node.pose.position.x)
            oy = float(obj_node.pose.position.y)
            dx = ox - rx
            dy = oy - ry
            dist = np.hypot(dx, dy)

            # ── Filter 1: distance ────────────────────────────────────────────
            if dist > max_los_range:
                continue  # already filtered by spatial query, but be defensive

            # ── Filter 2: field of view ───────────────────────────────────────
            if half_fov_rad is not None:
                angle = np.arctan2(dy, dx)
                # Normalise relative angle to [-π, π]
                delta = (angle - robot_yaw + np.pi) % (2.0 * np.pi) - np.pi
                if abs(delta) > half_fov_rad:
                    fov_filtered += 1
                    continue

            # ── Filter 3: Bresenham raycast on occupancy grid ─────────────────
            if raycast(rx, ry, ox, oy, unknown_is_occupied):
                visible_ids.add(obj_id)
                visible_nodes.append(obj_node)
            else:
                occluded_count += 1

        if visible_nodes:
            self._create_observation_edges_batch(
                current_pose,
                visible_nodes,
                existing_target_ids=existing_observation_targets,
            )

        # Persist and log
        self._persist_pose_los(current_pose, visible_ids)

        self.logger.debug(
            f"LoS pose {current_pose.id}: {len(candidate_nodes)} candidates → "
            f"{len(visible_ids)} visible, {occluded_count} occluded, "
            f"{fov_filtered} outside FOV"
        )

        return list(visible_ids)

    def _persist_pose_los(self, pose_node: BaseNode, visible_ids: Set[int]) -> None:
        """
        Write *visible_ids* into ``pose_node.attributes["object_in_los"]`` and
        persist the updated node to the scene graph.

        The attribute is **replaced** (not accumulated) so that each call
        reflects the current visibility state of the pose.

        Args:
            pose_node:   Pose node to update.
            visible_ids: Set of currently visible object IDs.
        """
        self._initialize_pose_attributes(pose_node)
        pose_node.attributes["object_in_los"] = sorted(int(v) for v in visible_ids)
        self.sg.update.update_node(pose_node.id, pose_node)

    def get_statistics(self) -> dict:
        """
        Get manager statistics.

        Returns:
            Dictionary with statistics (objects created, updated, etc.)
        """
        return self.stats.copy()
