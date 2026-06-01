"""
Pose Manager - Manages robot pose nodes in the scene graph.

This manager handles the creation and tracking of AGENT nodes in the MOTION layer,
representing the robot's trajectory through the environment.

Features:
- Threshold-based pose sampling (distance and time)
- Pose window tracking for recent poses
- Temporal edge creation between consecutive poses
- Thread-safe scene graph updates

Usage:
    pose_mgr = PoseNodeManager(sg_interface, logger, ...)
    pose_mgr.process_odometry_update(odom_msg)
"""

import math
import threading
from collections import Counter, deque
from typing import Callable, Optional, Set

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import BaseNode, Edge, EdgeType, NodeType, PoseNode
from scene_graph_ros.managers.semantic_signature_utils import (
    build_class_set_from_object_ids,
)


class PoseNodeManager:
    """
    Manages AGENT nodes representing robot poses in the MOTION layer.

    This class processes odometry updates and creates pose nodes based on
    distance and time thresholds, maintaining a sliding window of recent poses.
    """

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        # Pose sampling thresholds
        pose_distance_threshold: float = 1.0,
        pose_time_threshold: float = 2.0,
        pose_rotation_threshold: float = 0.0,
        pose_window_size: int = 3,
        pending_queue_size: int = 200,
        # Debug logging
        enable_debug_logging: bool = True,
        debug_log_interval: int = 10,  # Log every N pose updates
    ):
        """
        Initialize the PoseNodeManager.

        Args:
            sg_interface: Shared scene graph interface for thread-safe access
            logger: ROS logger for debug/info messages
            pose_distance_threshold: Minimum distance (m) between poses
            pose_time_threshold: Minimum time (s) between poses
            pose_window_size: Number of recent poses to track
            enable_debug_logging: Enable detailed debug logging
            debug_log_interval: Log detailed stats every N updates
        """
        self.sg = sg_interface
        self.logger = logger

        # Configuration
        self.pose_distance_threshold = max(0.0, float(pose_distance_threshold))
        self.pose_time_threshold = max(0.0, float(pose_time_threshold))
        self.pose_rotation_threshold = max(0.0, float(pose_rotation_threshold))
        self.pose_window_size = pose_window_size
        self.pending_queue_size = max(1, int(pending_queue_size))

        # Debug logging
        self.enable_debug_logging = enable_debug_logging
        self.debug_log_interval = max(1, int(debug_log_interval))
        self.update_counter = 0

        # Pose tracking state
        self.pose_window: deque[BaseNode] = deque(maxlen=self.pose_window_size)
        self._pending_lock = threading.Lock()
        self._pending_odometry: deque[Odometry] = deque(maxlen=self.pending_queue_size)
        self.last_pose: Optional[PoseStamped] = None
        self.prev_pose_node: Optional[BaseNode] = None
        self.curr_pose_node: Optional[BaseNode] = None
        self.last_received_stamp_sec: Optional[float] = None
        self.last_received_ros_time_sec: Optional[float] = None
        self.last_inserted_stamp_sec: Optional[float] = None
        self.last_inserted_position: Optional[tuple[float, float, float]] = None
        self.last_rejection_reason: Optional[str] = None
        self.graph_revision = 0

        # Statistics (for monitoring)
        self.stats = {
            "total_odom_received": 0,
            "pending_pose_updates": 0,
            "dropped_pending_poses": 0,
            "total_poses_created": 0,
            "total_temporal_edges_created": 0,
            "poses_skipped_distance": 0,
            "poses_skipped_time": 0,
            "poses_skipped_rotation": 0,
            "poses_rejected": 0,
            "last_rejection_reason": None,
            "last_distance_from_accepted": 0.0,
            "last_time_diff_from_accepted": 0.0,
            "last_rotation_diff_from_accepted": 0.0,
            "pose_graph_revision": 0,
        }
        self.rejection_reasons = Counter()

        self.logger.debug("PoseNodeManager initialized:")
        self.logger.debug(f"  - distance_threshold: {self.pose_distance_threshold}m")
        self.logger.debug(f"  - time_threshold: {self.pose_time_threshold}s")
        self.logger.debug(f"  - rotation_threshold: {self.pose_rotation_threshold}rad")
        self.logger.debug(f"  - window_size: {self.pose_window_size}")
        self.logger.debug(f"  - pending_queue_size: {self.pending_queue_size}")
        self.logger.debug(f"  - debug_logging: {self.enable_debug_logging}")

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return Time.from_msg(stamp).nanoseconds / 1e9

    @staticmethod
    def _orientation_angle(a, b) -> float:
        dot = (
            float(a.x) * float(b.x)
            + float(a.y) * float(b.y)
            + float(a.z) * float(b.z)
            + float(a.w) * float(b.w)
        )
        dot = max(-1.0, min(1.0, abs(dot)))
        return 2.0 * math.acos(dot)

    def _pose_node_count(self) -> int:
        try:
            return len(self.sg.query.find_nodes_by_type(NodeType.AGENT))
        except Exception as exc:
            self.logger.warning(
                f"Could not count pose nodes for diagnostics: {exc}",
                throttle_duration_sec=5.0,
            )
            return -1

    def enqueue_odometry_update(
        self,
        odom_data: Odometry,
        *,
        current_ros_time_sec: Optional[float] = None,
    ) -> int:
        """Record one odometry callback without mutating the scene graph."""
        if odom_data is None:
            self.logger.warning("No odometry data received", throttle_duration_sec=5.0)
            return self.pending_count()

        stamp_sec = self._stamp_to_sec(odom_data.header.stamp)
        pos = odom_data.pose.pose.position
        self.stats["total_odom_received"] += 1
        self.last_received_stamp_sec = stamp_sec
        self.last_received_ros_time_sec = current_ros_time_sec

        dropped = False
        with self._pending_lock:
            if len(self._pending_odometry) >= self.pending_queue_size:
                dropped = True
                self.stats["dropped_pending_poses"] += 1
            self._pending_odometry.append(odom_data)
            pending = len(self._pending_odometry)
            self.stats["pending_pose_updates"] = pending

        if self.enable_debug_logging:
            self.logger.debug(
                "[pose_callback] "
                f"count={self.stats['total_odom_received']} "
                f"stamp={stamp_sec:.6f}s ros_now={current_ros_time_sec} "
                f"frame_id='{odom_data.header.frame_id}' "
                f"child_frame_id='{odom_data.child_frame_id}' "
                f"pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f}) "
                f"pending={pending} dropped_oldest={dropped}",
                throttle_duration_sec=2.0,
            )

        return pending

    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending_odometry)

    def drain_pending_odometry(
        self,
        *,
        frame_id: str = "odom",
        max_messages: Optional[int] = None,
        on_pose_added: Optional[Callable[[BaseNode], None]] = None,
    ) -> list[BaseNode]:
        """Process queued odometry updates and return pose nodes that were inserted."""
        with self._pending_lock:
            if max_messages is None or max_messages <= 0:
                count = len(self._pending_odometry)
            else:
                count = min(int(max_messages), len(self._pending_odometry))
            batch = [self._pending_odometry.popleft() for _ in range(count)]
            self.stats["pending_pose_updates"] = len(self._pending_odometry)

        added_nodes: list[BaseNode] = []
        for odom_data in batch:
            node = self.process_odometry_update(odom_data, frame_id=frame_id)
            if node is None:
                continue
            added_nodes.append(node)
            if on_pose_added is not None:
                on_pose_added(node)

        return added_nodes

    def _log_update_stats(
        self,
        pose_added: bool,
        distance: float,
        time_diff: float,
        rotation_diff: float,
        reason: str,
    ):
        """Log detailed update statistics if debug logging is enabled."""
        if not self.enable_debug_logging:
            return

        self.update_counter += 1
        if self.update_counter % self.debug_log_interval != 0 and not pose_added:
            return
        if self.update_counter % self.debug_log_interval != 0:
            return

        self.logger.debug(f"=== PoseNodeManager Update #{self.update_counter} ===")
        self.logger.debug(f"  accepted: {pose_added} reason: {reason}")
        self.logger.debug(
            f"  received={self.stats['total_odom_received']} "
            f"pending={self.pending_count()} "
            f"created={self.stats['total_poses_created']} "
            f"rejected={self.stats['poses_rejected']}"
        )
        self.logger.debug(
            f"  temporal_edges={self.stats['total_temporal_edges_created']} "
            f"pose_nodes={self._pose_node_count()} "
            f"revision={self.graph_revision}"
        )
        self.logger.debug(
            f"  last delta: distance={distance:.3f}m "
            f"time={time_diff:.3f}s rotation={rotation_diff:.3f}rad"
        )
        self.logger.debug(f"  rejected_by_reason={dict(self.rejection_reasons)}")

    def process_odometry_update(
        self, odom_data: Odometry, frame_id: str = "odom"
    ) -> Optional[BaseNode]:
        """
        Process odometry update and create pose node if thresholds are met.

        Only creates a new pose node if the robot has moved sufficiently far
        (distance threshold) or enough time has passed (time threshold) since
        the last pose node was created.

        Args:
            odom_data: Odometry message with robot pose
            frame_id: Frame ID for the pose (unused, kept for API compatibility)

        Returns:
            Created pose node if added, None if skipped
        """
        if odom_data is None:
            self.logger.warn("No odometry data received", throttle_duration_sec=5.0)
            return None

        current_time = Time.from_msg(odom_data.header.stamp)
        current_stamp_sec = current_time.nanoseconds / 1e9
        current_pos = odom_data.pose.pose.position
        current_orientation = odom_data.pose.pose.orientation

        # Check thresholds
        should_add_pose = False
        accept_reason = "first_pose"
        rejection_reason = "below_thresholds"
        distance = 0.0
        time_diff = 0.0
        rotation_diff = 0.0

        if self.last_pose is None:
            # Always add the first pose
            should_add_pose = True
        else:
            # Calculate time difference
            time_diff = (
                current_time - Time.from_msg(self.last_pose.header.stamp)
            ).nanoseconds / 1e9

            # Calculate distance
            lp = self.last_pose.pose.position
            last_orientation = self.last_pose.pose.orientation
            distance = np.sqrt(
                (current_pos.x - lp.x) ** 2
                + (current_pos.y - lp.y) ** 2
                + (current_pos.z - lp.z) ** 2
            )
            rotation_diff = self._orientation_angle(
                current_orientation,
                last_orientation,
            )

            duplicate_sample = (
                abs(time_diff) <= 1e-9
                and distance <= 1e-9
                and rotation_diff <= 1e-9
            )

            # Add pose if EITHER distance OR time threshold is met
            if duplicate_sample:
                rejection_reason = "duplicate_sample"
            elif time_diff < 0.0:
                rejection_reason = "non_monotonic_timestamp"
                if distance >= self.pose_distance_threshold:
                    should_add_pose = True
                    accept_reason = "distance_threshold_after_time_jump"
                elif (
                    self.pose_rotation_threshold > 0.0
                    and rotation_diff >= self.pose_rotation_threshold
                ):
                    should_add_pose = True
                    accept_reason = "rotation_threshold_after_time_jump"
            elif distance >= self.pose_distance_threshold:
                should_add_pose = True
                accept_reason = "distance_threshold"
            elif time_diff >= self.pose_time_threshold:
                should_add_pose = True
                accept_reason = "time_threshold"
            elif (
                self.pose_rotation_threshold > 0.0
                and rotation_diff >= self.pose_rotation_threshold
            ):
                should_add_pose = True
                accept_reason = "rotation_threshold"
            else:
                # Track skipped poses
                if distance < self.pose_distance_threshold:
                    self.stats["poses_skipped_distance"] += 1
                if time_diff < self.pose_time_threshold:
                    self.stats["poses_skipped_time"] += 1
                if (
                    self.pose_rotation_threshold > 0.0
                    and rotation_diff < self.pose_rotation_threshold
                ):
                    self.stats["poses_skipped_rotation"] += 1

        # Only add if threshold met
        if not should_add_pose:
            self.stats["poses_rejected"] += 1
            self.stats["last_rejection_reason"] = rejection_reason
            self.stats["last_distance_from_accepted"] = float(distance)
            self.stats["last_time_diff_from_accepted"] = float(time_diff)
            self.stats["last_rotation_diff_from_accepted"] = float(rotation_diff)
            self.last_rejection_reason = rejection_reason
            self.rejection_reasons[rejection_reason] += 1
            if rejection_reason == "non_monotonic_timestamp":
                self.logger.warning(
                    "[pose_filter] rejected non-monotonic odometry stamp "
                    f"stamp={current_stamp_sec:.6f}s "
                    f"last={self._stamp_to_sec(self.last_pose.header.stamp):.6f}s "
                    f"distance={distance:.3f}m rotation={rotation_diff:.3f}rad "
                    f"pending={self.pending_count()}",
                    throttle_duration_sec=5.0,
                )
            else:
                self.logger.debug(
                    "[pose_filter] rejected pose "
                    f"reason={rejection_reason} "
                    f"stamp={current_stamp_sec:.6f}s "
                    f"distance={distance:.3f}/{self.pose_distance_threshold:.3f}m "
                    f"time={time_diff:.3f}/{self.pose_time_threshold:.3f}s "
                    f"rotation={rotation_diff:.3f}/{self.pose_rotation_threshold:.3f}rad "
                    f"pending={self.pending_count()}",
                    throttle_duration_sec=2.0,
                )
            self._log_update_stats(
                False, distance, time_diff, rotation_diff, rejection_reason
            )
            return None

        # Create AGENT node using PoseNode subclass
        node = PoseNode()  # ID will be assigned by graph
        node.pose.position.x = odom_data.pose.pose.position.x
        node.pose.position.y = odom_data.pose.pose.position.y
        node.pose.position.z = odom_data.pose.pose.position.z
        node.pose.orientation.x = odom_data.pose.pose.orientation.x
        node.pose.orientation.y = odom_data.pose.pose.orientation.y
        node.pose.orientation.z = odom_data.pose.pose.orientation.z
        node.pose.orientation.w = odom_data.pose.pose.orientation.w

        # Convert ROS Time to float timestamp
        timestamp = current_stamp_sec
        node.created_at = timestamp
        node.last_seen = timestamp

        # Add to graph (thread-safe)
        revision_before = self.graph_revision
        node_id = self.sg.update.add_node(node)
        node.id = node_id
        self.graph_revision += 1

        # Update tracking
        self.curr_pose_node = node
        self.pose_window.append(node)

        # Add edge from previous pose if exists
        if self.prev_pose_node is not None:
            edge = Edge(
                source_id=self.prev_pose_node.id,
                target_id=node_id,
                type=EdgeType.TEMPORAL_LINK,
            )
            self.sg.update.add_edge(edge)
            self.stats["total_temporal_edges_created"] += 1
            self.graph_revision += 1

        self.prev_pose_node = node
        self.last_pose = PoseStamped()
        self.last_pose.header = odom_data.header
        self.last_pose.pose = odom_data.pose.pose
        self.last_inserted_stamp_sec = timestamp
        self.last_inserted_position = (
            float(node.pose.position.x),
            float(node.pose.position.y),
            float(node.pose.position.z),
        )

        # Update statistics
        self.stats["total_poses_created"] += 1
        self.stats["last_rejection_reason"] = None
        self.stats["last_distance_from_accepted"] = float(distance)
        self.stats["last_time_diff_from_accepted"] = float(time_diff)
        self.stats["last_rotation_diff_from_accepted"] = float(rotation_diff)
        self.stats["pose_graph_revision"] = self.graph_revision

        # Log debug stats
        self._log_update_stats(True, distance, time_diff, rotation_diff, accept_reason)

        self.logger.debug(
            "[pose_insert] "
            f"node={node_id} reason={accept_reason} "
            f"stamp={timestamp:.6f}s frame_id='{odom_data.header.frame_id}' "
            f"child_frame_id='{odom_data.child_frame_id}' "
            f"pos=({node.pose.position.x:.2f},{node.pose.position.y:.2f},"
            f"{node.pose.position.z:.2f}) "
            f"dist={distance:.3f}m dt={time_diff:.3f}s "
            f"dtheta={rotation_diff:.3f}rad "
            f"revision={revision_before}->{self.graph_revision} "
            f"pose_nodes={self._pose_node_count()} "
            f"pending={self.pending_count()}",
            throttle_duration_sec=1.0,
        )

        return node

    def get_current_pose_node(self) -> Optional[BaseNode]:
        """
        Get the most recent pose node.

        Returns:
            Current pose node, or None if no poses created yet
        """
        return self.curr_pose_node

    def get_pose_window_los_object_ids(self) -> Set[int]:
        """Return deduplicated object IDs observed in LoS across the pose window."""
        object_ids: Set[int] = set()
        for pose_node in self.pose_window:
            attrs = pose_node.attributes or {}
            object_ids.update(attrs.get("object_in_los", []))
        return object_ids

    def get_pose_window_los_object_ids_for_pose_ids(
        self,
        pose_node_ids: Set[int],
    ) -> Set[int]:
        """Return LoS object IDs from the pose window restricted to specific poses."""
        if not pose_node_ids:
            return set()
        allowed_pose_ids = {
            int(pose_node_id)
            for pose_node_id in pose_node_ids
            if pose_node_id is not None
        }

        object_ids: Set[int] = set()
        for pose_node in self.pose_window:
            if pose_node.id is None or int(pose_node.id) not in allowed_pose_ids:
                continue
            attrs = pose_node.attributes or {}
            object_ids.update(attrs.get("object_in_los", []))

        return object_ids

    def build_pose_window_class_set(
        self,
        pose_node_ids: Optional[Set[int]] = None,
    ) -> Set[str]:
        """Build a semantic class set from unique LoS objects in the pose window."""
        if pose_node_ids is None:
            object_ids = self.get_pose_window_los_object_ids()
        else:
            object_ids = self.get_pose_window_los_object_ids_for_pose_ids(pose_node_ids)
        return build_class_set_from_object_ids(
            self.sg,
            object_ids,
        )

    def get_statistics(self) -> dict:
        """
        Get manager statistics.

        Returns:
            Dictionary with statistics (poses created, edges, etc.)
        """
        stats = self.stats.copy()
        stats["pending_pose_updates"] = self.pending_count()
        stats["rejected_by_reason"] = dict(self.rejection_reasons)
        stats["last_received_stamp_sec"] = self.last_received_stamp_sec
        stats["last_received_ros_time_sec"] = self.last_received_ros_time_sec
        stats["last_inserted_stamp_sec"] = self.last_inserted_stamp_sec
        stats["last_inserted_position"] = self.last_inserted_position
        stats["pose_node_count"] = self._pose_node_count()
        return stats
