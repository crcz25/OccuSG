"""Scene-graph orchestrator that materializes visited stable regions directly."""

from __future__ import annotations

import logging
import sys
import threading
import time
from enum import Enum
from typing import Dict, Optional, Set

import rclpy
from incremental_dude_msgs.msg import Region2DArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import NodeType
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray

from scene_graph_ros.json_export import (
    export_scene_graph_json_if_configured,
    register_export_triggers,
)
from scene_graph_ros.detection_queue import DetectionInputQueue, QueuedDetectionMessage
from scene_graph_ros.managers.free_space_manager import FreeSpaceNodeManager
from scene_graph_ros.managers.object_manager import ObjectNodeManager
from scene_graph_ros.managers.pose_manager import PoseNodeManager
from scene_graph_ros.managers.region_manager import PreparedTrackerRegion, RegionManager
from scene_graph_ros.managers.room_manager import RoomManager
from scene_graph_ros.profiling import ProfilingRecorder
from scene_graph_ros.visualization_node import VisualizationNode

logger = logging.getLogger(__name__)


class SemanticState(str, Enum):
    """Runtime phases for the direct region-to-room orchestrator."""

    BOOTSTRAP = "BOOTSTRAP"
    EXPLORING_ROOM = "EXPLORING_ROOM"


SCENE_GRAPH_PARAMETER_DEFAULTS = {
    "odom_topic": "/odom",
    "3d_detections_topic": "/semantic_node/detections",
    "map_topic": "/mapUAV",
    "stable_regions_topic": "/dude/regions_stable",
    "fixed_frame_id": "odom",
    "odom_qos_history": "keep_last",
    "odom_qos_reliability": "reliable",
    "odom_qos_durability": "volatile",
    "odom_qos_depth": 10,
    "detections_qos_history": "keep_last",
    "detections_qos_reliability": "reliable",
    "detections_qos_durability": "volatile",
    "detections_qos_depth": 10,
    "map_qos_history": "keep_last",
    "map_qos_reliability": "reliable",
    "map_qos_durability": "volatile",
    "map_qos_depth": 10,
    "stable_regions_qos_history": "keep_last",
    "stable_regions_qos_reliability": "reliable",
    "stable_regions_qos_durability": "volatile",
    "stable_regions_qos_depth": 20,
    "ignore_empty_region_updates_after_valid": True,
    "materialize_observed_regions": True,
    "pipeline_hz": 1.0,
    "maintenance_hz": 1.0,
    "pose_flush_hz": 20.0,
    "pose_flush_lock_timeout_ms": 5.0,
    "pose_flush_max_messages": 50,
    "detection_flush_hz": 20.0,
    "detection_queue_size": 100,
    "detection_flush_max_messages": 20,
    "detection_lock_timeout_ms": 5.0,
    "detection_watchdog_sec": 10.0,
    "pose_queue_size": 200,
    "pose_watchdog_sec": 10.0,
    "visualization_hz": 1.0,
    "process_map_in_callback": False,
    "detections_tf_lookup_timeout_sec": 0.2,
    "maintenance_tick_warn_ms": 500.0,
    "pose_distance_threshold": 5.0,
    "pose_time_threshold": 5.0,
    "pose_rotation_threshold": 0.0,
    "pose_window_size": 3,
    "obj_spatial_merge_threshold": 0.75,
    "room_z_offset": 12.0,
    "region_z_offset": 8.0,
    "nav_region_boundary_epsilon_m": 0.15,
    "nav_region_enable_neighbor_tiebreak": True,
    "fs_cell_stride_cells": 10,
    "fs_min_free_cell_count": 50,
    "fs_z_offset": 4.0,
    "fs_navigation_connectivity": 8,
    "nearest_link_max_distance_m": 1.0,
    "scene_graph_topic": "/dsg/scene_graph",
    "scene_graph_qos_history": "keep_last",
    "scene_graph_qos_reliability": "reliable",
    "scene_graph_qos_durability": "volatile",
    "scene_graph_qos_depth": 10,
    "enable_debug_logging": True,
    "debug_log_interval": 10,
    "life_time": 0.0,
    "enable_pose_markers": True,
    "enable_pose_labels": False,
    "enable_object_markers": True,
    "enable_object_labels": True,
    "enable_room_markers": True,
    "enable_room_labels": True,
    "enable_region_markers": True,
    "enable_region_labels": True,
    "enable_navigation_markers": True,
    "enable_navigation_labels": False,
    "enable_pose_edges": True,
    "enable_observation_edges": True,
    "enable_navigation_edges": True,
    "enable_region_contains_edges": True,
    "enable_room_region_edges": True,
    "enable_room_adjacency_edges": True,
    "enable_nearest_freespace_edges": True,
    "export_json_path": "",
    "export_json_on_shutdown": True,
    "export_json_compact": False,
    "enable_profiling": False,
    "profiling_output_path": "",
    "profiling_run_name": "run",
    "profiling_save_on_shutdown": True,
    "profiling_discard_first_n": 5,
    "pose_marker_stride": 1,
    "pose_label_stride": 4,
    "pose_edge_stride": 1,
    "navigation_marker_stride": 1,
    "navigation_label_stride": 4,
    "navigation_edge_stride": 1,
    "visualization_warn_ms": 250.0,
    "visualization_stats_interval": 20,
}

SCENE_GRAPH_PARAMETER_GROUPS = {
    "ROS Topics & Frames": [
        "odom_topic",
        "3d_detections_topic",
        "map_topic",
        "stable_regions_topic",
        "fixed_frame_id",
    ],
    "QoS Profiles": [
        "odom_qos_history",
        "odom_qos_reliability",
        "odom_qos_durability",
        "odom_qos_depth",
        "detections_qos_history",
        "detections_qos_reliability",
        "detections_qos_durability",
        "detections_qos_depth",
        "map_qos_history",
        "map_qos_reliability",
        "map_qos_durability",
        "map_qos_depth",
        "stable_regions_qos_history",
        "stable_regions_qos_reliability",
        "stable_regions_qos_durability",
        "stable_regions_qos_depth",
        "ignore_empty_region_updates_after_valid",
        "materialize_observed_regions",
    ],
    "Runtime Stability": [
        "pipeline_hz",
        "maintenance_hz",
        "pose_flush_hz",
        "pose_flush_lock_timeout_ms",
        "pose_flush_max_messages",
        "detection_flush_hz",
        "detection_queue_size",
        "detection_flush_max_messages",
        "detection_lock_timeout_ms",
        "detection_watchdog_sec",
        "pose_queue_size",
        "pose_watchdog_sec",
        "visualization_hz",
        "process_map_in_callback",
        "detections_tf_lookup_timeout_sec",
        "maintenance_tick_warn_ms",
    ],
    "Pose Manager (pose_*)": [
        "pose_distance_threshold",
        "pose_time_threshold",
        "pose_rotation_threshold",
        "pose_window_size",
    ],
    "Object Manager (obj_*)": [
        "obj_spatial_merge_threshold",
    ],
    "Room Bootstrap": ["room_z_offset"],
    "Region Bootstrap": [
        "region_z_offset",
        "nav_region_boundary_epsilon_m",
        "nav_region_enable_neighbor_tiebreak",
    ],
    "Free Space Manager (fs_*)": [
        "fs_cell_stride_cells",
        "fs_min_free_cell_count",
        "fs_z_offset",
        "fs_navigation_connectivity",
        "nearest_link_max_distance_m",
    ],
    "Debug & Visualization": [
        "scene_graph_topic",
        "scene_graph_qos_history",
        "scene_graph_qos_reliability",
        "scene_graph_qos_durability",
        "scene_graph_qos_depth",
        "enable_debug_logging",
        "debug_log_interval",
        "life_time",
        "visualization_warn_ms",
        "visualization_stats_interval",
    ],
    "Visualization Layers": [
        "enable_pose_markers",
        "enable_pose_labels",
        "enable_object_markers",
        "enable_object_labels",
        "enable_room_markers",
        "enable_room_labels",
        "enable_region_markers",
        "enable_region_labels",
        "enable_navigation_markers",
        "enable_navigation_labels",
        "enable_pose_edges",
        "enable_observation_edges",
        "enable_navigation_edges",
        "enable_region_contains_edges",
        "enable_room_region_edges",
        "enable_room_adjacency_edges",
        "enable_nearest_freespace_edges",
    ],
    "JSON Export": [
        "export_json_path",
        "export_json_on_shutdown",
        "export_json_compact",
    ],
    "Profiling": [
        "enable_profiling",
        "profiling_output_path",
        "profiling_run_name",
        "profiling_save_on_shutdown",
        "profiling_discard_first_n",
    ],
    "Visualization Sampling": [
        "pose_marker_stride",
        "pose_label_stride",
        "pose_edge_stride",
        "navigation_marker_stride",
        "navigation_label_stride",
        "navigation_edge_stride",
    ],
}


class SceneGraphOrchestrator(Node):
    """Own the geometric runtime and materialize stable-region rooms directly."""

    def __init__(self):
        super().__init__("scene_graph_region")

        self.sg = create_scene_graph_interface()
        self._sg_lock = threading.RLock()
        self._json_export_completed = False

        self._declare_parameters()
        self._load_parameters()
        self.profiler = ProfilingRecorder(
            node_name="scene_graph_region",
            package_name="scene_graph_ros",
            run_name=str(self._param_dict.get("profiling_run_name", "run")),
            output_path=str(self._param_dict.get("profiling_output_path", "")),
            enabled=bool(self._param_dict.get("enable_profiling", False)),
            save_on_shutdown=bool(
                self._param_dict.get("profiling_save_on_shutdown", True)
            ),
            discard_first_n=int(self._param_dict.get("profiling_discard_first_n", 5)),
            file_tag="scene_graph_region",
            metadata={
                "entity_assignment_graph_assembly_definition": (
                    "maintenance tick samples that performed useful work"
                ),
                "scene_graph_pipeline_tick_definition": (
                    "region-to-room decision timer callback"
                ),
                "detection_assignment_definition": (
                    "object insertion/update timing in detection flush"
                ),
            },
        )

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odom_msg = None
        self.global_map_msg = None
        self._map_dirty = False
        self.latest_stable_regions_msg = None
        self.last_valid_stable_regions_msg = None
        self._stable_regions_snapshot_is_stale = False
        self._consecutive_empty_region_updates = 0
        self._pose_callback_group = ReentrantCallbackGroup()

        self.semantic_state = SemanticState.BOOTSTRAP
        self.current_region: Optional[int] = None
        self.current_room_id: Optional[int] = None
        self.last_non_none_tracker_region_id: Optional[int] = None
        self.detection_queue = DetectionInputQueue(
            max_messages=int(self._param_dict.get("detection_queue_size", 100))
        )
        self._reset_semantic_evidence()

        self.pose_manager = PoseNodeManager(
            sg_interface=self.sg,
            logger=self.get_logger(),
            pose_distance_threshold=self._param_dict.get(
                "pose_distance_threshold", 5.0
            ),
            pose_time_threshold=self._param_dict.get("pose_time_threshold", 5.0),
            pose_rotation_threshold=self._param_dict.get(
                "pose_rotation_threshold", 0.0
            ),
            pose_window_size=self._param_dict.get("pose_window_size", 3),
            pending_queue_size=self._param_dict.get("pose_queue_size", 200),
            enable_debug_logging=self._param_dict.get("enable_debug_logging", True),
            debug_log_interval=self._param_dict.get("debug_log_interval", 10),
        )
        self.fs_manager = FreeSpaceNodeManager(
            sg_interface=self.sg,
            logger=self.get_logger(),
            cell_stride_cells=self._param_dict.get("fs_cell_stride_cells", 10),
            min_free_cell_count=self._param_dict.get("fs_min_free_cell_count", 50),
            z_offset=self._param_dict.get("fs_z_offset", 4.0),
            navigation_connectivity=int(
                self._param_dict.get("fs_navigation_connectivity", 8)
            ),
            nearest_link_max_distance_m=float(
                self._param_dict.get("nearest_link_max_distance_m", 1.0)
            ),
            enable_debug_logging=self._param_dict.get("enable_debug_logging", True),
            debug_log_interval=self._param_dict.get("debug_log_interval", 10),
        )
        self.obj_manager = ObjectNodeManager(
            sg_interface=self.sg,
            logger=self.get_logger(),
            spatial_merge_threshold=self._param_dict.get(
                "obj_spatial_merge_threshold", 0.75
            ),
            enable_debug_logging=self._param_dict.get("enable_debug_logging", True),
            debug_log_interval=self._param_dict.get("debug_log_interval", 10),
        )
        self.room_manager = RoomManager(
            sg_interface=self.sg,
            logger=self.get_logger(),
            z_offset=float(self._param_dict.get("room_z_offset", 12.0)),
        )
        self.region_manager = RegionManager(
            sg_interface=self.sg,
            logger=self.get_logger(),
            z_offset=float(self._param_dict.get("region_z_offset", 8.0)),
            nav_region_boundary_epsilon_m=float(
                self._param_dict.get("nav_region_boundary_epsilon_m", 0.15)
            ),
            nav_region_enable_neighbor_tiebreak=bool(
                self._param_dict.get("nav_region_enable_neighbor_tiebreak", True)
            ),
        )

        self._create_subscribers()

        self.visualization_node = VisualizationNode(
            sg_interface=self.sg,
            param_dict=self._param_dict,
        )

        pipeline_hz = self._get_validated_timer_hz("pipeline_hz", 1.0)
        maintenance_hz = self._get_validated_timer_hz("maintenance_hz", 1.0)
        pose_flush_hz = self._get_validated_timer_hz("pose_flush_hz", 20.0)
        detection_flush_hz = self._get_validated_timer_hz(
            "detection_flush_hz", 20.0
        )

        pipeline_period = 1.0 / pipeline_hz
        self._pipeline_timer = self.create_timer(pipeline_period, self._pipeline_tick)
        self.get_logger().info(
            f"Pipeline timer created: {pipeline_hz:.2f} Hz "
            f"(period={pipeline_period:.3f} s)"
        )

        maintenance_period = 1.0 / maintenance_hz
        self._maintenance_timer = self.create_timer(
            maintenance_period,
            self._maintenance_tick,
        )
        self.get_logger().info(
            f"Maintenance timer created: {maintenance_hz:.2f} Hz "
            f"(period={maintenance_period:.3f} s)"
        )

        pose_flush_period = 1.0 / pose_flush_hz
        self._pose_flush_timer = self.create_timer(
            pose_flush_period,
            self._pose_flush_tick,
            callback_group=self._pose_callback_group,
        )
        self.get_logger().info(
            f"Pose flush timer created: {pose_flush_hz:.2f} Hz "
            f"(period={pose_flush_period:.3f} s)"
        )

        detection_flush_period = 1.0 / detection_flush_hz
        self._detection_flush_timer = self.create_timer(
            detection_flush_period,
            self._detection_flush_tick,
        )
        self.get_logger().info(
            f"Detection flush timer created: {detection_flush_hz:.2f} Hz "
            f"(period={detection_flush_period:.3f} s)"
        )

        self.get_logger().info("SceneGraphOrchestrator initialization complete")

    def _reset_semantic_evidence(self):
        """Clear the latest pipeline-evidence snapshot."""
        self.latest_semantic_evidence = {
            "semantic_state": self.semantic_state.value,
            "current_region": self.current_region,
            "current_room_id": self.current_room_id,
            "transition_anchor_region_id": self.last_non_none_tracker_region_id,
            "resolved_tracker_region_id": self.current_region,
            "resolved_region_node_id": None,
            "current_region_owner_id": self.current_room_id,
            "region_changed": False,
            "room_created": False,
            "region_created": False,
            "decision_outcome": None,
        }

    def _record_pipeline_decision(
        self,
        *,
        transition_anchor_region_id: Optional[int],
        resolved_tracker_region_id: Optional[int],
        resolved_region_node_id: Optional[int],
        current_region_owner_id: Optional[int],
        current_region: Optional[int],
        current_room_id: Optional[int],
        region_changed: bool,
        room_created: bool,
        region_created: bool,
        decision_outcome: str,
    ) -> None:
        """Persist and log the current pipeline decision."""
        self.latest_semantic_evidence = {
            "semantic_state": self.semantic_state.value,
            "current_region": (
                int(current_region) if current_region is not None else None
            ),
            "current_room_id": (
                int(current_room_id) if current_room_id is not None else None
            ),
            "transition_anchor_region_id": (
                int(transition_anchor_region_id)
                if transition_anchor_region_id is not None
                else None
            ),
            "resolved_tracker_region_id": (
                int(resolved_tracker_region_id)
                if resolved_tracker_region_id is not None
                else None
            ),
            "resolved_region_node_id": (
                int(resolved_region_node_id)
                if resolved_region_node_id is not None
                else None
            ),
            "current_region_owner_id": (
                int(current_region_owner_id)
                if current_region_owner_id is not None
                else None
            ),
            "region_changed": bool(region_changed),
            "room_created": bool(room_created),
            "region_created": bool(region_created),
            "decision_outcome": str(decision_outcome),
        }

        self.get_logger().info(
            "\n".join(
                [
                    "[pipeline_tick] region_decision",
                    f"  state: {self.semantic_state.value}",
                    f"  transition_anchor_region: {transition_anchor_region_id}",
                    f"  resolved_tracker_region: {resolved_tracker_region_id}",
                    f"  resolved_region_node: {resolved_region_node_id}",
                    f"  current_region_owner: {current_region_owner_id}",
                    f"  current_room: {current_room_id}",
                    f"  region_changed: {region_changed}",
                    f"  room_created: {room_created}",
                    f"  region_created: {region_created}",
                    f"  decision_outcome: {decision_outcome}",
                ]
            )
        )

    def _clear_pruned_runtime_references(
        self,
        *,
        pruned_region_node_ids: Set[int],
        pruned_tracker_region_ids: Set[int],
        pruned_room_ids: Set[int],
    ) -> None:
        """Clear cached runtime references that still point at deleted structure."""
        if self.current_region in pruned_tracker_region_ids:
            self.current_region = None
        if self.last_non_none_tracker_region_id in pruned_tracker_region_ids:
            self.last_non_none_tracker_region_id = None

        self.current_room_id = self._get_room_owner_for_tracker_region(
            self.current_region
        )
        if self.current_room_id in pruned_room_ids:
            self.current_room_id = None

        if not isinstance(self.latest_semantic_evidence, dict):
            return

        if (
            self.latest_semantic_evidence.get("resolved_region_node_id")
            in pruned_region_node_ids
        ):
            self.latest_semantic_evidence["resolved_region_node_id"] = None
        if (
            self.latest_semantic_evidence.get("resolved_tracker_region_id")
            in pruned_tracker_region_ids
        ):
            self.latest_semantic_evidence["resolved_tracker_region_id"] = None
        if (
            self.latest_semantic_evidence.get("transition_anchor_region_id")
            in pruned_tracker_region_ids
        ):
            self.latest_semantic_evidence["transition_anchor_region_id"] = None
        if self.latest_semantic_evidence.get("current_room_id") in pruned_room_ids:
            self.latest_semantic_evidence["current_room_id"] = None
        if (
            self.latest_semantic_evidence.get("current_region_owner_id")
            in pruned_room_ids
        ):
            self.latest_semantic_evidence["current_region_owner_id"] = None

        self.latest_semantic_evidence["current_region"] = self.current_region
        self.latest_semantic_evidence["current_room_id"] = self.current_room_id
        self.latest_semantic_evidence["transition_anchor_region_id"] = (
            self.last_non_none_tracker_region_id
        )
        self.latest_semantic_evidence["current_region_owner_id"] = self.current_room_id

    def _get_room_owner_for_tracker_region(
        self,
        tracker_region_id: Optional[int],
    ) -> Optional[int]:
        """Return the room anchored to a tracker/stable region."""
        return self.room_manager.get_room_id_for_tracker_region(tracker_region_id)

    def _declare_parameters(self):
        for param_name, default_value in SCENE_GRAPH_PARAMETER_DEFAULTS.items():
            self.declare_parameter(param_name, default_value)

    def _load_parameters(self):
        self._param_dict = {
            name: self.get_parameter(name).value
            for name in SCENE_GRAPH_PARAMETER_DEFAULTS
        }

        self.get_logger().info("=" * 80)
        self.get_logger().info("EFFECTIVE PARAMETERS (from YAML + defaults)")
        self.get_logger().info("=" * 80)
        for group_name, keys in SCENE_GRAPH_PARAMETER_GROUPS.items():
            self.get_logger().info(f"{group_name}:")
            for key in keys:
                self.get_logger().info(f"  {key}: {self._param_dict.get(key)}")
        self.get_logger().info("=" * 80)

    def _get_validated_timer_hz(self, param_name: str, default_hz: float) -> float:
        """Return a positive timer frequency, warning when normalization is needed."""
        raw_value = self._param_dict.get(param_name, default_hz)
        try:
            hz = float(raw_value)
        except (TypeError, ValueError):
            self.get_logger().warning(
                f"Invalid {param_name} '{raw_value}', using {default_hz:.2f} Hz"
            )
            return float(default_hz)

        if hz <= 0.0:
            self.get_logger().warning(
                f"{param_name} must be > 0.0 Hz; using {default_hz:.2f} Hz instead "
                f"of '{raw_value}'"
            )
            return float(default_hz)

        return float(hz)

    def _parse_qos_history(self, value: str) -> HistoryPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "keep_all":
            return HistoryPolicy.KEEP_ALL
        if value_norm == "keep_last":
            return HistoryPolicy.KEEP_LAST
        self.get_logger().warning(f"Invalid QoS history '{value}', using keep_last")
        return HistoryPolicy.KEEP_LAST

    def _parse_qos_reliability(self, value: str) -> ReliabilityPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "best_effort":
            return ReliabilityPolicy.BEST_EFFORT
        if value_norm == "reliable":
            return ReliabilityPolicy.RELIABLE
        self.get_logger().warning(f"Invalid QoS reliability '{value}', using reliable")
        return ReliabilityPolicy.RELIABLE

    def _parse_qos_durability(self, value: str) -> DurabilityPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "transient_local":
            return DurabilityPolicy.TRANSIENT_LOCAL
        if value_norm == "volatile":
            return DurabilityPolicy.VOLATILE
        self.get_logger().warning(f"Invalid QoS durability '{value}', using volatile")
        return DurabilityPolicy.VOLATILE

    def _build_topic_qos(self, prefix: str, default_depth: int = 10) -> QoSProfile:
        history = self._parse_qos_history(
            self._param_dict.get(f"{prefix}_qos_history", "keep_last")
        )
        reliability = self._parse_qos_reliability(
            self._param_dict.get(f"{prefix}_qos_reliability", "reliable")
        )
        durability = self._parse_qos_durability(
            self._param_dict.get(f"{prefix}_qos_durability", "volatile")
        )
        raw_depth = self._param_dict.get(f"{prefix}_qos_depth", default_depth)
        try:
            depth = max(1, int(raw_depth))
        except (TypeError, ValueError):
            depth = default_depth

        return QoSProfile(
            history=history,
            reliability=reliability,
            durability=durability,
            depth=depth,
        )

    def _create_subscribers(self):
        odom_qos = self._build_topic_qos("odom")
        detections_qos = self._build_topic_qos("detections")
        map_qos = self._build_topic_qos("map")
        stable_regions_qos = self._build_topic_qos("stable_regions", default_depth=20)

        self.odom_subscriber = self.create_subscription(
            Odometry,
            self._param_dict.get("odom_topic"),
            self._odom_callback,
            odom_qos,
            callback_group=self._pose_callback_group,
        )
        self.detections_subscriber = self.create_subscription(
            Detection3DArray,
            self._param_dict.get("3d_detections_topic"),
            self._detections_callback,
            detections_qos,
        )
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            self._param_dict.get("map_topic"),
            self._map_callback,
            map_qos,
        )
        self.stable_regions_subscriber = self.create_subscription(
            Region2DArray,
            self._param_dict.get("stable_regions_topic"),
            self._stable_regions_callback,
            stable_regions_qos,
        )

        self.get_logger().info("Subscribers created:")
        self.get_logger().info(f"  - Odometry: {self._param_dict.get('odom_topic')}")
        self.get_logger().info(
            f"  - 3D Detections: {self._param_dict.get('3d_detections_topic')}"
        )
        self.get_logger().info(f"  - Map: {self._param_dict.get('map_topic')}")
        self.get_logger().info(
            f"  - Stable Regions: {self._param_dict.get('stable_regions_topic')}"
        )

    def _odom_callback(self, msg: Odometry):
        """Ingest odometry without blocking on graph maintenance."""
        self.odom_msg = msg
        self.pose_manager.enqueue_odometry_update(
            msg,
            current_ros_time_sec=self.get_clock().now().nanoseconds / 1e9,
        )

    def _pose_flush_tick(self):
        """Flush queued odometry into the graph with bounded lock waiting."""
        pending = self.pose_manager.pending_count()
        if pending <= 0:
            return

        timeout_ms = float(self._param_dict.get("pose_flush_lock_timeout_ms", 5.0))
        max_messages = int(self._param_dict.get("pose_flush_max_messages", 50))
        start_t = time.perf_counter()
        acquired = self._sg_lock.acquire(timeout=max(0.0, timeout_ms / 1000.0))
        wait_ms = (time.perf_counter() - start_t) * 1000.0
        if not acquired:
            stats = self.pose_manager.get_statistics()
            self.get_logger().warning(
                "[pose_flush] skipped because graph lock is busy "
                f"pending={pending} received={stats['total_odom_received']} "
                f"created={stats['total_poses_created']} "
                f"last_received_stamp={stats['last_received_stamp_sec']} "
                f"last_inserted_stamp={stats['last_inserted_stamp_sec']}",
                throttle_duration_sec=float(
                    self._param_dict.get("pose_watchdog_sec", 10.0)
                ),
            )
            return

        try:
            revision_before = self.pose_manager.graph_revision
            added_nodes = self.pose_manager.drain_pending_odometry(
                frame_id=self._param_dict.get("fixed_frame_id", "odom"),
                max_messages=max_messages,
            )
        except Exception as exc:
            import traceback

            self.get_logger().error(
                f"[pose_flush] failed while applying queued odometry: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return
        finally:
            self._sg_lock.release()

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        revision_after = self.pose_manager.graph_revision
        if added_nodes or elapsed_ms > float(
            self._param_dict.get("maintenance_tick_warn_ms", 500.0)
        ):
            self.get_logger().debug(
                "[pose_flush] "
                f"pending_before={pending} added={len(added_nodes)} "
                f"pending_after={self.pose_manager.pending_count()} "
                f"lock_wait={wait_ms:.1f}ms elapsed={elapsed_ms:.1f}ms "
                f"revision={revision_before}->{revision_after}",
                throttle_duration_sec=2.0,
            )

    def _detections_callback(self, msg: Detection3DArray):
        """Queue detections without blocking behind graph maintenance."""
        now_sec = self.get_clock().now().nanoseconds / 1e9
        queued = self.detection_queue.enqueue(msg, now_sec)
        self.get_logger().debug(
            "[detections_callback] queued "
            f"seq={queued.sequence} detections={len(msg.detections)} "
            f"stamp={self.detection_queue.snapshot().get('last_msg_stamp_sec')} "
            f"ros_now={now_sec:.3f} frame={msg.header.frame_id!r} "
            f"pending={self.detection_queue.pending_count()}",
            throttle_duration_sec=2.0,
        )
        self._flush_pending_detections(max_messages=1)

    def _resolve_detection_transform(self, msg: Detection3DArray):
        """Resolve the optional detection-frame transform outside the graph lock."""
        fixed_frame_id = self._param_dict.get("fixed_frame_id", "world")
        detection_frame = str(msg.header.frame_id or "")
        if detection_frame and detection_frame != fixed_frame_id:
            try:
                timeout_sec = float(
                    self._param_dict.get("detections_tf_lookup_timeout_sec", 0.2)
                )
            except (TypeError, ValueError):
                timeout_sec = 0.2
            try:
                return self.tf_buffer.lookup_transform(
                    fixed_frame_id,
                    detection_frame,
                    Time(),
                    Duration(seconds=max(0.0, timeout_sec)),
                )
            except Exception as exc:
                self.get_logger().warning(
                    "[detections_flush] rejected message: "
                    f"tf_lookup_failed {detection_frame}->{fixed_frame_id}: {exc}",
                    throttle_duration_sec=5.0,
                )
                raise

        return None

    def _detection_flush_tick(self):
        """Apply queued detections with bounded graph-lock waiting."""
        self._flush_pending_detections()

    def _flush_pending_detections(self, max_messages: Optional[int] = None):
        """Drain queued detection messages into OBJECT nodes."""
        flush_start_t = time.perf_counter()
        pending = self.detection_queue.pending_count()
        if pending <= 0:
            return

        if max_messages is None:
            max_messages = int(
                self._param_dict.get("detection_flush_max_messages", 20)
            )
        batch = self.detection_queue.pop_batch(max_messages)
        if not batch:
            return

        fixed_frame_id = self._param_dict.get("fixed_frame_id", "world")
        ready: list[tuple[QueuedDetectionMessage, object]] = []
        for queued in batch:
            try:
                transform_stamped = self._resolve_detection_transform(queued.msg)
            except Exception:
                self.detection_queue.record_tf_rejection("tf_lookup_failed")
                continue
            ready.append((queued, transform_stamped))

        if not ready:
            return

        timeout_ms = float(self._param_dict.get("detection_lock_timeout_ms", 5.0))
        start_t = time.perf_counter()
        acquired = self._sg_lock.acquire(timeout=max(0.0, timeout_ms / 1000.0))
        wait_ms = (time.perf_counter() - start_t) * 1000.0
        if not acquired:
            self.detection_queue.push_front(item for item, _ in ready)
            diag = self.detection_queue.snapshot()
            self.get_logger().warning(
                "[detections_flush] deferred because graph lock is busy "
                f"pending={diag['pending_messages']} "
                f"received={diag['messages_received']} "
                f"accepted={diag['detections_accepted']} "
                f"rejected={diag['detections_rejected']} "
                f"created={diag['objects_created']} "
                f"updated={diag['objects_updated']} "
                f"lock_wait={wait_ms:.1f}ms",
                throttle_duration_sec=float(
                    self._param_dict.get("detection_watchdog_sec", 10.0)
                ),
            )
            return

        try:
            applied_messages = 0
            accepted_detections = 0
            for queued, transform_stamped in ready:
                object_count_before = len(
                    self.sg.query.find_nodes_by_type(NodeType.OBJECT)
                )
                detection_stats = self.obj_manager.process_detections_update(
                    queued.msg,
                    self.tf_buffer,
                    fixed_frame_id=fixed_frame_id,
                    transform_stamped=transform_stamped,
                )
                self.detection_queue.record_apply_result(
                    detection_stats,
                    self.get_clock().now().nanoseconds / 1e9,
                )
                object_count_after = len(
                    self.sg.query.find_nodes_by_type(NodeType.OBJECT)
                )
                self.get_logger().debug(
                    "[detections_flush] applied "
                    f"seq={queued.sequence} detections={len(queued.msg.detections)} "
                    f"accepted={detection_stats.get('accepted_detections', 0)} "
                    f"rejected={detection_stats.get('rejected_detections', 0)} "
                    f"created={detection_stats.get('new_objects', 0)} "
                    f"updated={detection_stats.get('updated_objects', 0)} "
                    f"objects={object_count_before}->{object_count_after} "
                    f"pending={self.detection_queue.pending_count()}",
                    throttle_duration_sec=2.0,
                )
                applied_messages += 1
                accepted_detections += int(
                    detection_stats.get("accepted_detections", 0)
                )

                new_object_ids = detection_stats.get("new_object_ids", [])
                if not new_object_ids:
                    continue

                self.fs_manager.queue_object_ids_for_nearest_link(new_object_ids)
        except Exception as exc:
            import traceback

            self.get_logger().error(
                f"[detections_flush] failed while applying queued detections: {exc}\n"
                f"{traceback.format_exc()}"
            )
        finally:
            self._sg_lock.release()

        self.profiler.record(
            "detection_assignment_ms",
            (time.perf_counter() - flush_start_t) * 1000.0,
            metadata={
                "messages": applied_messages,
                "accepted_detections": accepted_detections,
            },
        )

    def get_detection_diagnostics(self) -> dict:
        """Return detection ingestion counters for debug tooling/tests."""
        diag = self.detection_queue.snapshot()
        diag["object_manager"] = dict(getattr(self.obj_manager, "stats", {}))
        diag["object_count"] = len(self.sg.query.find_nodes_by_type(NodeType.OBJECT))
        diag["semantic_state"] = str(self.semantic_state)
        return diag

    def _map_callback(self, msg: OccupancyGrid):
        """Ingest occupancy maps used for navigation-node maintenance."""
        with self._sg_lock:
            self.global_map_msg = msg
            self._map_dirty = True

    def _stable_regions_callback(self, stable_regions_msg: Region2DArray):
        """Cache the latest stable-region snapshot for geometric maintenance."""
        with self._sg_lock:
            region_count = len(stable_regions_msg.regions)
            if region_count == 0:
                self._consecutive_empty_region_updates = (
                    getattr(self, "_consecutive_empty_region_updates", 0) + 1
                )
                if (
                    bool(
                        self._param_dict.get(
                            "ignore_empty_region_updates_after_valid", True
                        )
                    )
                    and getattr(self, "last_valid_stable_regions_msg", None)
                    is not None
                ):
                    self.latest_stable_regions_msg = self.last_valid_stable_regions_msg
                    self._stable_regions_snapshot_is_stale = True
                    self.get_logger().warning(
                        "[stable_regions_callback] ignored empty stable-region "
                        "update and preserved previous valid snapshot "
                        f"consecutive_empty={self._consecutive_empty_region_updates} "
                        f"preserved_regions={len(self.latest_stable_regions_msg.regions)}"
                    )
                    return

            self.latest_stable_regions_msg = stable_regions_msg
            self._stable_regions_snapshot_is_stale = False
            if region_count > 0:
                self.last_valid_stable_regions_msg = stable_regions_msg
                self._consecutive_empty_region_updates = 0

    def _get_current_pose_node_id(self) -> Optional[int]:
        current_pose_node = self.pose_manager.get_current_pose_node()
        if current_pose_node is None or current_pose_node.id is None:
            return None
        return int(current_pose_node.id)

    def _resolve_current_tracker_region(
        self,
        current_pose_node_id: Optional[int],
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> tuple[
        Optional[int],
        Optional[PreparedTrackerRegion],
        Optional[int],
        Optional[int],
    ]:
        """Resolve the tracker region, promoted region, and owner for the current pose."""
        resolved_tracker_region_id = self.region_manager.find_tracker_region_for_pose(
            current_pose_node_id,
            prepared_regions,
        )
        self.region_manager.set_current_resolved_region(resolved_tracker_region_id)
        if resolved_tracker_region_id is None:
            return None, None, None, None

        resolved_tracker_region_id = int(resolved_tracker_region_id)
        prepared_region = prepared_regions.get(int(resolved_tracker_region_id))
        if prepared_region is None:
            self.region_manager.set_current_resolved_region(None)
            return None, None, None, None

        current_region_owner_id = self._get_room_owner_for_tracker_region(
            int(resolved_tracker_region_id)
        )
        return (
            int(resolved_tracker_region_id),
            prepared_region,
            None,
            current_region_owner_id,
        )

    def _sync_room_members_from_region(
        self,
        room_node_id: int,
        prepared_region: PreparedTrackerRegion,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> Dict[NodeType, set[int]]:
        """Attach current geometric members directly to one anchored room."""
        member_ids = self.region_manager.gather_region_member_ids(
            prepared_region,
            prepared_regions=prepared_regions,
        )
        self.room_manager.sync_room_membership_from_region(
            int(room_node_id), member_ids
        )
        return member_ids

    def _materialize_rooms_for_observed_regions(
        self,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> tuple[set[int], dict[int, set[int]]]:
        """Create room anchors for unanchored regions with graph entities."""
        if not bool(self._param_dict.get("materialize_observed_regions", True)):
            return set(), {}

        created_room_ids: set[int] = set()
        region_nav_ids_by_tracker_region: dict[int, set[int]] = {}

        for tracker_region_id, prepared_region in sorted(prepared_regions.items()):
            tracker_region_id = int(tracker_region_id)
            member_ids = self.region_manager.gather_region_member_ids(
                prepared_region,
                prepared_regions=prepared_regions,
            )
            if not self.region_manager.has_meaningful_member_ids(member_ids):
                continue

            region_nav_ids_by_tracker_region[tracker_region_id] = set(
                member_ids.get(NodeType.NAVIGATION, set())
            )
            existing_room_id = self._get_room_owner_for_tracker_region(
                tracker_region_id
            )
            if existing_room_id is not None:
                continue

            room_id = self.room_manager.create_room_from_region(
                prepared_region,
                reason="region_contains_existing_entities",
            )
            if room_id is None:
                continue

            self.room_manager.associate_room_with_tracker_region(
                int(room_id),
                tracker_region_id,
                prepared_region,
            )
            self.room_manager.sync_room_membership_from_region(
                int(room_id),
                member_ids,
            )
            self.room_manager.build_room_region_signature_set(
                int(room_id), persist=True
            )
            created_room_ids.add(int(room_id))
            self.get_logger().info(
                "[maintenance_tick] materialized room for observed region "
                f"tracker_region_id={tracker_region_id} room_id={room_id} "
                f"agent_count={len(member_ids.get(NodeType.AGENT, set()))} "
                f"object_count={len(member_ids.get(NodeType.OBJECT, set()))} "
                "navigation_count="
                f"{len(member_ids.get(NodeType.NAVIGATION, set()))}"
            )

        return created_room_ids, region_nav_ids_by_tracker_region

    def _materialize_room_for_current_region(
        self,
        *,
        pose_node_id: Optional[int],
        tracker_region_id: int,
        prepared_region: PreparedTrackerRegion,
        prepared_regions: Dict[int, PreparedTrackerRegion],
        existing_room_id: Optional[int] = None,
        room_name: Optional[str] = None,
        is_bootstrap_region: bool = False,
    ) -> tuple[Optional[int], bool]:
        """Ensure the current tracker region has an anchored room and direct members."""
        if pose_node_id is None:
            return None, False

        room_created = False
        room_id = existing_room_id
        if room_id is None:
            if is_bootstrap_region:
                room_id = self.room_manager.create_initial_room(int(pose_node_id))
            else:
                room_id = self.room_manager.create_room_from_pose(
                    int(pose_node_id),
                    name=room_name,
                )
            if room_id is None:
                return None, False
            room_created = True

        self.room_manager.associate_room_with_tracker_region(
            int(room_id),
            int(tracker_region_id),
            prepared_region,
            is_bootstrap_region=is_bootstrap_region,
        )
        self._sync_room_members_from_region(
            int(room_id),
            prepared_region,
            prepared_regions,
        )
        self.region_manager.set_current_resolved_region(int(tracker_region_id))
        self.room_manager.build_room_region_signature_set(int(room_id), persist=True)
        return int(room_id), room_created

    def _pipeline_bootstrap_tick(
        self,
        *,
        current_pose_node_id: Optional[int],
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> None:
        """Execute the one-time bootstrap transition."""
        if current_pose_node_id is None:
            self.get_logger().debug(
                "[pipeline_tick] waiting for initial bootstrap pose",
                throttle_duration_sec=5.0,
            )
            return

        (
            resolved_tracker_region_id,
            prepared_region,
            _,
            current_region_owner_id,
        ) = self._resolve_current_tracker_region(
            current_pose_node_id,
            prepared_regions,
        )
        if resolved_tracker_region_id is None or prepared_region is None:
            self.get_logger().debug(
                "[pipeline_tick] waiting for initial bootstrap tracker region",
                throttle_duration_sec=5.0,
            )
            return

        room_id, room_created = self._materialize_room_for_current_region(
            pose_node_id=current_pose_node_id,
            tracker_region_id=int(resolved_tracker_region_id),
            prepared_region=prepared_region,
            prepared_regions=prepared_regions,
            existing_room_id=current_region_owner_id,
            room_name="room_0",
            is_bootstrap_region=True,
        )
        if room_id is None:
            self.get_logger().warning(
                "[pipeline_tick] bootstrap materialization failed"
            )
            return

        self.current_region = int(resolved_tracker_region_id)
        self.current_room_id = int(room_id)
        self.last_non_none_tracker_region_id = int(resolved_tracker_region_id)
        self.semantic_state = SemanticState.EXPLORING_ROOM
        self._record_pipeline_decision(
            transition_anchor_region_id=int(resolved_tracker_region_id),
            resolved_tracker_region_id=int(resolved_tracker_region_id),
            resolved_region_node_id=None,
            current_region_owner_id=int(room_id),
            current_region=int(resolved_tracker_region_id),
            current_room_id=int(room_id),
            region_changed=False,
            room_created=room_created,
            region_created=False,
            decision_outcome="bootstrapped",
        )

    def _pipeline_exploring_tick(
        self,
        *,
        current_pose_node_id: Optional[int],
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> None:
        """Create or reuse the room-region structure for the currently visited region."""
        transition_anchor_region_id = self.last_non_none_tracker_region_id
        (
            resolved_tracker_region_id,
            prepared_region,
            resolved_region_node_id,
            current_region_owner_id,
        ) = self._resolve_current_tracker_region(
            current_pose_node_id,
            prepared_regions,
        )

        if resolved_tracker_region_id is None or prepared_region is None:
            self.current_region = None
            self.current_room_id = None
            self.semantic_state = SemanticState.EXPLORING_ROOM
            self._record_pipeline_decision(
                transition_anchor_region_id=transition_anchor_region_id,
                resolved_tracker_region_id=None,
                resolved_region_node_id=None,
                current_region_owner_id=None,
                current_region=None,
                current_room_id=None,
                region_changed=False,
                room_created=False,
                region_created=False,
                decision_outcome="no_resolved_region",
            )
            return

        region_changed = bool(
            transition_anchor_region_id is not None
            and int(resolved_tracker_region_id) != int(transition_anchor_region_id)
        )
        room_created = False
        region_created = False
        decision_outcome = "reused_existing_region"

        if current_region_owner_id is not None:
            self.room_manager.associate_room_with_tracker_region(
                int(current_region_owner_id),
                int(resolved_tracker_region_id),
                prepared_region,
            )
            self._sync_room_members_from_region(
                int(current_region_owner_id),
                prepared_region,
                prepared_regions,
            )
            self.room_manager.build_room_region_signature_set(
                int(current_region_owner_id),
                persist=True,
            )
        else:
            (
                current_region_owner_id,
                room_created,
            ) = self._materialize_room_for_current_region(
                pose_node_id=current_pose_node_id,
                tracker_region_id=int(resolved_tracker_region_id),
                prepared_region=prepared_region,
                prepared_regions=prepared_regions,
            )
            if current_region_owner_id is None:
                self.current_region = int(resolved_tracker_region_id)
                self.current_room_id = None
                self.semantic_state = SemanticState.EXPLORING_ROOM
                self._record_pipeline_decision(
                    transition_anchor_region_id=transition_anchor_region_id,
                    resolved_tracker_region_id=int(resolved_tracker_region_id),
                    resolved_region_node_id=None,
                    current_region_owner_id=None,
                    current_region=int(resolved_tracker_region_id),
                    current_room_id=None,
                    region_changed=region_changed,
                    room_created=False,
                    region_created=False,
                    decision_outcome="materialization_failed",
                )
                self.last_non_none_tracker_region_id = int(resolved_tracker_region_id)
                return

            if room_created:
                decision_outcome = "created_room"
            else:
                decision_outcome = "materialized_room_anchor"

        self.current_region = int(resolved_tracker_region_id)
        self.current_room_id = int(current_region_owner_id)
        self.semantic_state = SemanticState.EXPLORING_ROOM
        self._record_pipeline_decision(
            transition_anchor_region_id=transition_anchor_region_id,
            resolved_tracker_region_id=int(resolved_tracker_region_id),
            resolved_region_node_id=None,
            current_region_owner_id=int(current_region_owner_id),
            current_region=int(resolved_tracker_region_id),
            current_room_id=int(current_region_owner_id),
            region_changed=region_changed,
            room_created=room_created,
            region_created=region_created,
            decision_outcome=decision_outcome,
        )
        self.last_non_none_tracker_region_id = int(resolved_tracker_region_id)

    def _pipeline_tick(self):
        """Run the direct region-to-room pipeline."""
        start_t = time.perf_counter()
        with self._sg_lock:
            stable_regions_msg = self.latest_stable_regions_msg
            regions_snapshot_is_stale = getattr(
                self, "_stable_regions_snapshot_is_stale", False
            )
            current_pose_node_id = self._get_current_pose_node_id()
            semantic_state = self.semantic_state

        if regions_snapshot_is_stale:
            self.get_logger().warning(
                "[pipeline_tick] skipped region-to-room decision because "
                "stable-region input is stale",
                throttle_duration_sec=5.0,
            )
            return

        snapshot_start_t = time.perf_counter()
        snapshot_valid, prepared_regions = self.region_manager.prepare_region_snapshot(
            stable_regions_msg
        )
        self.profiler.record(
            "region_snapshot_prepare_ms",
            (time.perf_counter() - snapshot_start_t) * 1000.0,
            metadata={"snapshot_valid": bool(snapshot_valid)},
        )

        with self._sg_lock:
            if semantic_state == SemanticState.BOOTSTRAP:
                if not snapshot_valid:
                    self.get_logger().debug(
                        "[pipeline_tick] waiting for initial bootstrap region snapshot",
                        throttle_duration_sec=5.0,
                    )
                    return
                self._pipeline_bootstrap_tick(
                    current_pose_node_id=current_pose_node_id,
                    prepared_regions=prepared_regions,
                )
            else:
                self._pipeline_exploring_tick(
                    current_pose_node_id=current_pose_node_id,
                    prepared_regions=prepared_regions if snapshot_valid else {},
                )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        self.profiler.record(
            "scene_graph_pipeline_tick_ms",
            elapsed_ms,
            metadata={
                "semantic_state": semantic_state.value,
                "snapshot_valid": bool(snapshot_valid),
            },
        )
        warn_ms = float(self._param_dict.get("maintenance_tick_warn_ms", 500.0))
        if elapsed_ms > warn_ms:
            self.get_logger().warning(
                f"pipeline_tick took {elapsed_ms:.1f}ms (> {warn_ms:.1f}ms)"
            )

    def _maintenance_tick(self):
        """Run geometric-only graph maintenance."""
        start_t = time.perf_counter()
        map_update_stats = None
        useful_work = False
        with self._sg_lock:
            self.get_logger().debug(
                "[maintenance_tick] geometric containment and structure only",
                throttle_duration_sec=10.0,
            )

            actionable_nearest_link_work = (
                self.fs_manager.has_processed_map_snapshot()
                and self.fs_manager.has_pending_nearest_link_work()
            )
            has_rooms = bool(self.sg.query.find_nodes_by_type(NodeType.ROOM))
            if (
                not self._map_dirty
                and not actionable_nearest_link_work
                and self.latest_stable_regions_msg is None
                and not has_rooms
            ):
                return
            useful_work = True

            stable_regions_msg = self.latest_stable_regions_msg
            regions_snapshot_is_stale = getattr(
                self, "_stable_regions_snapshot_is_stale", False
            )
            map_msg = self.global_map_msg if self._map_dirty else None
            odom_msg = self.odom_msg
            if map_msg is not None:
                self._map_dirty = False

        if map_msg is not None:
            fs_start_t = time.perf_counter()
            map_update_stats = self.fs_manager.process_occupancy_grid_update(
                map_msg,
                odom_msg,
                frame_id=self._param_dict.get("fixed_frame_id", "odom"),
            )
            self.profiler.record(
                "free_space_manager_update_ms",
                (time.perf_counter() - fs_start_t) * 1000.0,
                metadata={
                    "new_nav_nodes": int(map_update_stats.get("new_nav_nodes", 0)),
                    "deleted_nav_nodes": int(
                        map_update_stats.get("deleted_nav_nodes", 0)
                    ),
                    "total_nav_nodes": int(
                        map_update_stats.get("total_nav_nodes", 0)
                    ),
                },
            )

        full_relink_requested = self.fs_manager.pending_full_relink or (
            map_update_stats is not None
            and bool(map_update_stats.get("full_rescan_required"))
        )
        if full_relink_requested:
            self.fs_manager.rebuild_object_block_index()
            self.fs_manager.pending_full_relink = False
            self.fs_manager.drain_queued_object_ids()

        snapshot_start_t = time.perf_counter()
        snapshot_valid, prepared_regions = self.region_manager.prepare_region_snapshot(
            stable_regions_msg
        )
        self.profiler.record(
            "region_snapshot_prepare_ms",
            (time.perf_counter() - snapshot_start_t) * 1000.0,
            metadata={"snapshot_valid": bool(snapshot_valid)},
        )
        dirty_room_ids = set()
        relinked_tracker_region_ids: dict[int, int] = {}
        region_nav_ids_by_tracker_region: dict[int, set[int]] = {}
        if snapshot_valid and not regions_snapshot_is_stale:
            relinked_rooms = self.room_manager.relink_rooms_to_replacement_regions(
                prepared_regions
            )
            relinked_tracker_region_ids = {
                int(item["old_tracker_region_id"]): int(item["new_tracker_region_id"])
                for item in relinked_rooms
            }
            dirty_room_ids.update(
                int(item["room_node_id"])
                for item in relinked_rooms
                if item.get("room_node_id") is not None
            )

            for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
                if room_node.id is None:
                    continue
                tracker_region_id = self.room_manager.get_tracker_region_id_for_room(
                    int(room_node.id)
                )
                prepared_region = (
                    prepared_regions.get(int(tracker_region_id))
                    if tracker_region_id is not None
                    else None
                )
                if prepared_region is None:
                    continue
                self.room_manager.associate_room_with_tracker_region(
                    int(room_node.id),
                    int(tracker_region_id),
                    prepared_region,
                )
                member_ids = self.region_manager.gather_region_member_ids(
                    prepared_region,
                    prepared_regions=prepared_regions,
                )
                region_nav_ids_by_tracker_region[int(tracker_region_id)] = set(
                    member_ids.get(NodeType.NAVIGATION, set())
                )
                self.room_manager.sync_room_membership_from_region(
                    int(room_node.id),
                    member_ids,
                )
                dirty_room_ids.add(int(room_node.id))

            (
                materialized_room_ids,
                materialized_region_nav_ids,
            ) = self._materialize_rooms_for_observed_regions(prepared_regions)
            dirty_room_ids.update(materialized_room_ids)
            region_nav_ids_by_tracker_region.update(materialized_region_nav_ids)

        if regions_snapshot_is_stale:
            self.get_logger().warning(
                "[maintenance_tick] graph pruning skipped because "
                "stable-region input is stale "
                f"consecutive_empty={self._consecutive_empty_region_updates}",
                throttle_duration_sec=5.0,
            )
            pruned_rooms = []
        else:
            pruned_rooms = (
                # In the region pipeline, direct members are derived attachments and
                # never protect rooms whose DuDe stable-region anchor disappeared.
                self.room_manager.prune_rooms_without_valid_anchors(
                    prepared_regions.keys()
                )
                if snapshot_valid
                else []
            )
        pruned_region_node_ids: Set[int] = set()
        pruned_tracker_region_ids = {
            int(item["stable_region_id"])
            for item in pruned_rooms
            if item.get("stable_region_id") is not None
        }
        pruned_room_ids = {
            int(item["room_node_id"])
            for item in pruned_rooms
            if item.get("room_node_id") is not None
        }

        dirty_room_ids.update(
            room_id
            for room_id in self.room_manager.dirty_room_ids
            if room_id not in pruned_room_ids
        )

        self.room_manager.recompute_room_centroids_from_anchors(dirty_room_ids)
        if not regions_snapshot_is_stale:
            self.room_manager.rebuild_room_adjacency(
                prepared_regions=prepared_regions if snapshot_valid else {},
                dirty_room_ids=dirty_room_ids,
                region_nav_ids_by_tracker_region=region_nav_ids_by_tracker_region,
            )

        if full_relink_requested:
            self.fs_manager.update_nearest_freespace_links_for_objects()
        elif self.fs_manager.has_processed_map_snapshot():
            affected_object_ids = self.fs_manager.drain_queued_object_ids()
            if map_update_stats is not None:
                affected_object_ids.update(
                    map_update_stats.get("removed_linked_object_ids", set())
                )

            object_nodes = []
            for object_id in affected_object_ids:
                node = self.sg.query.get_node(int(object_id))
                if node is not None:
                    object_nodes.append(node)

            if object_nodes:
                self.fs_manager.update_nearest_freespace_links_for_objects(object_nodes)

        with self._sg_lock:
            for (
                old_tracker_region_id,
                new_tracker_region_id,
            ) in relinked_tracker_region_ids.items():
                if self.current_region == old_tracker_region_id:
                    self.current_region = new_tracker_region_id
                if self.last_non_none_tracker_region_id == old_tracker_region_id:
                    self.last_non_none_tracker_region_id = new_tracker_region_id

            self._clear_pruned_runtime_references(
                pruned_region_node_ids=pruned_region_node_ids,
                pruned_tracker_region_ids=pruned_tracker_region_ids,
                pruned_room_ids=pruned_room_ids,
            )
            self.room_manager.dirty_room_ids.difference_update(dirty_room_ids)
            self.room_manager.dirty_room_ids.difference_update(pruned_room_ids)

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        if useful_work:
            self.profiler.record(
                "entity_assignment_graph_assembly_ms",
                elapsed_ms,
                metadata={
                    "map_updated": map_update_stats is not None,
                    "snapshot_valid": bool(snapshot_valid),
                    "dirty_room_count": len(dirty_room_ids),
                    "pruned_room_count": len(pruned_room_ids),
                    "full_relink_requested": bool(full_relink_requested),
                },
            )
        self.profiler.record(
            "scene_graph_maintenance_tick_ms",
            elapsed_ms,
            metadata={"useful_work": bool(useful_work)},
        )
        warn_ms = float(self._param_dict.get("maintenance_tick_warn_ms", 500.0))
        if elapsed_ms > warn_ms:
            self.get_logger().warning(
                f"maintenance_tick took {elapsed_ms:.1f}ms (> {warn_ms:.1f}ms)"
            )


def main(args=None):
    """Create the orchestrator and its visualization node."""
    rclpy.init(args=args)

    orchestrator = None
    executor = None

    try:
        orchestrator = SceneGraphOrchestrator()
        logger.info("SceneGraphOrchestrator created")
        register_export_triggers(orchestrator)

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(orchestrator)
        executor.add_node(orchestrator.visualization_node)

        logger.info("Starting multi-threaded executor...")
        executor.spin()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except ExternalShutdownException:
        logger.info("ROS context shutdown received, shutting down...")
    except Exception as exc:
        logger.error(f"Failed to create or spin SceneGraphOrchestrator: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if orchestrator is not None:
            export_scene_graph_json_if_configured(orchestrator)
            orchestrator.profiler.save()
            orchestrator.visualization_node.destroy_node()
            orchestrator.destroy_node()
        if executor is not None:
            executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
