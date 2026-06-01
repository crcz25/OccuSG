"""
Visualization Node - Efficient scene-graph visualization with RViz markers.

This node snapshots the graph once per timer tick, derives a normalized marker
state, diffs it against the previous publish, and sends only incremental
 marker updates to RViz. Dense homogeneous layers are batched into list markers
 to reduce marker count and publication overhead.
"""

from __future__ import annotations

import colorsys
import math
from collections import defaultdict
from dataclasses import dataclass, field
from time import perf_counter
from typing import DefaultDict, Dict, Iterable, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import EdgeType, NodeType, get_type_scoped_id


IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
BATCH_MARKER_ID = 0

POSE_NS = "pose_layer"
POSE_LABEL_NS = "pose_labels"
OBJECT_NS = "objects_layer"
OBJECT_LABEL_NS = "object_labels"
ROOM_NS = "semantic_layer"
ROOM_LABEL_NS = "room_labels"
ROOM_GEOMETRY_NS = "room_geometry"
NAVIGATION_NS = "navigation_layer"
NAVIGATION_LABEL_NS = "navigation_labels"
POSE_EDGE_NS = "pose_link"
OBSERVATION_EDGE_NS = "pose_object_link"
NAVIGATION_EDGE_NS = "navigation_links"
NEAREST_FREE_SPACE_EDGE_NS = "object_to_free"
ROOM_CONTAINS_EDGE_NS = "room_contains_link"
ROOM_ADJACENCY_EDGE_NS = "room_adjacency_link"

POSE_SCALE = (0.2, 0.2, 0.2)
OBJECT_SCALE = (0.3, 0.3, 0.3)
ROOM_SCALE = (0.5, 0.5, 0.5)
NAVIGATION_SCALE = (0.5, 0.5, 0.2)

POSE_LABEL_SCALE = (0.15, 0.15, 0.15)
OBJECT_LABEL_SCALE = (0.2, 0.2, 0.2)
ROOM_LABEL_SCALE = (0.3, 0.3, 0.3)
NAVIGATION_LABEL_SCALE = (0.15, 0.15, 0.15)

POSE_COLOR = (0.0, 1.0, 0.0, 1.0)
OBJECT_COLOR = (1.0, 0.0, 0.0, 1.0)
NAVIGATION_COLOR = (0.2, 0.8, 1.0, 0.6)

POSE_EDGE_COLOR = (1.0, 0.0, 0.0, 1.0)
OBSERVATION_EDGE_COLOR = (0.0, 0.0, 1.0, 1.0)
NAVIGATION_EDGE_COLOR = (0.2, 0.8, 1.0, 0.5)
NEAREST_FREE_SPACE_EDGE_COLOR = (1.0, 1.0, 0.0, 1.0)
ROOM_ADJACENCY_EDGE_COLOR = (1.0, 1.0, 1.0, 0.6)

POSE_LABEL_OFFSET_Z = 0.3
OBJECT_LABEL_OFFSET_Z = 0.5
ROOM_LABEL_OFFSET_Z = 0.5
NAVIGATION_LABEL_OFFSET_Z = 0.3


@dataclass(frozen=True)
class MarkerKey:
    """Stable key for one RViz marker."""

    namespace: str
    marker_id: int


@dataclass(frozen=True)
class MarkerSpec:
    """Normalized marker state used for diffing without ROS allocations."""

    key: MarkerKey
    marker_type: int
    scale: Tuple[float, float, float]
    color: Tuple[float, float, float, float]
    pose: Tuple[float, float, float, float, float, float, float] = IDENTITY_POSE
    text: str = ""
    points: Tuple[Tuple[float, float, float], ...] = ()
    colors: Tuple[Tuple[float, float, float, float], ...] = ()

    def to_marker(self, *, stamp, frame_id: str, lifetime_msg) -> Marker:
        """Convert the normalized state into a ROS marker message."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = self.key.namespace
        marker.id = self.key.marker_id
        marker.type = self.marker_type
        marker.action = Marker.ADD

        marker.pose.position.x = self.pose[0]
        marker.pose.position.y = self.pose[1]
        marker.pose.position.z = self.pose[2]
        marker.pose.orientation.x = self.pose[3]
        marker.pose.orientation.y = self.pose[4]
        marker.pose.orientation.z = self.pose[5]
        marker.pose.orientation.w = self.pose[6]

        marker.scale.x = self.scale[0]
        marker.scale.y = self.scale[1]
        marker.scale.z = self.scale[2]

        marker.color.r = self.color[0]
        marker.color.g = self.color[1]
        marker.color.b = self.color[2]
        marker.color.a = self.color[3]

        marker.text = self.text
        marker.lifetime = lifetime_msg

        if self.points:
            marker.points = [
                Point(x=point[0], y=point[1], z=point[2]) for point in self.points
            ]
        if self.colors:
            marker.colors = []
            for color in self.colors:
                point_color = ColorRGBA()
                point_color.r = color[0]
                point_color.g = color[1]
                point_color.b = color[2]
                point_color.a = color[3]
                marker.colors.append(point_color)

        return marker


@dataclass(frozen=True)
class GraphSnapshot:
    """One read-only graph snapshot reused across the full visualization tick."""

    node_by_id: Dict[int, object]
    nodes_by_type: Dict[NodeType, Tuple[object, ...]]
    edges_by_type: Dict[EdgeType, Tuple[object, ...]]

    @classmethod
    def from_scene_graph(cls, sg: SceneGraphInterface) -> "GraphSnapshot":
        nodes = sg.query.get_all_nodes()
        edges = sg.query.get_all_edges()

        node_by_id: Dict[int, object] = {}
        nodes_by_type: DefaultDict[NodeType, list] = defaultdict(list)
        edges_by_type: DefaultDict[EdgeType, list] = defaultdict(list)

        for node in nodes:
            node_id = getattr(node, "id", None)
            node_type = getattr(node, "node_type", None)
            if node_id is None or node_type is None:
                continue
            node_by_id[int(node_id)] = node
            nodes_by_type[node_type].append(node)

        for edge in edges:
            edge_type = getattr(edge, "type", None)
            if edge_type is None:
                continue
            edges_by_type[edge_type].append(edge)

        sorted_nodes = {
            node_type: tuple(
                sorted(nodes_for_type, key=lambda item: int(item.id) if item.id is not None else -1)
            )
            for node_type, nodes_for_type in nodes_by_type.items()
        }
        sorted_edges = {
            edge_type: tuple(
                sorted(
                    edges_for_type,
                    key=lambda item: (
                        int(item.id) if item.id is not None else -1,
                        int(item.source_id),
                        int(item.target_id),
                    ),
                )
            )
            for edge_type, edges_for_type in edges_by_type.items()
        }

        return cls(
            node_by_id=node_by_id,
            nodes_by_type=sorted_nodes,
            edges_by_type=sorted_edges,
        )


@dataclass
class RenderContext:
    """Cross-layer selection state shared between node and edge rendering."""

    navigation_render_ids: set[int] = field(default_factory=set)
    navigation_color_by_node_id: Dict[int, Tuple[float, float, float, float]] = field(
        default_factory=dict
    )


class VisualizationNode(Node):
    """
    Visualization node for scene graph marker publishing.

    All reads are performed on the shared SceneGraphInterface (thread-safe).
    """

    def __init__(self, sg_interface: SceneGraphInterface, param_dict: dict = None):
        super().__init__("visualization_node", use_global_arguments=False)

        self.sg = sg_interface
        self._param_dict = param_dict or {}

        self._declare_parameters()
        self._load_parameters()
        self._initialize_state()
        self._create_publishers()
        self._create_timers()

        self.get_logger().info("VisualizationNode initialized successfully")
        self.get_logger().debug(
            f"Viz rate: {self.visualization_hz}Hz, Topic: {self.scene_graph_topic}"
        )

    def _declare_parameters(self):
        """Declare all ROS2 parameters using values from param_dict."""
        self.declare_parameter(
            "scene_graph_topic",
            self._param_dict.get("scene_graph_topic", "/dsg/scene_graph"),
        )
        self.declare_parameter(
            "visualization_hz", self._param_dict.get("visualization_hz", 1.0)
        )
        self.declare_parameter(
            "fixed_frame_id", self._param_dict.get("fixed_frame_id", "world")
        )
        self.declare_parameter("life_time", self._param_dict.get("life_time", 0.0))
        self.declare_parameter(
            "scene_graph_qos_history",
            self._param_dict.get("scene_graph_qos_history", "keep_last"),
        )
        self.declare_parameter(
            "scene_graph_qos_reliability",
            self._param_dict.get("scene_graph_qos_reliability", "reliable"),
        )
        self.declare_parameter(
            "scene_graph_qos_durability",
            self._param_dict.get("scene_graph_qos_durability", "volatile"),
        )
        self.declare_parameter(
            "scene_graph_qos_depth",
            self._param_dict.get("scene_graph_qos_depth", 10),
        )

        self.declare_parameter(
            "enable_pose_markers",
            self._param_dict.get("enable_pose_markers", True),
        )
        self.declare_parameter(
            "enable_pose_labels",
            self._param_dict.get("enable_pose_labels", False),
        )
        self.declare_parameter(
            "enable_object_markers",
            self._param_dict.get("enable_object_markers", True),
        )
        self.declare_parameter(
            "enable_object_labels",
            self._param_dict.get("enable_object_labels", True),
        )
        self.declare_parameter(
            "enable_room_markers",
            self._param_dict.get("enable_room_markers", True),
        )
        self.declare_parameter(
            "enable_room_labels",
            self._param_dict.get("enable_room_labels", True),
        )
        self.declare_parameter(
            "enable_region_markers",
            self._param_dict.get("enable_region_markers", True),
        )
        self.declare_parameter(
            "enable_region_labels",
            self._param_dict.get("enable_region_labels", True),
        )
        self.declare_parameter(
            "enable_navigation_markers",
            self._param_dict.get("enable_navigation_markers", True),
        )
        self.declare_parameter(
            "enable_navigation_labels",
            self._param_dict.get("enable_navigation_labels", False),
        )
        self.declare_parameter(
            "enable_pose_edges",
            self._param_dict.get("enable_pose_edges", True),
        )
        self.declare_parameter(
            "enable_observation_edges",
            self._param_dict.get("enable_observation_edges", True),
        )
        self.declare_parameter(
            "enable_navigation_edges",
            self._param_dict.get("enable_navigation_edges", True),
        )
        self.declare_parameter(
            "enable_region_contains_edges",
            self._param_dict.get("enable_region_contains_edges", True),
        )
        self.declare_parameter(
            "enable_room_region_edges",
            self._param_dict.get("enable_room_region_edges", True),
        )
        self.declare_parameter(
            "enable_room_adjacency_edges",
            self._param_dict.get("enable_room_adjacency_edges", True),
        )
        self.declare_parameter(
            "enable_nearest_freespace_edges",
            self._param_dict.get("enable_nearest_freespace_edges", True),
        )

        self.declare_parameter(
            "pose_marker_stride",
            self._param_dict.get("pose_marker_stride", 1),
        )
        self.declare_parameter(
            "pose_label_stride",
            self._param_dict.get("pose_label_stride", 4),
        )
        self.declare_parameter(
            "pose_edge_stride",
            self._param_dict.get("pose_edge_stride", 1),
        )
        self.declare_parameter(
            "navigation_marker_stride",
            self._param_dict.get("navigation_marker_stride", 1),
        )
        self.declare_parameter(
            "navigation_label_stride",
            self._param_dict.get("navigation_label_stride", 4),
        )
        self.declare_parameter(
            "navigation_edge_stride",
            self._param_dict.get("navigation_edge_stride", 1),
        )
        self.declare_parameter(
            "fs_cell_stride_cells",
            self._param_dict.get("fs_cell_stride_cells", 10),
        )
        self.declare_parameter(
            "fs_min_free_cell_count",
            self._param_dict.get("fs_min_free_cell_count", 50),
        )

        self.declare_parameter(
            "visualization_warn_ms",
            self._param_dict.get("visualization_warn_ms", 250.0),
        )
        self.declare_parameter(
            "visualization_stats_interval",
            self._param_dict.get("visualization_stats_interval", 20),
        )

    def _load_parameters(self):
        """Load parameter values."""
        self.scene_graph_topic = self.get_parameter("scene_graph_topic").value
        self.visualization_hz = self.get_parameter("visualization_hz").value
        self.fixed_frame_id = self.get_parameter("fixed_frame_id").value
        self.life_time = float(self.get_parameter("life_time").value)
        self.scene_graph_qos_history = self.get_parameter("scene_graph_qos_history").value
        self.scene_graph_qos_reliability = self.get_parameter(
            "scene_graph_qos_reliability"
        ).value
        self.scene_graph_qos_durability = self.get_parameter(
            "scene_graph_qos_durability"
        ).value
        self.scene_graph_qos_depth = self.get_parameter("scene_graph_qos_depth").value

        self.enable_pose_markers = bool(self.get_parameter("enable_pose_markers").value)
        self.enable_pose_labels = bool(self.get_parameter("enable_pose_labels").value)
        self.enable_object_markers = bool(
            self.get_parameter("enable_object_markers").value
        )
        self.enable_object_labels = bool(
            self.get_parameter("enable_object_labels").value
        )
        self.enable_room_markers = bool(self.get_parameter("enable_room_markers").value)
        self.enable_room_labels = bool(self.get_parameter("enable_room_labels").value)
        self.enable_region_markers = bool(
            self.get_parameter("enable_region_markers").value
        )
        self.enable_region_labels = bool(
            self.get_parameter("enable_region_labels").value
        )
        self.enable_navigation_markers = bool(
            self.get_parameter("enable_navigation_markers").value
        )
        self.enable_navigation_labels = bool(
            self.get_parameter("enable_navigation_labels").value
        )
        self.enable_pose_edges = bool(self.get_parameter("enable_pose_edges").value)
        self.enable_observation_edges = bool(
            self.get_parameter("enable_observation_edges").value
        )
        self.enable_navigation_edges = bool(
            self.get_parameter("enable_navigation_edges").value
        )
        self.enable_region_contains_edges = bool(
            self.get_parameter("enable_region_contains_edges").value
        )
        self.enable_room_region_edges = bool(
            self.get_parameter("enable_room_region_edges").value
        )
        self.enable_room_adjacency_edges = bool(
            self.get_parameter("enable_room_adjacency_edges").value
        )
        self.enable_nearest_freespace_edges = bool(
            self.get_parameter("enable_nearest_freespace_edges").value
        )

        self.pose_marker_stride = self._get_positive_int_parameter(
            "pose_marker_stride", default=1
        )
        self.pose_label_stride = self._get_positive_int_parameter(
            "pose_label_stride", default=4
        )
        self.pose_edge_stride = self._get_positive_int_parameter(
            "pose_edge_stride", default=1
        )
        self.navigation_marker_stride = self._get_positive_int_parameter(
            "navigation_marker_stride", default=1
        )
        self.navigation_label_stride = self._get_positive_int_parameter(
            "navigation_label_stride", default=4
        )
        self.navigation_edge_stride = self._get_positive_int_parameter(
            "navigation_edge_stride", default=1
        )
        self.fs_cell_stride_cells = self._get_positive_int_parameter(
            "fs_cell_stride_cells", default=10
        )
        self.fs_min_free_cell_count = self._get_positive_int_parameter(
            "fs_min_free_cell_count", default=50
        )
        self.visualization_warn_ms = float(
            self.get_parameter("visualization_warn_ms").value
        )
        self.visualization_stats_interval = self._get_positive_int_parameter(
            "visualization_stats_interval", default=20
        )

        self._lifetime_msg = Duration(seconds=max(0.0, self.life_time)).to_msg()

    def _get_positive_int_parameter(self, name: str, *, default: int) -> int:
        """Return one positive integer parameter, warning when normalization is needed."""
        raw_value = self.get_parameter(name).value
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            self.get_logger().warning(
                f"Invalid {name} '{raw_value}', using {default}",
            )
            return int(default)

        if value < 1:
            self.get_logger().warning(
                f"{name} must be >= 1, using {default} instead of '{raw_value}'",
            )
            return int(default)

        return int(value)

    def _parse_qos_history(self, value: str) -> HistoryPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "keep_all":
            return HistoryPolicy.KEEP_ALL
        return HistoryPolicy.KEEP_LAST

    def _parse_qos_reliability(self, value: str) -> ReliabilityPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "best_effort":
            return ReliabilityPolicy.BEST_EFFORT
        return ReliabilityPolicy.RELIABLE

    def _parse_qos_durability(self, value: str) -> DurabilityPolicy:
        value_norm = str(value).strip().lower()
        if value_norm == "transient_local":
            return DurabilityPolicy.TRANSIENT_LOCAL
        return DurabilityPolicy.VOLATILE

    def _initialize_state(self):
        """Initialize node state variables."""
        self._marker_arr = MarkerArray()
        self._published_state: Dict[MarkerKey, MarkerSpec] = {}
        self._published_once = False

        self.room_colors: Dict[int, Tuple[float, float, float, float]] = {}
        self._color_index = 0

        self._viz_tick_count = 0

        self.get_logger().info("Visualization state initialized")

    def _create_publishers(self):
        """Create all ROS2 publishers."""
        try:
            qos_depth = max(1, int(self.scene_graph_qos_depth))
        except (TypeError, ValueError):
            qos_depth = 10
        qos_profile = QoSProfile(
            history=self._parse_qos_history(self.scene_graph_qos_history),
            reliability=self._parse_qos_reliability(self.scene_graph_qos_reliability),
            durability=self._parse_qos_durability(self.scene_graph_qos_durability),
            depth=qos_depth,
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray, self.scene_graph_topic, qos_profile
        )
        self.get_logger().info("Publishers created")

    def _create_timers(self):
        """Create visualization timer."""
        self.timer_visualization = self.create_timer(
            1.0 / self.visualization_hz, self.visualization_callback
        )
        self.get_logger().info("Visualization timer created")

    def visualization_callback(self):
        """Snapshot the graph, diff desired markers, and publish incremental updates."""
        self._viz_tick_count += 1
        callback_start = perf_counter()
        stamp = self.get_clock().now().to_msg()

        snapshot_start = perf_counter()
        snapshot = GraphSnapshot.from_scene_graph(self.sg)
        snapshot_ms = (perf_counter() - snapshot_start) * 1000.0

        node_start = perf_counter()
        desired_state: Dict[MarkerKey, MarkerSpec] = {}
        render_context = self._append_node_marker_specs(snapshot, desired_state)
        node_ms = (perf_counter() - node_start) * 1000.0

        edge_start = perf_counter()
        self._append_edge_marker_specs(snapshot, desired_state, render_context)
        edge_ms = (perf_counter() - edge_start) * 1000.0

        publish_start = perf_counter()
        publish_stats = self._publish_marker_diff(desired_state, stamp)
        publish_ms = (perf_counter() - publish_start) * 1000.0

        total_ms = (perf_counter() - callback_start) * 1000.0

        self._log_visualization_stats(
            snapshot=snapshot,
            desired_state=desired_state,
            snapshot_ms=snapshot_ms,
            node_ms=node_ms,
            edge_ms=edge_ms,
            publish_ms=publish_ms,
            total_ms=total_ms,
            publish_stats=publish_stats,
        )

        # Release the largest per-tick temporaries as soon as we are done with them.
        del render_context
        del desired_state
        del snapshot
        del publish_stats

    def _append_node_marker_specs(
        self,
        snapshot: GraphSnapshot,
        desired_state: Dict[MarkerKey, MarkerSpec],
    ) -> RenderContext:
        """Populate node and label markers, returning cross-layer render selections."""
        render_context = RenderContext()

        pose_nodes = snapshot.nodes_by_type.get(NodeType.AGENT, ())
        if self.enable_pose_markers:
            decimated_pose_nodes = self._apply_stride(
                pose_nodes, self.pose_marker_stride
            )
            spec = self._build_sphere_list_spec(
                namespace=POSE_NS,
                nodes=decimated_pose_nodes,
                scale=POSE_SCALE,
                color=POSE_COLOR,
            )
            if spec is not None:
                desired_state[spec.key] = spec
        if self.enable_pose_labels:
            for pose_node in self._apply_stride(pose_nodes, self.pose_label_stride):
                spec = self._build_text_spec(
                    namespace=POSE_LABEL_NS,
                    marker_id=self._type_scoped_id(pose_node, NodeType.AGENT),
                    text=f"POSE: {self._type_scoped_id(pose_node, NodeType.AGENT)}",
                    position=self._point_from_node(pose_node, z_offset=POSE_LABEL_OFFSET_Z),
                    scale=POSE_LABEL_SCALE,
                    color=POSE_COLOR,
                )
                desired_state[spec.key] = spec

        object_nodes = snapshot.nodes_by_type.get(NodeType.OBJECT, ())
        if self.enable_object_markers:
            for obj_node in object_nodes:
                marker_id = self._type_scoped_id(obj_node, NodeType.OBJECT)
                spec = self._build_pose_marker_spec(
                    namespace=OBJECT_NS,
                    marker_id=marker_id,
                    marker_type=Marker.CUBE,
                    pose=self._pose_from_node(obj_node),
                    scale=OBJECT_SCALE,
                    color=OBJECT_COLOR,
                )
                desired_state[spec.key] = spec
        if self.enable_object_labels:
            for obj_node in object_nodes:
                marker_id = self._type_scoped_id(obj_node, NodeType.OBJECT)
                class_name = "unknown"
                if getattr(obj_node, "attributes", None):
                    class_name = obj_node.attributes.get("class_name", "unknown")
                spec = self._build_text_spec(
                    namespace=OBJECT_LABEL_NS,
                    marker_id=marker_id,
                    text=f"OBJECT: {class_name}_{marker_id}",
                    position=self._point_from_node(obj_node, z_offset=OBJECT_LABEL_OFFSET_Z),
                    scale=OBJECT_LABEL_SCALE,
                    color=WHITE,
                )
                desired_state[spec.key] = spec

        room_nodes = snapshot.nodes_by_type.get(NodeType.ROOM, ())
        self._prime_room_colors(room_nodes)
        if self.enable_room_markers:
            spec = self._build_sphere_list_spec(
                namespace=ROOM_NS,
                nodes=room_nodes,
                scale=ROOM_SCALE,
                color=WHITE,
                per_node_colors=tuple(
                    self._get_room_color(int(room_node.id))
                    for room_node in room_nodes
                    if room_node.id is not None
                ),
            )
            if spec is not None:
                desired_state[spec.key] = spec
        if self.enable_room_labels:
            for room_node in room_nodes:
                marker_id = self._type_scoped_id(room_node, NodeType.ROOM)
                spec = self._build_text_spec(
                    namespace=ROOM_LABEL_NS,
                    marker_id=marker_id,
                    text=f"ROOM: {marker_id}",
                    position=self._point_from_node(room_node, z_offset=ROOM_LABEL_OFFSET_Z),
                    scale=ROOM_LABEL_SCALE,
                    color=WHITE,
                )
                desired_state[spec.key] = spec

        if self.enable_room_markers:
            for room_node in room_nodes:
                spec = self._build_room_geometry_spec(room_node)
                if spec is not None:
                    desired_state[spec.key] = spec

        navigation_nodes = snapshot.nodes_by_type.get(NodeType.NAVIGATION, ())
        decimated_navigation_nodes = self._apply_stride(
            navigation_nodes, self.navigation_marker_stride
        )
        render_context.navigation_render_ids = {
            int(node.id)
            for node in decimated_navigation_nodes
            if getattr(node, "id", None) is not None
        }
        render_context.navigation_color_by_node_id = self._resolve_navigation_color_by_node_id(
            snapshot
        )
        if self.enable_navigation_markers:
            spec = self._build_sphere_list_spec(
                namespace=NAVIGATION_NS,
                nodes=decimated_navigation_nodes,
                scale=self._resolve_navigation_marker_scale(decimated_navigation_nodes),
                color=WHITE,
                per_node_colors=tuple(
                    self._get_navigation_color_for_node(
                        nav_node,
                        render_context.navigation_color_by_node_id,
                        alpha=NAVIGATION_COLOR[3],
                    )
                    for nav_node in decimated_navigation_nodes
                ),
            )
            if spec is not None:
                desired_state[spec.key] = spec
        if self.enable_navigation_labels:
            for nav_node in self._apply_stride(
                navigation_nodes, self.navigation_label_stride
            ):
                marker_id = self._type_scoped_id(nav_node, NodeType.NAVIGATION)
                spec = self._build_text_spec(
                    namespace=NAVIGATION_LABEL_NS,
                    marker_id=marker_id,
                    text=f"FREE: {marker_id}",
                    position=self._point_from_node(nav_node, z_offset=NAVIGATION_LABEL_OFFSET_Z),
                    scale=NAVIGATION_LABEL_SCALE,
                    color=WHITE,
                )
                desired_state[spec.key] = spec

        return render_context

    def _append_edge_marker_specs(
        self,
        snapshot: GraphSnapshot,
        desired_state: Dict[MarkerKey, MarkerSpec],
        render_context: RenderContext,
    ) -> None:
        """Populate batched edge layers from the already captured graph snapshot."""
        if self.enable_pose_edges:
            spec = self._build_line_list_spec(
                namespace=POSE_EDGE_NS,
                edges=self._apply_stride(
                    snapshot.edges_by_type.get(EdgeType.TEMPORAL_LINK, ()),
                    self.pose_edge_stride,
                ),
                node_by_id=snapshot.node_by_id,
                scale=0.05,
                default_color=POSE_EDGE_COLOR,
            )
            if spec is not None:
                desired_state[spec.key] = spec

        if self.enable_observation_edges:
            spec = self._build_line_list_spec(
                namespace=OBSERVATION_EDGE_NS,
                edges=snapshot.edges_by_type.get(EdgeType.OBSERVATION_ANCHOR, ()),
                node_by_id=snapshot.node_by_id,
                scale=0.01,
                default_color=OBSERVATION_EDGE_COLOR,
            )
            if spec is not None:
                desired_state[spec.key] = spec

        if self.enable_nearest_freespace_edges:
            spec = self._build_line_list_spec(
                namespace=NEAREST_FREE_SPACE_EDGE_NS,
                edges=snapshot.edges_by_type.get(EdgeType.NEAREST_FREE_SPACE, ()),
                node_by_id=snapshot.node_by_id,
                scale=0.02,
                default_color=NEAREST_FREE_SPACE_EDGE_COLOR,
            )
            if spec is not None:
                desired_state[spec.key] = spec

        if self.enable_navigation_edges:
            navigation_edges = snapshot.edges_by_type.get(EdgeType.NAVIGABLE_PATH, ())
            if (
                self.enable_navigation_markers
                and self.navigation_marker_stride > 1
                and render_context.navigation_render_ids
            ):
                navigation_edges = tuple(
                    edge
                    for edge in navigation_edges
                    if int(edge.source_id) in render_context.navigation_render_ids
                    and int(edge.target_id) in render_context.navigation_render_ids
                )
            navigation_edge_entries = []
            for edge in self._apply_stride(navigation_edges, self.navigation_edge_stride):
                source_node = snapshot.node_by_id.get(int(edge.source_id))
                target_node = snapshot.node_by_id.get(int(edge.target_id))
                if (
                    source_node is None
                    or target_node is None
                    or source_node.node_type != NodeType.NAVIGATION
                    or target_node.node_type != NodeType.NAVIGATION
                ):
                    continue
                navigation_edge_entries.append(
                    (
                        edge,
                        self._get_navigation_color_for_node(
                            source_node,
                            render_context.navigation_color_by_node_id,
                            alpha=NAVIGATION_EDGE_COLOR[3],
                        ),
                        self._get_navigation_color_for_node(
                            target_node,
                            render_context.navigation_color_by_node_id,
                            alpha=NAVIGATION_EDGE_COLOR[3],
                        ),
                    )
                )
            spec = self._build_line_list_spec(
                namespace=NAVIGATION_EDGE_NS,
                edges=tuple(edge for edge, _, _ in navigation_edge_entries),
                node_by_id=snapshot.node_by_id,
                scale=0.08,
                default_color=NAVIGATION_EDGE_COLOR,
                edge_endpoint_colors=tuple(
                    ((source_color), (target_color))
                    for _, source_color, target_color in navigation_edge_entries
                ),
            )
            if spec is not None:
                desired_state[spec.key] = spec

        if self.enable_room_region_edges:
            room_contains_entries = []
            for edge in snapshot.edges_by_type.get(EdgeType.ROOM_CONTAINS, ()):
                source_node = snapshot.node_by_id.get(int(edge.source_id))
                target_node = snapshot.node_by_id.get(int(edge.target_id))
                if (
                    source_node is None
                    or target_node is None
                    or source_node.node_type != NodeType.ROOM
                ):
                    continue
                if target_node.node_type not in (
                    NodeType.OBJECT,
                    NodeType.AGENT,
                    NodeType.NAVIGATION,
                ):
                    continue
                edge_color = self._get_room_color(int(source_node.id))
                room_contains_entries.append((edge, edge_color))
            spec = self._build_line_list_spec(
                namespace=ROOM_CONTAINS_EDGE_NS,
                edges=tuple(edge for edge, _ in room_contains_entries),
                node_by_id=snapshot.node_by_id,
                scale=0.05,
                default_color=WHITE,
                edge_colors=tuple(color for _, color in room_contains_entries),
            )
            if spec is not None:
                desired_state[spec.key] = spec

        if self.enable_room_adjacency_edges:
            room_adjacency_entries = []
            seen_pairs = set()
            for edge in snapshot.edges_by_type.get(EdgeType.ROOM_ADJACENCY, ()):
                source_node = snapshot.node_by_id.get(int(edge.source_id))
                target_node = snapshot.node_by_id.get(int(edge.target_id))
                if (
                    source_node is None
                    or target_node is None
                    or source_node.node_type != NodeType.ROOM
                    or target_node.node_type != NodeType.ROOM
                ):
                    continue

                pair = tuple(sorted((int(edge.source_id), int(edge.target_id))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                room_adjacency_entries.append(
                    (edge, self._get_room_color(int(pair[0])))
                )

            spec = self._build_line_list_spec(
                namespace=ROOM_ADJACENCY_EDGE_NS,
                edges=tuple(edge for edge, _ in room_adjacency_entries),
                node_by_id=snapshot.node_by_id,
                scale=0.06,
                default_color=ROOM_ADJACENCY_EDGE_COLOR,
                required_types=(NodeType.ROOM, NodeType.ROOM),
                edge_colors=tuple(color for _, color in room_adjacency_entries),
            )
            if spec is not None:
                desired_state[spec.key] = spec

    def _publish_marker_diff(
        self,
        desired_state: Dict[MarkerKey, MarkerSpec],
        stamp,
    ) -> Dict[str, int | bool]:
        """Publish only the marker updates required to reach ``desired_state``."""
        previous_state = self._published_state
        deleted_keys = sorted(previous_state.keys() - desired_state.keys(), key=self._sort_marker_key)
        changed_keys = sorted(
            (
                key
                for key, spec in desired_state.items()
                if previous_state.get(key) != spec
            ),
            key=self._sort_marker_key,
        )

        markers = MarkerArray()
        added_count = 0
        updated_count = 0
        deleted_count = 0

        if not self._published_once:
            markers.markers.append(self._make_delete_all_marker(stamp))

        for key in deleted_keys:
            markers.markers.append(self._make_delete_marker(key, stamp))
            deleted_count += 1

        for key in changed_keys:
            spec = desired_state[key]
            markers.markers.append(
                spec.to_marker(
                    stamp=stamp,
                    frame_id=self.fixed_frame_id,
                    lifetime_msg=self._lifetime_msg,
                )
            )
            if key in previous_state:
                updated_count += 1
            else:
                added_count += 1

        marker_count = len(markers.markers)
        if marker_count == 0 and self._published_once:
            del deleted_keys
            del changed_keys
            del markers
            return {
                "published": False,
                "marker_count": 0,
                "added": 0,
                "updated": 0,
                "deleted": 0,
            }

        self._marker_arr = markers
        self._marker_publisher.publish(markers)
        self._published_state = dict(desired_state)
        self._published_once = True

        del deleted_keys
        del changed_keys

        return {
            "published": True,
            "marker_count": marker_count,
            "added": added_count,
            "updated": updated_count,
            "deleted": deleted_count,
        }

    def _make_delete_all_marker(self, stamp) -> Marker:
        """Create one startup DELETEALL marker to clear stale RViz state."""
        marker = Marker()
        marker.header.frame_id = self.fixed_frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        return marker

    def _make_delete_marker(self, key: MarkerKey, stamp) -> Marker:
        """Create one targeted DELETE marker for a removed namespace/id pair."""
        marker = Marker()
        marker.header.frame_id = self.fixed_frame_id
        marker.header.stamp = stamp
        marker.ns = key.namespace
        marker.id = key.marker_id
        marker.action = Marker.DELETE
        return marker

    def _build_pose_marker_spec(
        self,
        *,
        namespace: str,
        marker_id: int,
        marker_type: int,
        pose: Tuple[float, float, float, float, float, float, float],
        scale: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
        text: str = "",
    ) -> MarkerSpec:
        return MarkerSpec(
            key=MarkerKey(namespace=namespace, marker_id=marker_id),
            marker_type=marker_type,
            scale=scale,
            color=color,
            pose=pose,
            text=text,
        )

    def _build_text_spec(
        self,
        *,
        namespace: str,
        marker_id: int,
        text: str,
        position: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
    ) -> MarkerSpec:
        return self._build_pose_marker_spec(
            namespace=namespace,
            marker_id=marker_id,
            marker_type=Marker.TEXT_VIEW_FACING,
            pose=(position[0], position[1], position[2], 0.0, 0.0, 0.0, 1.0),
            scale=scale,
            color=color,
            text=text,
        )

    def _build_sphere_list_spec(
        self,
        *,
        namespace: str,
        nodes: Sequence[object],
        scale: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
        per_node_colors: Tuple[Tuple[float, float, float, float], ...] = (),
    ) -> MarkerSpec | None:
        if not nodes:
            return None

        points = tuple(self._point_from_node(node) for node in nodes)
        return MarkerSpec(
            key=MarkerKey(namespace=namespace, marker_id=BATCH_MARKER_ID),
            marker_type=Marker.SPHERE_LIST,
            scale=scale,
            color=color,
            points=points,
            colors=per_node_colors,
        )

    def _build_room_geometry_spec(self, room_node) -> MarkerSpec | None:
        """Build a room footprint outline from geometry mirrored onto the ROOM node."""
        room_id = getattr(room_node, "id", None)
        if room_id is None:
            return None
        attrs = dict(getattr(room_node, "attributes", None) or {})
        polygon = attrs.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return None

        points = []
        z = float(room_node.pose.position.z)
        for raw_point in polygon:
            try:
                if isinstance(raw_point, dict):
                    x = float(raw_point["x"])
                    y = float(raw_point["y"])
                else:
                    x = float(raw_point[0])
                    y = float(raw_point[1])
            except (KeyError, TypeError, ValueError, IndexError):
                return None
            points.append((x, y, z))

        if points[0] != points[-1]:
            points.append(points[0])

        return MarkerSpec(
            key=MarkerKey(
                namespace=ROOM_GEOMETRY_NS,
                marker_id=self._type_scoped_id(room_node, NodeType.ROOM),
            ),
            marker_type=Marker.LINE_STRIP,
            scale=(0.08, 0.0, 0.0),
            color=self._get_room_color(int(room_id)),
            points=tuple(points),
        )

    def _resolve_navigation_marker_scale(
        self, navigation_nodes: Sequence[object]
    ) -> Tuple[float, float, float]:
        """Scale nav markers to the free-space block footprint implied by fs_* params."""
        stride_cells = max(1, int(getattr(self, "fs_cell_stride_cells", 10)))
        min_free_cell_count = max(1, int(getattr(self, "fs_min_free_cell_count", 50)))

        for node in navigation_nodes:
            attributes = getattr(node, "attributes", None) or {}
            bounds = attributes.get("bounds")
            if not isinstance(bounds, dict):
                continue

            try:
                min_x = float(bounds["min_x"])
                max_x = float(bounds["max_x"])
                min_y = float(bounds["min_y"])
                max_y = float(bounds["max_y"])
            except (KeyError, TypeError, ValueError):
                continue

            width = max_x - min_x
            height = max_y - min_y
            if width <= 0.0 or height <= 0.0:
                continue

            cell_width = width / float(stride_cells)
            cell_height = height / float(stride_cells)
            threshold_area = float(min_free_cell_count) * cell_width * cell_height
            if threshold_area <= 0.0:
                continue

            diameter = 2.0 * math.sqrt(threshold_area / math.pi)
            diameter = min(diameter, width, height)
            if diameter > 0.0:
                return (diameter, diameter, diameter)

        return NAVIGATION_SCALE

    def _build_line_list_spec(
        self,
        *,
        namespace: str,
        edges: Sequence[object],
        node_by_id: Dict[int, object],
        scale: float,
        default_color: Tuple[float, float, float, float],
        required_types: Tuple[NodeType, NodeType] | None = None,
        edge_colors: Tuple[Tuple[float, float, float, float], ...] = (),
        edge_endpoint_colors: Tuple[
            Tuple[
                Tuple[float, float, float, float],
                Tuple[float, float, float, float],
            ],
            ...,
        ] = (),
    ) -> MarkerSpec | None:
        if not edges:
            return None

        points = []
        colors = []
        use_per_edge_colors = bool(edge_colors)
        use_per_endpoint_colors = bool(edge_endpoint_colors)

        for index, edge in enumerate(edges):
            source_node = node_by_id.get(int(edge.source_id))
            target_node = node_by_id.get(int(edge.target_id))
            if source_node is None or target_node is None:
                continue
            if required_types is not None and (
                source_node.node_type != required_types[0]
                or target_node.node_type != required_types[1]
            ):
                continue

            points.append(self._point_from_node(source_node))
            points.append(self._point_from_node(target_node))

            if use_per_endpoint_colors:
                source_color, target_color = edge_endpoint_colors[index]
                colors.append(source_color)
                colors.append(target_color)
            elif use_per_edge_colors:
                edge_color = edge_colors[index]
                colors.append(edge_color)
                colors.append(edge_color)

        if not points:
            return None

        return MarkerSpec(
            key=MarkerKey(namespace=namespace, marker_id=BATCH_MARKER_ID),
            marker_type=Marker.LINE_LIST,
            scale=(scale, 0.0, 0.0),
            color=default_color,
            points=tuple(points),
            colors=tuple(colors),
        )

    def _apply_stride(
        self,
        items: Sequence[object],
        stride: int,
    ) -> Tuple[object, ...]:
        """Deterministically decimate a stable sorted sequence."""
        if stride <= 1 or len(items) <= 1:
            return tuple(items)
        return tuple(item for index, item in enumerate(items) if index % stride == 0)

    def _point_from_node(
        self,
        node,
        *,
        z_offset: float = 0.0,
    ) -> Tuple[float, float, float]:
        """Extract one XYZ tuple from a node pose."""
        return (
            float(node.pose.position.x),
            float(node.pose.position.y),
            float(node.pose.position.z + z_offset),
        )

    def _pose_from_node(self, node) -> Tuple[float, float, float, float, float, float, float]:
        """Extract one pose tuple from a node."""
        return (
            float(node.pose.position.x),
            float(node.pose.position.y),
            float(node.pose.position.z),
            float(node.pose.orientation.x),
            float(node.pose.orientation.y),
            float(node.pose.orientation.z),
            float(node.pose.orientation.w),
        )

    def _type_scoped_id(self, node, node_type: NodeType) -> int:
        """Return the type-scoped ID used for stable marker IDs."""
        return get_type_scoped_id(int(node.id), node_type)

    def _sort_marker_key(self, key: MarkerKey) -> Tuple[str, int]:
        return (key.namespace, key.marker_id)

    def _prime_room_colors(self, room_nodes: Iterable[object]) -> None:
        """Assign colors for unseen rooms in deterministic room-id order."""
        for room_node in room_nodes:
            if getattr(room_node, "id", None) is None:
                continue
            self._get_room_color(int(room_node.id))

    def _resolve_navigation_color_by_node_id(
        self,
        snapshot: GraphSnapshot,
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """Resolve one owner-derived color for each NAVIGATION node in the snapshot."""
        room_color_by_nav_id: Dict[int, Tuple[float, float, float, float]] = {}

        for edge in snapshot.edges_by_type.get(EdgeType.ROOM_CONTAINS, ()):
            source_node = snapshot.node_by_id.get(int(edge.source_id))
            target_node = snapshot.node_by_id.get(int(edge.target_id))
            if (
                source_node is None
                or target_node is None
                or source_node.node_type != NodeType.ROOM
                or target_node.node_type != NodeType.NAVIGATION
            ):
                continue
            room_color_by_nav_id.setdefault(
                int(target_node.id),
                self._get_room_color(int(source_node.id)),
            )

        return room_color_by_nav_id

    def _get_navigation_color_for_node(
        self,
        node,
        navigation_color_by_node_id: Dict[int, Tuple[float, float, float, float]],
        *,
        alpha: float,
    ) -> Tuple[float, float, float, float]:
        """Return the owner-derived NAVIGATION color, falling back when unowned."""
        node_id = getattr(node, "id", None)
        if node_id is None:
            return NAVIGATION_COLOR

        return self._with_alpha(
            navigation_color_by_node_id.get(int(node_id), NAVIGATION_COLOR),
            alpha=alpha,
        )

    def _log_visualization_stats(
        self,
        *,
        snapshot: GraphSnapshot,
        desired_state: Dict[MarkerKey, MarkerSpec],
        snapshot_ms: float,
        node_ms: float,
        edge_ms: float,
        publish_ms: float,
        total_ms: float,
        publish_stats: Dict[str, int | bool],
    ) -> None:
        """Emit lightweight timing and counter instrumentation."""
        if total_ms >= float(self.visualization_warn_ms):
            self.get_logger().warning(
                "Visualization callback is slow: "
                f"{total_ms:.1f}ms "
                f"(snapshot={snapshot_ms:.1f}ms, nodes={node_ms:.1f}ms, "
                f"edges={edge_ms:.1f}ms, publish={publish_ms:.1f}ms)",
                throttle_duration_sec=5.0,
            )

        if self._viz_tick_count % self.visualization_stats_interval != 0:
            return

        node_counts = ", ".join(
            f"{node_type.value.lower()}={len(snapshot.nodes_by_type.get(node_type, ()))}"
            for node_type in (NodeType.AGENT, NodeType.OBJECT, NodeType.ROOM, NodeType.NAVIGATION)
        )
        edge_counts = ", ".join(
            f"{edge_type.value.lower()}={len(snapshot.edges_by_type.get(edge_type, ()))}"
            for edge_type in (
                EdgeType.TEMPORAL_LINK,
                EdgeType.OBSERVATION_ANCHOR,
                EdgeType.NAVIGABLE_PATH,
                EdgeType.NEAREST_FREE_SPACE,
                EdgeType.ROOM_CONTAINS,
                EdgeType.ROOM_ADJACENCY,
            )
        )

        self.get_logger().debug(
            "Visualization stats: "
            f"total={total_ms:.1f}ms "
            f"snapshot={snapshot_ms:.1f}ms "
            f"nodes={node_ms:.1f}ms "
            f"edges={edge_ms:.1f}ms "
            f"publish={publish_ms:.1f}ms "
            f"desired_markers={len(desired_state)} "
            f"published={publish_stats['published']} "
            f"marker_msgs={publish_stats['marker_count']} "
            f"added={publish_stats['added']} "
            f"updated={publish_stats['updated']} "
            f"deleted={publish_stats['deleted']} "
            f"nodes[{node_counts}] "
            f"edges[{edge_counts}]"
        )

    def _get_room_color(self, room_id: int) -> Tuple[float, float, float, float]:
        """Get a stable color for one room id."""
        if room_id not in self.room_colors:
            golden_ratio = 0.618033988749895
            hue = (self._color_index * golden_ratio) % 1.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

            if 0.1 <= hue <= 0.2:
                hue = (hue + 0.3) % 1.0
                red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

            self.room_colors[room_id] = (red, green, blue, 0.8)
            self._color_index += 1

        return self.room_colors[room_id]

    def _get_or_assign_distinct_color(
        self,
        cache: Dict[int, Tuple[float, float, float, float]],
        node_id: int,
        *,
        alpha: float = 0.8,
        hue_offset: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """Assign a deterministic, reusable color keyed by node id."""
        if node_id not in cache:
            golden_ratio = 0.618033988749895
            hue = ((int(node_id) * golden_ratio) + float(hue_offset)) % 1.0

            excluded_hue_ranges = (
                (0.0, 0.04),
                (0.10, 0.20),
                (0.30, 0.39),
                (0.50, 0.58),
            )
            for lower, upper in excluded_hue_ranges:
                if lower <= hue <= upper:
                    hue = (upper + 0.08) % 1.0
                    break

            red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            cache[node_id] = (red, green, blue, alpha)

        return cache[node_id]

    def _with_alpha(
        self,
        color: Tuple[float, float, float, float],
        *,
        alpha: float,
    ) -> Tuple[float, float, float, float]:
        """Reuse one RGB tuple with a layer-specific alpha channel."""
        return (color[0], color[1], color[2], alpha)


def main(args=None):
    """Standalone entry point for visualization node (for testing)."""
    from scene_graph_core.graph_interface import create_scene_graph_interface

    rclpy.init(args=args)

    sg = create_scene_graph_interface()
    node = VisualizationNode(sg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
