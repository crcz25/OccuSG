#!/usr/bin/env python3
"""Focused visualization tests for room-only semantic place rendering."""

from builtin_interfaces.msg import Time
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import Edge, EdgeType, NavNode, ObjectNode, RoomNode
from scene_graph_ros.visualization_node import GraphSnapshot, VisualizationNode


class _FakeClock:
    class _FakeNow:
        @staticmethod
        def to_msg():
            return Time()

    @staticmethod
    def now():
        return _FakeClock._FakeNow()


class _FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, marker_array):
        self.published.append(marker_array)


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _make_object_node(x: float, y: float) -> ObjectNode:
    node = ObjectNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    node.attributes = {"class_name": "chair"}
    return node


def _make_nav_node(x: float, y: float) -> NavNode:
    node = NavNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    node.attributes = {}
    return node


def _make_room_node(x: float, y: float, z: float = 12.0) -> RoomNode:
    node = RoomNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.position.z = float(z)
    node.pose.orientation.w = 1.0
    node.attributes = {
        "name": "room_0",
        "stable_region_id": 11,
        "tracker_region_id": 11,
        "polygon": [
            {"x": 0.0, "y": 0.0},
            {"x": 2.0, "y": 0.0},
            {"x": 2.0, "y": 2.0},
            {"x": 0.0, "y": 2.0},
        ],
        "centroid": {"x": float(x), "y": float(y)},
    }
    return node


def _make_viz(sg, **overrides):
    viz = object.__new__(VisualizationNode)
    viz.sg = sg
    viz.fixed_frame_id = "world"
    viz._lifetime_msg = Duration(seconds=0.0).to_msg()
    viz._marker_arr = MarkerArray()
    viz._published_state = {}
    viz._published_once = False
    viz.room_colors = {}
    viz._color_index = 0
    viz._viz_tick_count = 0
    viz._marker_publisher = _FakePublisher()
    viz._logger = _FakeLogger()
    viz.get_clock = lambda: _FakeClock()
    viz.get_logger = lambda: viz._logger
    viz.enable_pose_markers = True
    viz.enable_pose_labels = False
    viz.enable_object_markers = True
    viz.enable_object_labels = True
    viz.enable_room_markers = True
    viz.enable_room_labels = True
    viz.enable_region_markers = True
    viz.enable_region_labels = True
    viz.enable_navigation_markers = True
    viz.enable_navigation_labels = False
    viz.enable_pose_edges = True
    viz.enable_observation_edges = True
    viz.enable_navigation_edges = True
    viz.enable_region_contains_edges = True
    viz.enable_room_region_edges = True
    viz.enable_room_adjacency_edges = True
    viz.enable_nearest_freespace_edges = True
    viz.pose_marker_stride = 1
    viz.pose_label_stride = 4
    viz.pose_edge_stride = 1
    viz.navigation_marker_stride = 1
    viz.navigation_label_stride = 4
    viz.navigation_edge_stride = 1
    viz.fs_cell_stride_cells = 10
    viz.fs_min_free_cell_count = 50
    viz.visualization_warn_ms = 250.0
    viz.visualization_stats_interval = 1000
    for key, value in overrides.items():
        setattr(viz, key, value)
    return viz


def test_room_geometry_is_rendered_from_room_attributes_without_region_layer():
    sg = create_scene_graph_interface()
    room_id = sg.update.add_node(_make_room_node(1.0, 1.0))
    viz = _make_viz(sg)

    viz.visualization_callback()

    namespaces = {key.namespace for key in viz._published_state}
    assert "room_geometry" in namespaces
    assert "region_layer" not in namespaces
    assert "region_labels" not in namespaces
    geometry_spec = viz._published_state[
        next(key for key in viz._published_state if key.namespace == "room_geometry")
    ]
    assert geometry_spec.marker_type == Marker.LINE_STRIP
    assert geometry_spec.points[0] == (0.0, 0.0, 12.0)
    assert geometry_spec.points[-1] == geometry_spec.points[0]
    assert room_id is not None


def test_direct_room_contains_edges_render_to_objects_and_navigation():
    sg = create_scene_graph_interface()
    room_id = sg.update.add_node(_make_room_node(1.0, 1.0))
    object_id = sg.update.add_node(_make_object_node(1.5, 1.5))
    nav_id = sg.update.add_node(_make_nav_node(1.0, 0.5))
    sg.update.add_edge(Edge(source_id=room_id, target_id=object_id, type=EdgeType.ROOM_CONTAINS))
    sg.update.add_edge(Edge(source_id=room_id, target_id=nav_id, type=EdgeType.ROOM_CONTAINS))
    viz = _make_viz(sg)
    desired_state = {}
    snapshot = GraphSnapshot.from_scene_graph(sg)
    render_context = viz._append_node_marker_specs(snapshot, desired_state)

    viz._append_edge_marker_specs(snapshot, desired_state, render_context)

    edge_spec = desired_state[
        next(key for key in desired_state if key.namespace == "room_contains_link")
    ]
    assert len(edge_spec.points) == 4


def test_region_contains_edges_are_not_rendered_even_if_legacy_edges_exist():
    sg = create_scene_graph_interface()
    room_id = sg.update.add_node(_make_room_node(1.0, 1.0))
    object_id = sg.update.add_node(_make_object_node(1.5, 1.5))
    sg.update.add_edge(Edge(source_id=room_id, target_id=object_id, type=EdgeType.REGION_CONTAINS))
    viz = _make_viz(sg)
    desired_state = {}
    snapshot = GraphSnapshot.from_scene_graph(sg)
    render_context = viz._append_node_marker_specs(snapshot, desired_state)

    viz._append_edge_marker_specs(snapshot, desired_state, render_context)

    assert "region_contains_link" not in {key.namespace for key in desired_state}


def test_navigation_nodes_use_direct_room_owner_color():
    sg = create_scene_graph_interface()
    room_id = sg.update.add_node(_make_room_node(1.0, 1.0))
    nav_id = sg.update.add_node(_make_nav_node(1.0, 0.5))
    sg.update.add_edge(Edge(source_id=room_id, target_id=nav_id, type=EdgeType.ROOM_CONTAINS))
    viz = _make_viz(sg)
    desired_state = {}
    snapshot = GraphSnapshot.from_scene_graph(sg)

    viz._append_node_marker_specs(snapshot, desired_state)

    marker_spec = desired_state[
        next(key for key in desired_state if key.namespace == "navigation_layer")
    ]
    expected_color = VisualizationNode._with_alpha(
        viz,
        VisualizationNode._get_room_color(viz, int(room_id)),
        alpha=0.6,
    )
    assert marker_spec.colors == (expected_color,)
