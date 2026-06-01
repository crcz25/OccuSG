#!/usr/bin/env python3
"""Unit tests for the direct room-anchor region orchestrator."""

import threading
from collections import deque

from geometry_msgs.msg import Point32
from incremental_dude_msgs.msg import Region2D, Region2DArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    EdgeType,
    NavNode,
    NodeType,
    ObjectNode,
    PoseNode,
)
from scene_graph_ros.managers.pose_manager import PoseNodeManager
from scene_graph_ros.managers.region_manager import RegionManager
from scene_graph_ros.managers.room_manager import RoomManager
from scene_graph_ros.scene_graph_region import SceneGraphOrchestrator, SemanticState


class RecordingLogger:
    def __init__(self):
        self.records = []

    def debug(self, msg, *args, **kwargs):
        self.records.append(("debug", str(msg)))

    def info(self, msg, *args, **kwargs):
        self.records.append(("info", str(msg)))

    def warning(self, msg, *args, **kwargs):
        self.records.append(("warning", str(msg)))


class DummyFreeSpaceManager:
    pending_full_relink = False

    def has_processed_map_snapshot(self):
        return False

    def has_pending_nearest_link_work(self):
        return False

    def process_occupancy_grid_update(self, grid_msg, odom_msg, frame_id="odom"):
        del grid_msg, odom_msg, frame_id
        return {}

    def rebuild_object_block_index(self):
        return 0

    def drain_queued_object_ids(self):
        return set()

    def update_nearest_freespace_links_for_objects(self, object_nodes=None):
        del object_nodes
        return 0


def _make_pose_node(x: float, y: float) -> PoseNode:
    node = PoseNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    return node


def _make_object_node(class_name: str, x: float, y: float) -> ObjectNode:
    node = ObjectNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    node.attributes = {"class_name": str(class_name)}
    return node


def _make_nav_node(x: float, y: float) -> NavNode:
    node = NavNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    return node


def _make_region(
    tracker_region_id: int,
    centroid_xy: tuple[float, float],
    polygon_xy: list[tuple[float, float]],
) -> Region2D:
    region = Region2D()
    region.id = int(tracker_region_id)
    region.centroid.x = float(centroid_xy[0])
    region.centroid.y = float(centroid_xy[1])
    region.area = 1.0
    for x, y in polygon_xy:
        point = Point32()
        point.x = float(x)
        point.y = float(y)
        region.polygon.points.append(point)
        region.convex_hull.points.append(point)
    return region


def _make_region_array(*regions: Region2D) -> Region2DArray:
    msg = Region2DArray()
    for region in regions:
        msg.regions.append(region)
    return msg


def _make_orchestrator_double():
    orchestrator = object.__new__(SceneGraphOrchestrator)
    logger = RecordingLogger()
    sg = create_scene_graph_interface()
    orchestrator.get_logger = lambda: logger
    orchestrator._sg_lock = threading.RLock()
    orchestrator.sg = sg
    orchestrator._map_dirty = False
    orchestrator.odom_msg = None
    orchestrator.global_map_msg = None
    orchestrator.latest_stable_regions_msg = None
    orchestrator._param_dict = {
        "fixed_frame_id": "odom",
        "maintenance_tick_warn_ms": 500.0,
    }
    orchestrator.semantic_state = SemanticState.BOOTSTRAP
    orchestrator.current_region = None
    orchestrator.current_room_id = None
    orchestrator.last_non_none_tracker_region_id = None
    orchestrator.pose_manager = PoseNodeManager(
        sg_interface=sg,
        logger=logger,
        pose_window_size=3,
    )
    orchestrator.fs_manager = DummyFreeSpaceManager()
    orchestrator.room_manager = RoomManager(sg_interface=sg, logger=logger, z_offset=12.0)
    orchestrator.region_manager = RegionManager(sg_interface=sg, logger=logger, z_offset=8.0)
    orchestrator._reset_semantic_evidence()
    return orchestrator


def _set_current_pose(orchestrator: SceneGraphOrchestrator, pose_id: int) -> None:
    pose_node = orchestrator.sg.query.get_node(int(pose_id))
    orchestrator.pose_manager.pose_window = deque(
        [pose_node],
        maxlen=orchestrator.pose_manager.pose_window_size,
    )
    orchestrator.pose_manager.curr_pose_node = pose_node


def _bootstrap_region(orchestrator: SceneGraphOrchestrator, pose_id: int) -> None:
    _set_current_pose(orchestrator, pose_id)
    orchestrator.latest_stable_regions_msg = _make_region_array(
        _make_region(
            11,
            centroid_xy=(1.0, 1.0),
            polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
        )
    )
    orchestrator._pipeline_tick()


def test_bootstrap_creates_only_room_0_with_region_anchor_and_direct_pose_member():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))

    _bootstrap_region(orchestrator, pose_id)

    rooms = orchestrator.sg.query.find_nodes_by_type(NodeType.ROOM)
    assert [room.attributes["name"] for room in rooms] == ["room_0"]
    assert orchestrator.sg.query.find_nodes_by_type(NodeType.REGION) == []
    room = rooms[0]
    assert room.attributes["stable_region_id"] == 11
    assert room.attributes["tracker_region_id"] == 11
    assert room.attributes["polygon"]
    assert orchestrator.room_manager.get_room_id_for_direct_member(pose_id) == room.id
    assert orchestrator.latest_semantic_evidence["resolved_region_node_id"] is None
    assert orchestrator.latest_semantic_evidence["region_created"] is False


def test_exploring_new_region_creates_room_anchor_and_geometry_gated_members():
    orchestrator = _make_orchestrator_double()
    pose_a_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    _bootstrap_region(orchestrator, pose_a_id)
    pose_b_id = orchestrator.sg.update.add_node(_make_pose_node(5.0, 1.0))
    inside_object_id = orchestrator.sg.update.add_node(_make_object_node("desk", 5.2, 1.0))
    outside_object_id = orchestrator.sg.update.add_node(_make_object_node("plant", 1.2, 1.0))
    _set_current_pose(orchestrator, pose_b_id)
    orchestrator.latest_stable_regions_msg = _make_region_array(
        _make_region(
            11,
            centroid_xy=(1.0, 1.0),
            polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
        ),
        _make_region(
            22,
            centroid_xy=(5.0, 1.0),
            polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
        ),
    )

    orchestrator._pipeline_tick()

    rooms = orchestrator.sg.query.find_nodes_by_type(NodeType.ROOM)
    current_room = orchestrator.sg.query.get_node(orchestrator.current_room_id)
    assert len(rooms) == 2
    assert current_room.attributes["stable_region_id"] == 22
    assert orchestrator.room_manager.get_room_id_for_direct_member(pose_b_id) == current_room.id
    assert orchestrator.room_manager.get_room_id_for_direct_member(inside_object_id) == current_room.id
    assert orchestrator.room_manager.get_room_id_for_direct_member(outside_object_id) != current_room.id
    assert orchestrator.sg.query.get_all_edges(EdgeType.REGION_CONTAINS) == []


def test_maintenance_keeps_room_with_valid_anchor_when_direct_members_disappear():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    _bootstrap_region(orchestrator, pose_id)
    room_id = orchestrator.current_room_id
    orchestrator.sg.update.remove_node(pose_id)

    orchestrator._maintenance_tick()

    assert orchestrator.sg.query.get_node(room_id) is not None
    assert orchestrator.room_manager.get_attached_direct_member_ids(room_id)[NodeType.AGENT] == set()


def test_maintenance_prunes_room_when_anchor_region_disappears():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    _bootstrap_region(orchestrator, pose_id)
    room_id = orchestrator.current_room_id
    assert room_id is not None
    orchestrator.latest_stable_regions_msg = _make_region_array()

    orchestrator._maintenance_tick()

    assert orchestrator.sg.query.get_node(room_id) is None
    assert orchestrator.room_manager.get_room_id_for_direct_member(pose_id) is None
    assert orchestrator.current_room_id is None
    assert orchestrator.current_region is None
    assert orchestrator.last_non_none_tracker_region_id is None
    assert orchestrator.latest_semantic_evidence["current_room_id"] is None
    assert orchestrator.latest_semantic_evidence["current_region"] is None
    assert orchestrator.latest_semantic_evidence["resolved_tracker_region_id"] is None
    assert orchestrator.latest_semantic_evidence["transition_anchor_region_id"] is None


def test_empty_region_callback_after_valid_snapshot_does_not_prune_rooms():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    _set_current_pose(orchestrator, pose_id)

    valid_regions = _make_region_array(
        _make_region(
            11,
            centroid_xy=(1.0, 1.0),
            polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
        )
    )
    orchestrator._stable_regions_callback(valid_regions)
    orchestrator._pipeline_tick()
    assert len(orchestrator.sg.query.find_nodes_by_type(NodeType.ROOM)) == 1

    orchestrator._stable_regions_callback(_make_region_array())
    orchestrator._maintenance_tick()

    rooms = orchestrator.sg.query.find_nodes_by_type(NodeType.ROOM)
    assert len(rooms) == 1
    assert rooms[0].attributes["stable_region_id"] == 11
    assert any(
        "graph pruning skipped because stable-region input is stale" in message
        for level, message in orchestrator.get_logger().records
        if level == "warning"
    )


def test_maintenance_materializes_observed_region_with_existing_entities():
    orchestrator = _make_orchestrator_double()
    object_id = orchestrator.sg.update.add_node(
        _make_object_node("chair", 5.0, 1.0)
    )
    nav_id = orchestrator.sg.update.add_node(_make_nav_node(5.2, 1.0))
    orchestrator.latest_stable_regions_msg = _make_region_array(
        _make_region(
            22,
            centroid_xy=(5.0, 1.0),
            polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
        )
    )

    orchestrator._maintenance_tick()

    rooms = orchestrator.sg.query.find_nodes_by_type(NodeType.ROOM)
    assert len(rooms) == 1
    room = rooms[0]
    assert room.attributes["stable_region_id"] == 22
    assert room.attributes["tracker_region_id"] == 22
    assert (
        orchestrator.room_manager.get_room_id_for_direct_member(object_id)
        == room.id
    )
    assert (
        orchestrator.room_manager.get_room_id_for_direct_member(nav_id)
        == room.id
    )
    assert any(
        "materialized room for observed region" in message
        for level, message in orchestrator.get_logger().records
        if level == "info"
    )


def test_maintenance_relinks_live_room_when_anchor_region_is_replaced():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    object_id = orchestrator.sg.update.add_node(_make_object_node("chair", 1.5, 1.5))
    _bootstrap_region(orchestrator, pose_id)
    room_id = orchestrator.current_room_id
    orchestrator.room_manager.attach_direct_member_to_room(room_id, object_id)
    orchestrator.latest_stable_regions_msg = _make_region_array(
        _make_region(
            22,
            centroid_xy=(1.05, 1.0),
            polygon_xy=[(-0.1, 0.0), (2.1, 0.0), (2.1, 2.0), (-0.1, 2.0)],
        )
    )

    orchestrator._maintenance_tick()

    room = orchestrator.sg.query.get_node(room_id)
    assert room.attributes["stable_region_id"] == 22
    assert room.attributes["tracker_region_id"] == 22
    assert orchestrator.current_region == 22
    assert orchestrator.room_manager.get_room_id_for_tracker_region(11) is None
    assert orchestrator.room_manager.get_room_id_for_tracker_region(22) == room_id
    assert orchestrator.room_manager.get_room_id_for_direct_member(object_id) == room_id


def test_maintenance_prunes_room_when_anchor_and_direct_members_disappear():
    orchestrator = _make_orchestrator_double()
    pose_id = orchestrator.sg.update.add_node(_make_pose_node(1.0, 1.0))
    _bootstrap_region(orchestrator, pose_id)
    room_id = orchestrator.current_room_id
    orchestrator.sg.update.remove_node(pose_id)
    orchestrator.latest_stable_regions_msg = _make_region_array()

    orchestrator._maintenance_tick()

    assert orchestrator.sg.query.get_node(room_id) is None
