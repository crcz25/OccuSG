#!/usr/bin/env python3
"""Unit tests for pose-window semantic evidence helpers."""

from collections import deque

from nav_msgs.msg import Odometry
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import NodeType
from scene_graph_core.representation import ObjectNode, PoseNode
from scene_graph_ros.managers.pose_manager import PoseNodeManager


class MockLogger:
    """Captures log messages for tests."""

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass


def _make_object_node(
    class_name: str | None,
    x: float,
    y: float,
    z: float = 0.0,
) -> ObjectNode:
    node = ObjectNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.position.z = float(z)
    node.pose.orientation.w = 1.0
    node.attributes = {}
    if class_name is not None:
        node.attributes["class_name"] = str(class_name)
    return node


def _make_pose_node(object_ids: list[int]) -> PoseNode:
    node = PoseNode()
    node.pose.orientation.w = 1.0
    node.attributes = {
        "object_in_los": list(object_ids),
    }
    return node


def _make_odom(
    x: float,
    y: float,
    *,
    stamp_sec: int,
    stamp_nanosec: int = 0,
    frame_id: str = "odom",
    child_frame_id: str = "base_link",
) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int(stamp_nanosec)
    msg.child_frame_id = child_frame_id
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.w = 1.0
    return msg


def test_build_pose_window_class_set_uses_class_presence_across_window():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_window_size=3,
    )

    chair_a_id = sg.update.add_node(_make_object_node("chair", 0.0, 0.0))
    chair_b_id = sg.update.add_node(_make_object_node("chair", 1.0, 0.0))
    table_id = sg.update.add_node(_make_object_node("table", 2.0, 0.0))
    unlabeled_id = sg.update.add_node(_make_object_node(None, 3.0, 0.0))

    manager.pose_window = deque(
        [
            _make_pose_node([chair_a_id, table_id, unlabeled_id]),
            _make_pose_node([chair_a_id, chair_b_id]),
            _make_pose_node([chair_b_id, table_id]),
        ],
        maxlen=manager.pose_window_size,
    )

    assert manager.get_pose_window_los_object_ids() == {
        chair_a_id,
        chair_b_id,
        table_id,
        unlabeled_id,
    }
    assert manager.build_pose_window_class_set() == {"chair", "table"}


def test_build_pose_window_class_set_can_filter_to_specific_recent_poses():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_window_size=3,
    )

    chair_id = sg.update.add_node(_make_object_node("chair", 0.0, 0.0))
    table_id = sg.update.add_node(_make_object_node("table", 1.0, 0.0))
    lamp_id = sg.update.add_node(_make_object_node("lamp", 2.0, 0.0))

    pose_a_id = sg.update.add_node(_make_pose_node([chair_id, table_id]))
    pose_b_id = sg.update.add_node(_make_pose_node([chair_id, lamp_id]))

    manager.pose_window = deque(
        [
            sg.query.get_node(int(pose_a_id)),
            sg.query.get_node(int(pose_b_id)),
        ],
        maxlen=manager.pose_window_size,
    )

    assert manager.get_pose_window_los_object_ids_for_pose_ids({pose_b_id}) == {
        chair_id,
        lamp_id,
    }
    assert manager.build_pose_window_class_set({pose_b_id}) == {"chair", "lamp"}


def test_continuous_valid_odometry_queue_creates_pose_nodes_over_time():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_distance_threshold=1.0,
        pose_time_threshold=10.0,
        pending_queue_size=10,
    )

    for idx in range(6):
        manager.enqueue_odometry_update(_make_odom(float(idx), 0.0, stamp_sec=idx))

    added = manager.drain_pending_odometry()

    assert len(added) == 6
    assert manager.pending_count() == 0
    assert len(sg.query.find_nodes_by_type(NodeType.AGENT)) == 6
    assert manager.get_statistics()["total_poses_created"] == 6


def test_stationary_robot_does_not_create_unbounded_duplicates():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_distance_threshold=1.0,
        pose_time_threshold=10.0,
    )

    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=0))
    for stamp in range(1, 5):
        manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=stamp))

    added = manager.drain_pending_odometry()

    assert len(added) == 1
    assert len(sg.query.find_nodes_by_type(NodeType.AGENT)) == 1
    stats = manager.get_statistics()
    assert stats["poses_rejected"] == 4
    assert stats["rejected_by_reason"]["below_thresholds"] == 4


def test_stationary_robot_creates_time_threshold_samples():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_distance_threshold=1.0,
        pose_time_threshold=2.0,
    )

    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=0))
    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=1))
    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=2))
    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=3))
    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=4))

    added = manager.drain_pending_odometry()

    assert [node.created_at for node in added] == [0.0, 2.0, 4.0]
    assert len(sg.query.find_nodes_by_type(NodeType.AGENT)) == 3


def test_non_monotonic_timestamp_is_rejected_unless_motion_threshold_passes():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_distance_threshold=1.0,
        pose_time_threshold=1.0,
    )

    manager.enqueue_odometry_update(_make_odom(0.0, 0.0, stamp_sec=10))
    manager.enqueue_odometry_update(_make_odom(0.1, 0.0, stamp_sec=9))
    manager.enqueue_odometry_update(_make_odom(2.0, 0.0, stamp_sec=8))

    added = manager.drain_pending_odometry()

    assert len(added) == 2
    stats = manager.get_statistics()
    assert stats["rejected_by_reason"]["non_monotonic_timestamp"] == 1
    assert [node.created_at for node in added] == [10.0, 8.0]


def test_pending_pose_queue_is_bounded_and_drops_oldest_messages():
    sg = create_scene_graph_interface()
    manager = PoseNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        pose_distance_threshold=1.0,
        pose_time_threshold=10.0,
        pending_queue_size=3,
    )

    for idx in range(5):
        manager.enqueue_odometry_update(_make_odom(float(idx), 0.0, stamp_sec=idx))

    added = manager.drain_pending_odometry()

    assert [node.created_at for node in added] == [2.0, 3.0, 4.0]
    assert manager.get_statistics()["dropped_pending_poses"] == 2
