#!/usr/bin/env python3
"""Unit tests for class-aware object merge behavior."""

import math

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import NodeType, ObjectNode
from scene_graph_ros.managers.object_manager import ObjectNodeManager
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


class MockLogger:
    """Captures log messages for tests."""

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def warn(self, msg, *args, **kwargs):
        pass


def _make_manager(*, spatial_merge_threshold: float = 0.75) -> ObjectNodeManager:
    return ObjectNodeManager(
        sg_interface=create_scene_graph_interface(),
        logger=MockLogger(),
        spatial_merge_threshold=spatial_merge_threshold,
        enable_debug_logging=False,
    )


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
        node.attributes["class_name"] = class_name
    return node


def _make_detection_array(
    x: float,
    y: float,
    *,
    z: float = 0.0,
    class_name: str = "chair",
    score: float = 0.95,
    frame_id: str = "odom",
) -> Detection3DArray:
    msg = Detection3DArray()
    msg.header.frame_id = frame_id

    detection = Detection3D()
    detection.id = class_name
    detection.bbox.center.position.x = float(x)
    detection.bbox.center.position.y = float(y)
    detection.bbox.center.position.z = float(z)
    detection.bbox.center.orientation.w = 1.0
    detection.bbox.size.x = 0.3
    detection.bbox.size.y = 0.3
    detection.bbox.size.z = 0.3

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_name
    hypothesis.hypothesis.score = float(score)
    detection.results.append(hypothesis)

    msg.detections.append(detection)
    return msg


def test_find_or_add_merges_nearest_compatible_same_class_candidate():
    manager = _make_manager(spatial_merge_threshold=0.5)

    incompatible = _make_object_node("table", 0.05, 0.0)
    incompatible.id = manager.sg.update.add_node(incompatible)

    compatible = _make_object_node("Chair", 0.20, 0.0)
    compatible.id = manager.sg.update.add_node(compatible)

    incoming = _make_object_node(" chair ", 0.0, 0.0)

    merged_node, is_new = manager._find_or_add(incoming, range_m=0.5)

    assert is_new is False
    assert merged_node.id == compatible.id
    assert manager.stats["total_objects_merged"] == 1
    assert manager.sg.query.graph.node_count() == 2


def test_find_or_add_does_not_merge_nearby_different_class():
    manager = _make_manager(spatial_merge_threshold=0.5)

    existing = _make_object_node("chair", 0.30, 0.0)
    existing.id = manager.sg.update.add_node(existing)

    incoming = _make_object_node("table", 0.0, 0.0)

    result_node, is_new = manager._find_or_add(incoming, range_m=0.5)

    assert is_new is True
    assert result_node.id != existing.id
    assert manager.stats["total_objects_merged"] == 0
    assert manager.sg.query.graph.node_count() == 2


def test_find_or_add_suppresses_same_pose_different_class_duplicate():
    manager = _make_manager(spatial_merge_threshold=0.5)

    existing = _make_object_node("chair", 0.05, 0.0)
    existing.id = manager.sg.update.add_node(existing)

    incoming = _make_object_node("airplane", 0.0, 0.0)

    result_node, is_new = manager._find_or_add(incoming, range_m=0.5)

    assert is_new is False
    assert result_node.id == existing.id
    assert manager.stats["total_objects_merged"] == 0
    assert manager.sg.query.graph.node_count() == 1


def test_find_or_add_does_not_merge_same_class_when_far_apart():
    manager = _make_manager(spatial_merge_threshold=0.5)

    existing = _make_object_node("chair", 2.0, 0.0)
    existing.id = manager.sg.update.add_node(existing)

    incoming = _make_object_node("chair", 0.0, 0.0)

    result_node, is_new = manager._find_or_add(incoming, range_m=0.5)

    assert is_new is True
    assert result_node.id != existing.id
    assert manager.stats["total_objects_merged"] == 0
    assert manager.sg.query.graph.node_count() == 2


def test_find_or_add_keeps_unlabeled_objects_distinct():
    manager = _make_manager(spatial_merge_threshold=0.5)

    unlabeled = _make_object_node(None, 0.05, 0.0)
    unlabeled.id = manager.sg.update.add_node(unlabeled)

    incoming_unlabeled = _make_object_node(None, 0.0, 0.0)
    unlabeled_result, unlabeled_is_new = manager._find_or_add(
        incoming_unlabeled, range_m=0.5
    )

    incoming_labeled = _make_object_node("chair", 0.0, 0.0)
    labeled_result, labeled_is_new = manager._find_or_add(
        incoming_labeled, range_m=0.5
    )

    assert unlabeled_is_new is False
    assert unlabeled_result.id == unlabeled.id
    assert labeled_is_new is False
    assert labeled_result.id == unlabeled.id
    assert manager.stats["total_objects_merged"] == 0
    assert manager.sg.query.graph.node_count() == 1


def test_continuous_valid_detections_create_objects_over_time():
    manager = _make_manager(spatial_merge_threshold=0.5)

    for x in (0.0, 1.0, 2.0, 3.0):
        stats = manager.process_detections_update(
            _make_detection_array(x, 0.0),
            tf_buffer=None,
            fixed_frame_id="odom",
        )
        assert stats["new_objects"] == 1
        assert stats["rejected_detections"] == 0

    assert len(manager.sg.query.find_nodes_by_type(NodeType.OBJECT)) == 4
    assert manager.stats["total_detections_accepted"] == 4


def test_repeated_detections_update_existing_object_with_diagnostics():
    manager = _make_manager(spatial_merge_threshold=0.5)

    first = manager.process_detections_update(
        _make_detection_array(0.0, 0.0),
        tf_buffer=None,
        fixed_frame_id="odom",
    )
    second = manager.process_detections_update(
        _make_detection_array(0.1, 0.0),
        tf_buffer=None,
        fixed_frame_id="odom",
    )

    assert first["new_objects"] == 1
    assert second["new_objects"] == 0
    assert second["updated_objects"] == 1
    assert second["updated_object_ids"] == first["new_object_ids"]
    assert len(manager.sg.query.find_nodes_by_type(NodeType.OBJECT)) == 1


def test_invalid_detection_is_rejected_and_future_detection_still_creates_object():
    manager = _make_manager(spatial_merge_threshold=0.5)

    invalid = _make_detection_array(math.nan, 0.0)
    invalid_stats = manager.process_detections_update(
        invalid,
        tf_buffer=None,
        fixed_frame_id="odom",
    )

    valid_stats = manager.process_detections_update(
        _make_detection_array(1.0, 0.0),
        tf_buffer=None,
        fixed_frame_id="odom",
    )

    assert invalid_stats["new_objects"] == 0
    assert invalid_stats["rejected_detections"] == 1
    assert invalid_stats["rejected_by_reason"] == {"nonfinite_bbox_center": 1}
    assert valid_stats["new_objects"] == 1
    assert len(manager.sg.query.find_nodes_by_type(NodeType.OBJECT)) == 1
