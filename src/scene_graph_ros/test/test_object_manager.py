#!/usr/bin/env python3
"""Unit tests for semantic-perception object association and node state."""

import math

import numpy as np
import pytest
from semantic_perception_msgs.msg import ObjectProposal3D, ObjectProposal3DArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    Edge,
    EdgeType,
    NodeType,
    PoseNode,
    RoomNode,
)
from scene_graph_core.representation.object_schema import object_group
from scene_graph_ros.managers.object_manager import ObjectNodeManager
from scene_graph_ros.managers.room_manager import RoomManager


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

    def error(self, msg, *args, **kwargs):
        pass


# Orthogonal directions give unambiguous cosine outcomes. The manager only ever
# sees the fused vector, so these stand in for concat(mask, bbox, label).
CHAIR_EMBEDDING = (1.0, 0.0, 0.0)
TABLE_EMBEDDING = (0.0, 1.0, 0.0)
LAMP_EMBEDDING = (0.0, 0.0, 1.0)


def _make_manager(
    *,
    sg=None,
    spatial_association_distance: float = 0.75,
    semantic_similarity_threshold: float = 0.70,
    position_update_policy: str = "running_mean",
    room_manager=None,
    room_resolver=None,
) -> ObjectNodeManager:
    return ObjectNodeManager(
        sg_interface=sg if sg is not None else create_scene_graph_interface(),
        logger=MockLogger(),
        spatial_association_distance=spatial_association_distance,
        semantic_similarity_threshold=semantic_similarity_threshold,
        position_update_policy=position_update_policy,
        room_manager=room_manager,
        room_resolver=room_resolver,
        enable_debug_logging=False,
    )


def _make_proposal(
    x: float,
    y: float,
    *,
    z: float = 0.0,
    embedding=CHAIR_EMBEDDING,
    class_name: str = "chair",
    confidence: float = 0.9,
    valid_3d: bool = True,
    proposal_id: int = 0,
    label_embedding=(0.0, 0.0, 1.0),
) -> ObjectProposal3D:
    proposal = ObjectProposal3D()
    proposal.id = int(proposal_id)
    proposal.class_name = class_name
    proposal.class_id = 7
    proposal.detection_confidence = float(confidence)
    proposal.detector_source = "groundingdino"
    proposal.valid_3d = bool(valid_3d)
    proposal.centroid_3d.x = float(x)
    proposal.centroid_3d.y = float(y)
    proposal.centroid_3d.z = float(z)
    proposal.bbox_3d.size.x = 0.3
    proposal.bbox_3d.size.y = 0.3
    proposal.bbox_3d.size.z = 0.3
    proposal.mask_embedding = [1.0, 0.0, 0.0]
    proposal.bbox_embedding = [0.0, 1.0, 0.0]
    if label_embedding is not None:
        proposal.label_embedding = [float(value) for value in label_embedding]
    if embedding is not None:
        proposal.fused_embedding = [float(value) for value in embedding]
    return proposal


def _make_proposal_array(
    *proposals: ObjectProposal3D,
    frame_id: str = "odom",
    stamp_sec: int = 100,
    stamp_nanosec: int = 0,
) -> ObjectProposal3DArray:
    msg = ObjectProposal3DArray()
    msg.header.frame_id = frame_id
    msg.header.stamp.sec = int(stamp_sec)
    msg.header.stamp.nanosec = int(stamp_nanosec)
    for proposal in proposals:
        msg.proposals.append(proposal)
    return msg


def _apply(manager: ObjectNodeManager, msg, *, room_id=None) -> dict:
    return manager.process_detections_update(
        msg, fixed_frame_id="odom", room_id=room_id
    )


def _object_nodes(manager: ObjectNodeManager):
    return manager.sg.query.find_nodes_by_type(NodeType.OBJECT)


def _make_room(sg, x: float = 0.0, y: float = 0.0, name: str = "room_0") -> int:
    room = RoomNode(attributes={"name": name})
    room.pose.position.x = float(x)
    room.pose.position.y = float(y)
    room.pose.orientation.w = 1.0
    return int(sg.update.add_node(room))


# ========== no nearby node creates one node ==========


def test_no_nearby_node_creates_one_object():
    manager = _make_manager()

    stats = _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))

    assert stats["new_objects"] == 1
    assert stats["updated_objects"] == 0
    assert stats["ambiguous_detections"] == 0
    assert len(_object_nodes(manager)) == 1

    node = _object_nodes(manager)[0]
    assert set(node.attributes) == {
        "geometry",
        "semantic",
        "detection",
        "embeddings",
        "observations",
    }
    assert not any(
        key in node.attributes
        for key in (
            "class_name",
            "detection_confidence",
            "detection_observation_count",
            "first_seen",
            "embedding_sum",
        )
    )
    assert object_group(node.attributes, "semantic")["class_name"] == "chair"
    assert object_group(node.attributes, "observations")["detection_observation_count"] == 1
    assert object_group(node.attributes, "observations")["embedding_observation_count"] == 1
    assert object_group(node.attributes, "embeddings")["object_embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert object_group(node.attributes, "embeddings")["label_embedding"] == pytest.approx([0.0, 0.0, 1.0])
    assert object_group(node.attributes, "embeddings")["mask_embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert object_group(node.attributes, "embeddings")["bbox_embedding"] == pytest.approx([0.0, 1.0, 0.0])
    assert node.created_at == pytest.approx(100.0)


def test_distant_proposals_create_separate_objects():
    manager = _make_manager(spatial_association_distance=0.75)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager, _make_proposal_array(_make_proposal(5.0, 1.0), stamp_sec=101)
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


# ========== compatible nearby node is updated ==========


def test_compatible_nearby_node_is_updated():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101)
    )

    assert stats["updated_objects"] == 1
    assert stats["new_objects"] == 0
    assert len(_object_nodes(manager)) == 1

    node = _object_nodes(manager)[0]
    assert object_group(node.attributes, "observations")["detection_observation_count"] == 2
    assert object_group(node.attributes, "observations")["embedding_observation_count"] == 2
    assert math.isclose(object_group(node.attributes, "observations")["last_semantic_similarity"], 1.0, rel_tol=1e-6)
    assert node.last_seen == pytest.approx(101.0)
    assert node.created_at == pytest.approx(100.0)


def test_repeated_observations_update_a_single_node():
    manager = _make_manager()

    for index in range(6):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.0 + 0.02 * index, 1.0), stamp_sec=100 + index
            ),
        )

    assert len(_object_nodes(manager)) == 1
    assert object_group(_object_nodes(manager)[0].attributes, "observations")["detection_observation_count"] == 6


# ========== best semantic match is selected ==========


def test_best_semantic_match_is_selected():
    manager = _make_manager(semantic_similarity_threshold=0.5)

    # Seed two nodes far enough apart that neither observation is ambiguous.
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(2.0, 1.0, embedding=TABLE_EMBEDDING, class_name="table"),
            stamp_sec=101,
        ),
    )
    assert len(_object_nodes(manager)) == 2

    chair_id = min(node.id for node in _object_nodes(manager))
    table_id = max(node.id for node in _object_nodes(manager))

    # Sits between both nodes and closer to the table, but is semantically a chair.
    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.6, 1.0), stamp_sec=102)
    )

    assert stats["updated_object_ids"] == [chair_id]
    node = manager.sg.query.get_node(table_id)
    assert object_group(node.attributes, "observations")["detection_observation_count"] == 1


def test_equal_similarity_breaks_on_distance():
    manager = _make_manager(semantic_similarity_threshold=0.5)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, embedding=TABLE_EMBEDDING, class_name="table"),
            stamp_sec=101,
        ),
    )
    far_node = max(_object_nodes(manager), key=lambda node: node.id)
    far_node.pose.position.x = 1.4
    manager.sg.update.update_node(far_node.id, far_node)
    near_id = min(node.id for node in _object_nodes(manager))

    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=(1.0, 1.0, 0.0)), stamp_sec=102
        ),
    )

    assert stats["updated_object_ids"] == [near_id]


# ========== spatial ambiguity ==========


def test_incompatible_nearby_node_produces_ambiguity_without_creating_a_node():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING, class_name="table"),
            stamp_sec=101,
        ),
    )

    # Semantic disagreement is not evidence of a second physical object here.
    assert stats["new_objects"] == 0
    assert stats["updated_objects"] == 0
    assert stats["ambiguous_detections"] == 1
    assert stats["rejected_by_reason"] == {"spatially_ambiguous": 1}
    assert len(_object_nodes(manager)) == 1


def test_ambiguity_counter_increments():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    for index in range(3):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING),
                stamp_sec=101 + index,
            ),
        )

    assert manager.get_statistics()["detections_spatially_ambiguous"] == 3
    assert manager.get_statistics()["objects_created"] == 1


def test_candidate_without_a_usable_embedding_still_blocks_creation():
    manager = _make_manager()
    sg = manager.sg

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    node = _object_nodes(manager)[0]
    object_group(node.attributes, "embeddings", create=True)["object_embedding"] = None
    sg.update.update_node(node.id, node)

    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.05, 1.0), stamp_sec=101)
    )

    assert stats["ambiguous_detections"] == 1
    assert len(_object_nodes(manager)) == 1


def test_dimension_mismatch_candidate_is_ambiguous_not_a_new_node():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.02, 1.0, embedding=(1.0, 0.0)), stamp_sec=101
        ),
    )

    assert stats["ambiguous_detections"] == 1
    assert len(_object_nodes(manager)) == 1


# ========== rejections ==========


@pytest.mark.parametrize(
    "kwargs,reason,counter",
    [
        (
            {"confidence": float("nan")},
            "nonfinite_detection_confidence",
            "rejected_invalid_geometry",
        ),
        ({"valid_3d": False}, "invalid_3d_geometry", "rejected_invalid_geometry"),
        ({"x": float("inf")}, "nonfinite_centroid", "rejected_invalid_geometry"),
        (
            {"embedding": (0.0, 0.0, 0.0)},
            "invalid_fused_embedding",
            "rejected_invalid_embedding",
        ),
        ({"embedding": None}, "invalid_fused_embedding", "rejected_invalid_embedding"),
        (
            {"class_name": ""},
            "missing_class_name",
            "rejected_invalid_embedding",
        ),
    ],
)
def test_invalid_detections_create_no_node(kwargs, reason, counter):
    manager = _make_manager()
    x = kwargs.pop("x", 1.0)

    stats = _apply(manager, _make_proposal_array(_make_proposal(x, 1.0, **kwargs)))

    assert stats["accepted_detections"] == 0
    assert stats["rejected_by_reason"] == {reason: 1}
    assert _object_nodes(manager) == []
    assert manager.get_statistics()[counter] == 1


def test_frame_mismatch_rejects_the_whole_message():
    manager = _make_manager()

    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.0, 1.0), frame_id="camera_optical_frame"),
    )

    assert stats["rejected_by_reason"] == {"frame_mismatch": 1}
    assert _object_nodes(manager) == []
    assert manager.get_statistics()["rejected_missing_tf"] == 1


def test_empty_and_none_messages_are_handled():
    manager = _make_manager()
    assert _apply(manager, _make_proposal_array())["new_objects"] == 0
    assert _apply(manager, None)["new_objects"] == 0
    assert _object_nodes(manager) == []


def test_one_bad_proposal_does_not_block_the_batch():
    manager = _make_manager()

    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, valid_3d=False),
            _make_proposal(3.0, 3.0, embedding=TABLE_EMBEDDING),
        ),
    )

    assert stats["accepted_detections"] == 1
    assert len(_object_nodes(manager)) == 1


# ========== node state ==========


def test_object_embedding_is_the_normalized_running_mean_of_the_sum():
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING), stamp_sec=101
        ),
    )

    node = _object_nodes(manager)[0]
    expected_sum = np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])
    assert object_group(node.attributes, "embeddings")["sum"] == pytest.approx(expected_sum, abs=1e-6)
    assert object_group(node.attributes, "embeddings")["object_embedding"] == pytest.approx(
        expected_sum / np.linalg.norm(expected_sum), abs=1e-6
    )
    assert object_group(node.attributes, "observations")["embedding_observation_count"] == 2

    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=LAMP_EMBEDDING), stamp_sec=102
        ),
    )
    node = _object_nodes(manager)[0]
    expected_sum = expected_sum + np.array([0.0, 0.0, 1.0])
    assert object_group(node.attributes, "embeddings")["sum"] == pytest.approx(expected_sum, abs=1e-6)
    assert object_group(node.attributes, "embeddings")["object_embedding"] == pytest.approx(
        expected_sum / np.linalg.norm(expected_sum), abs=1e-6
    )
    assert object_group(node.attributes, "observations")["embedding_observation_count"] == 3


def test_stored_object_embedding_is_unit_norm():
    manager = _make_manager(semantic_similarity_threshold=-1.0)
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0, embedding=(7.0, 0.0, 0.0))))
    _apply(
        manager,
        _make_proposal_array(_make_proposal(1.05, 1.0, embedding=(0.0, 3.0, 0.0)), stamp_sec=101),
    )

    stored = object_group(_object_nodes(manager)[0].attributes, "embeddings")["object_embedding"]
    assert math.isclose(float(np.linalg.norm(stored)), 1.0, rel_tol=1e-6)


def test_object_semantic_state_retains_only_the_current_class_name():
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    for index in range(4):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.0, 1.0, class_name="chair", confidence=0.9),
                stamp_sec=100 + index,
            ),
        )
    # The current proposal supplies the only persisted semantic field.
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, class_name="sofa", confidence=0.5), stamp_sec=110
        ),
    )

    node = _object_nodes(manager)[0]
    semantic = object_group(node.attributes, "semantic")
    assert semantic == {"class_name": "sofa"}


def test_detection_and_embedding_counters_are_separate():
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    # An opposing observation is folded in but cancels the accumulated direction.
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, embedding=(-1.0, 0.0, 0.0)), stamp_sec=101
        ),
    )

    node = _object_nodes(manager)[0]
    assert object_group(node.attributes, "observations")["detection_observation_count"] == 2
    assert object_group(node.attributes, "observations")["embedding_observation_count"] == 2
    assert object_group(node.attributes, "embeddings")["object_embedding"] is None


def test_position_update_policies():
    averaged = _make_manager(position_update_policy="running_mean")
    _apply(averaged, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(averaged, _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101))
    assert _object_nodes(averaged)[0].pose.position.x == pytest.approx(1.1)

    latest = _make_manager(position_update_policy="latest")
    _apply(latest, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(latest, _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101))
    assert _object_nodes(latest)[0].pose.position.x == pytest.approx(1.2)


def test_malformed_timestamp_falls_back_to_wall_clock():
    manager = _make_manager()
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0), stamp_sec=0))
    node = _object_nodes(manager)[0]
    assert node.created_at > 0.0 and math.isfinite(node.created_at)


# ========== room membership ==========


def test_object_is_attached_to_the_resolved_room():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _make_room(sg)
    manager = _make_manager(
        sg=sg, room_manager=room_manager, room_resolver=lambda x, y: room_id
    )

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))

    object_id = _object_nodes(manager)[0].id
    assert sg.query.has_edge(room_id, object_id, EdgeType.ROOM_CONTAINS)
    assert room_manager.get_room_id_for_direct_member(object_id) == room_id


def test_repeated_observations_do_not_duplicate_containment_edges():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _make_room(sg)
    manager = _make_manager(
        sg=sg, room_manager=room_manager, room_resolver=lambda x, y: room_id
    )

    for index in range(4):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.0 + 0.02 * index, 1.0), stamp_sec=100 + index
            ),
        )

    object_id = _object_nodes(manager)[0].id
    containment = [
        edge
        for edge in sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)
        if int(edge.source_id) == room_id
    ]
    assert len(containment) == 1


def test_previously_unassigned_object_is_attached_once_a_region_covers_it():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _make_room(sg)
    rooms = {"current": None}
    manager = _make_manager(
        sg=sg,
        room_manager=room_manager,
        room_resolver=lambda x, y: rooms["current"],
        semantic_similarity_threshold=-1.0,
    )

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    object_id = _object_nodes(manager)[0].id
    assert list(sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)) == []

    # A later region update now covers the object's position.
    rooms["current"] = room_id
    _apply(manager, _make_proposal_array(_make_proposal(1.05, 1.0), stamp_sec=101))

    assert len(_object_nodes(manager)) == 1
    assert room_manager.get_room_id_for_direct_member(object_id) == room_id


def test_detaching_an_object_removes_every_containment_edge():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _make_room(sg)
    manager = _make_manager(
        sg=sg, room_manager=room_manager, room_resolver=lambda x, y: room_id
    )

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    object_id = _object_nodes(manager)[0].id
    assert room_manager.get_room_id_for_direct_member(object_id) == room_id

    # The same call the manager makes when a position leaves every region.
    room_manager.detach_direct_member_from_rooms(object_id)

    assert list(sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)) == []
    assert room_manager.get_room_id_for_direct_member(object_id) is None


def test_association_does_not_cross_incompatible_rooms():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_a = _make_room(sg, name="room_0")
    room_b = _make_room(sg, x=10.0, name="room_1")
    rooms = {"current": room_a}
    manager = _make_manager(
        sg=sg, room_manager=room_manager, room_resolver=lambda x, y: rooms["current"]
    )

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))

    # Same place, same class, but the viewpoint resolves to a different room.
    rooms["current"] = room_b
    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.02, 1.0), stamp_sec=101)
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


def test_room_scoping_works_without_a_room_manager():
    sg = create_scene_graph_interface()
    room_id = _make_room(sg)
    manager = _make_manager(sg=sg, room_resolver=lambda x, y: room_id)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    object_id = _object_nodes(manager)[0].id
    sg.update.add_edge(Edge(room_id, object_id, type=EdgeType.ROOM_CONTAINS))

    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.02, 1.0), stamp_sec=101)
    )
    assert stats["updated_objects"] == 1
    assert len(_object_nodes(manager)) == 1


# ========== observation edges and statistics ==========


def test_line_of_sight_observation_edges_are_not_duplicated():
    sg = create_scene_graph_interface()
    manager = _make_manager(sg=sg)
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 0.0)))
    object_id = _object_nodes(manager)[0].id

    pose = PoseNode()
    pose.pose.orientation.w = 1.0
    pose_id = int(sg.update.add_node(pose))
    pose_node = sg.query.get_node(pose_id)

    manager.compute_line_of_sight_for_pose(pose_node)
    manager.compute_line_of_sight_for_pose(pose_node)

    edges = [
        edge
        for edge in sg.query.get_outgoing_edges(pose_id, EdgeType.OBSERVATION_ANCHOR)
        if int(edge.target_id) == object_id
    ]
    assert len(edges) <= 1


def test_statistics_cover_every_outcome():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(manager, _make_proposal_array(_make_proposal(1.1, 1.0), stamp_sec=101))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING), stamp_sec=102
        ),
    )
    _apply(
        manager,
        _make_proposal_array(_make_proposal(9.0, 9.0, valid_3d=False), stamp_sec=103),
    )
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(9.0, 9.0, embedding=(0.0, 0.0, 0.0)), stamp_sec=104
        ),
    )
    _apply(
        manager,
        _make_proposal_array(_make_proposal(1.0, 1.0), frame_id="camera", stamp_sec=105),
    )

    stats = manager.get_statistics()
    assert stats["objects_created"] == 1
    assert stats["objects_updated"] == 1
    assert stats["detections_spatially_ambiguous"] == 1
    assert stats["rejected_invalid_geometry"] == 1
    assert stats["rejected_invalid_embedding"] == 1
    assert stats["rejected_missing_tf"] == 1
