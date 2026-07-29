#!/usr/bin/env python3
"""Unit tests for semantic-perception object association and tracking."""

import math

import numpy as np
import pytest
from semantic_perception_msgs.msg import ObjectProposal3D, ObjectProposal3DArray

from scene_graph_core.algorithms.semantic import cosine_similarity, normalize_embedding
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    Edge,
    EdgeType,
    NodeType,
    PoseNode,
    RoomNode,
)
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


# Three mutually orthogonal directions give unambiguous similarity outcomes.
CHAIR_EMBEDDING = (1.0, 0.0, 0.0)
TABLE_EMBEDDING = (0.0, 1.0, 0.0)
LAMP_EMBEDDING = (0.0, 0.0, 1.0)


def _make_manager(
    *,
    sg=None,
    spatial_association_threshold: float = 0.75,
    semantic_similarity_threshold: float = 0.7,
    position_update_policy: str = "latest",
    room_manager=None,
) -> ObjectNodeManager:
    return ObjectNodeManager(
        sg_interface=sg if sg is not None else create_scene_graph_interface(),
        logger=MockLogger(),
        spatial_association_threshold=spatial_association_threshold,
        semantic_similarity_threshold=semantic_similarity_threshold,
        position_update_policy=position_update_policy,
        room_manager=room_manager,
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
) -> ObjectProposal3D:
    proposal = ObjectProposal3D()
    proposal.id = int(proposal_id)
    proposal.class_name = class_name
    proposal.detection_confidence = float(confidence)
    proposal.detector_source = "groundingdino"
    proposal.valid_3d = bool(valid_3d)
    proposal.centroid_3d.x = float(x)
    proposal.centroid_3d.y = float(y)
    proposal.centroid_3d.z = float(z)
    proposal.bbox_3d.size.x = 0.3
    proposal.bbox_3d.size.y = 0.3
    proposal.bbox_3d.size.z = 0.3
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
        msg,
        tf_buffer=None,
        fixed_frame_id="odom",
        room_id=room_id,
    )


def _object_nodes(manager: ObjectNodeManager):
    return manager.sg.query.find_nodes_by_type(NodeType.OBJECT)


def _make_room(sg, x: float = 0.0, y: float = 0.0, name: str = "room_0") -> int:
    room = RoomNode(attributes={"name": name})
    room.pose.position.x = float(x)
    room.pose.position.y = float(y)
    room.pose.orientation.w = 1.0
    return int(sg.update.add_node(room))


# ========== 1. Creating new object nodes ==========


def test_new_object_node_created_when_no_match_exists():
    manager = _make_manager()

    stats = _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))

    assert stats["new_objects"] == 1
    assert stats["updated_objects"] == 0
    assert stats["accepted_detections"] == 1
    assert len(_object_nodes(manager)) == 1

    node = _object_nodes(manager)[0]
    assert node.attributes["class_name"] == "chair"
    assert node.attributes["detector_source"] == "groundingdino"
    assert node.attributes["observation_count"] == 1
    assert node.attributes["embedding_observation_count"] == 1
    assert node.attributes["object_embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert node.attributes["bbox_3d_size"] == pytest.approx([0.3, 0.3, 0.3])
    assert node.attributes["semantic_perception_id"] == 0


def test_empty_and_none_messages_are_handled_without_creating_nodes():
    manager = _make_manager()

    assert _apply(manager, _make_proposal_array())["new_objects"] == 0
    assert _apply(manager, None)["new_objects"] == 0
    assert _object_nodes(manager) == []
    assert manager.stats["total_detection_messages"] == 2


# ========== 2. Updating an existing node ==========


def test_close_and_similar_proposal_updates_existing_node():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101),
    )

    assert stats["new_objects"] == 0
    assert stats["updated_objects"] == 1
    assert len(_object_nodes(manager)) == 1

    node = _object_nodes(manager)[0]
    assert node.attributes["observation_count"] == 2
    assert node.attributes["embedding_observation_count"] == 2
    assert math.isclose(node.attributes["last_semantic_similarity"], 1.0, rel_tol=1e-6)


# ========== 3. Close but semantically different ==========


def test_spatially_close_but_dissimilar_proposal_creates_separate_node():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING, class_name="table"),
            stamp_sec=101,
        ),
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


# ========== 4. Similar but spatially far ==========


def test_semantically_similar_but_distant_proposal_creates_separate_node():
    manager = _make_manager(spatial_association_threshold=0.75)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(5.0, 1.0), stamp_sec=101),
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


def test_association_thresholds_are_configurable():
    # cosine ≈ 0.9 against CHAIR_EMBEDDING
    partial = [0.9, 0.4358898943540674, 0.0]

    strict = _make_manager(semantic_similarity_threshold=0.99)
    _apply(strict, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        strict,
        _make_proposal_array(_make_proposal(1.1, 1.0, embedding=partial), stamp_sec=101),
    )
    assert len(_object_nodes(strict)) == 2

    lenient = _make_manager(semantic_similarity_threshold=0.5)
    _apply(lenient, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        lenient,
        _make_proposal_array(_make_proposal(1.1, 1.0, embedding=partial), stamp_sec=101),
    )
    assert len(_object_nodes(lenient)) == 1


# ========== 5. Best candidate among several ==========


def test_best_candidate_is_selected_among_multiple_nearby_nodes():
    manager = _make_manager(semantic_similarity_threshold=0.5)

    # Two distinct nearby objects, both inside the spatial threshold of the query.
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.3, 1.0, embedding=TABLE_EMBEDDING),
            stamp_sec=101,
        ),
    )
    assert len(_object_nodes(manager)) == 2

    chair_id = min(node.id for node in _object_nodes(manager))
    table_id = max(node.id for node in _object_nodes(manager))

    # A chair-like observation sits closer to the table node but must still bind
    # to the semantically better chair node.
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.25, 1.0), stamp_sec=102),
    )

    assert stats["updated_objects"] == 1
    assert stats["updated_object_ids"] == [chair_id]
    assert len(_object_nodes(manager)) == 2

    table_node = manager.sg.query.get_node(table_id)
    assert table_node.attributes["observation_count"] == 1


def test_equal_similarity_ties_are_broken_by_distance():
    manager = _make_manager(semantic_similarity_threshold=0.5)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, embedding=TABLE_EMBEDDING),
            stamp_sec=101,
        ),
    )
    # Same pose, orthogonal embedding -> two nodes, then move the second away.
    far_node = max(_object_nodes(manager), key=lambda node: node.id)
    far_node.pose.position.x = 1.4
    manager.sg.update.update_node(far_node.id, far_node)
    near_id = min(node.id for node in _object_nodes(manager))

    # An embedding equidistant from both candidates ties on similarity.
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=(1.0, 1.0, 0.0)),
            stamp_sec=102,
        ),
    )

    assert stats["updated_object_ids"] == [near_id]


# ========== 6. Running-mean object_embedding ==========


def test_object_embedding_follows_incremental_running_mean():
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    node = _object_nodes(manager)[0]
    assert node.attributes["object_embedding"] == pytest.approx([1.0, 0.0, 0.0])

    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=TABLE_EMBEDDING),
            stamp_sec=101,
        ),
    )

    node = _object_nodes(manager)[0]
    expected = np.array([1.0, 0.0, 0.0]) + np.array([0.0, 1.0, 0.0])
    expected = expected / np.linalg.norm(expected)
    assert node.attributes["object_embedding"] == pytest.approx(expected, abs=1e-6)
    assert node.attributes["embedding_observation_count"] == 2

    # A third observation must use n=2, not recompute from scratch.
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=LAMP_EMBEDDING),
            stamp_sec=102,
        ),
    )
    node = _object_nodes(manager)[0]
    expected = 2 * expected + np.array([0.0, 0.0, 1.0])
    expected = expected / np.linalg.norm(expected)
    assert node.attributes["object_embedding"] == pytest.approx(expected, abs=1e-6)
    assert node.attributes["embedding_observation_count"] == 3


def test_stored_object_embedding_stays_unit_norm():
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(
        manager,
        _make_proposal_array(_make_proposal(1.0, 1.0, embedding=(7.0, 0.0, 0.0))),
    )
    for index, embedding in enumerate(((0.0, 3.0, 0.0), (0.0, 0.0, 9.0))):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.05, 1.0, embedding=embedding),
                stamp_sec=101 + index,
            ),
        )

    stored = _object_nodes(manager)[0].attributes["object_embedding"]
    assert math.isclose(float(np.linalg.norm(stored)), 1.0, rel_tol=1e-6)


# ========== 7. Invalid / inconsistent embeddings ==========


@pytest.mark.parametrize(
    "embedding",
    [
        None,
        (),
        (0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0),
        (float("inf"), 0.0, 0.0),
    ],
)
def test_invalid_embeddings_create_nodes_without_representation(embedding):
    manager = _make_manager()

    stats = _apply(
        manager, _make_proposal_array(_make_proposal(1.0, 1.0, embedding=embedding))
    )

    assert stats["new_objects"] == 1
    node = _object_nodes(manager)[0]
    assert node.attributes["object_embedding"] is None
    assert node.attributes["embedding_observation_count"] == 0
    assert node.attributes["observation_count"] == 1


def test_invalid_embedding_never_associates_with_an_existing_node():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.02, 1.0, embedding=(0.0, 0.0, 0.0)),
            stamp_sec=101,
        ),
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


def test_dimension_mismatch_does_not_associate_or_corrupt_representation():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.02, 1.0, embedding=(1.0, 0.0)),
            stamp_sec=101,
        ),
    )

    assert stats["new_objects"] == 1
    nodes = sorted(_object_nodes(manager), key=lambda node: node.id)
    assert nodes[0].attributes["object_embedding"] == pytest.approx([1.0, 0.0, 0.0])
    assert nodes[1].attributes["object_embedding"] == pytest.approx([1.0, 0.0])


def test_cancelling_observation_preserves_the_previous_running_mean():
    # An exactly opposing observation makes the incremental mean degenerate; the
    # node must keep its previous representation instead of storing a zero vector.
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.05, 1.0, embedding=(-1.0, 0.0, 0.0)),
            stamp_sec=101,
        ),
    )

    assert stats["updated_objects"] == 1
    node = _object_nodes(manager)[0]
    assert len(_object_nodes(manager)) == 1
    assert node.attributes["object_embedding"] == pytest.approx([1.0, 0.0, 0.0])


# ========== 8. Counts and timestamps ==========


def test_observation_counts_and_timestamps_update_independently():
    manager = _make_manager()

    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0), stamp_sec=100, stamp_nanosec=500_000_000
        ),
    )
    node = _object_nodes(manager)[0]
    assert node.created_at == pytest.approx(100.5)
    assert node.last_seen == pytest.approx(100.5)

    _apply(manager, _make_proposal_array(_make_proposal(1.1, 1.0), stamp_sec=142))

    node = _object_nodes(manager)[0]
    assert node.created_at == pytest.approx(100.5)  # first_seen is preserved
    assert node.last_seen == pytest.approx(142.0)
    assert node.attributes["observation_count"] == 2


def test_malformed_timestamp_falls_back_to_wall_clock():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0), stamp_sec=0))

    node = _object_nodes(manager)[0]
    assert node.created_at > 0.0
    assert math.isfinite(node.created_at)


def test_detection_and_embedding_counters_are_distinct():
    # observation_count counts associated detections; embedding_observation_count
    # counts only the embeddings actually folded into the running mean.
    manager = _make_manager(semantic_similarity_threshold=-1.0)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    for index in range(3):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.05, 1.0, embedding=(-1.0, 0.0, 0.0)),
                stamp_sec=101 + index,
            ),
        )

    node = _object_nodes(manager)[0]
    assert node.attributes["observation_count"] == 4
    assert node.attributes["embedding_observation_count"] == 1


def test_object_without_embedding_never_absorbs_later_proposals():
    manager = _make_manager()

    _apply(
        manager,
        _make_proposal_array(_make_proposal(1.0, 1.0, embedding=(0.0, 0.0, 0.0))),
    )
    _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.02, 1.0, embedding=(0.0, 0.0, 0.0)), stamp_sec=101
        ),
    )

    # Without a usable representation the semantic threshold can never be met.
    assert len(_object_nodes(manager)) == 2
    for node in _object_nodes(manager):
        assert node.attributes["observation_count"] == 1
        assert node.attributes["embedding_observation_count"] == 0


def test_position_update_policies_change_the_spatial_representation():
    latest = _make_manager()
    _apply(latest, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(latest, _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101))
    assert _object_nodes(latest)[0].pose.position.x == pytest.approx(1.2)

    averaged = _make_manager(position_update_policy="running_mean")
    _apply(averaged, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(averaged, _make_proposal_array(_make_proposal(1.2, 1.0), stamp_sec=101))
    assert _object_nodes(averaged)[0].pose.position.x == pytest.approx(1.1)


# ========== Malformed proposals ==========


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"confidence": float("nan")}, "nonfinite_detection_confidence"),
        ({"valid_3d": False}, "invalid_3d_geometry"),
        ({"x": float("inf")}, "nonfinite_centroid"),
    ],
)
def test_malformed_proposals_are_rejected_without_crashing(kwargs, reason):
    manager = _make_manager()
    x = kwargs.pop("x", 1.0)

    stats = _apply(manager, _make_proposal_array(_make_proposal(x, 1.0, **kwargs)))

    assert stats["accepted_detections"] == 0
    assert stats["rejected_detections"] == 1
    assert stats["rejected_by_reason"] == {reason: 1}
    assert _object_nodes(manager) == []


def test_one_bad_proposal_does_not_block_the_rest_of_the_batch():
    manager = _make_manager()

    stats = _apply(
        manager,
        _make_proposal_array(
            _make_proposal(1.0, 1.0, valid_3d=False),
            _make_proposal(3.0, 3.0, embedding=TABLE_EMBEDDING),
        ),
    )

    assert stats["accepted_detections"] == 1
    assert stats["rejected_detections"] == 1
    assert len(_object_nodes(manager)) == 1


# ========== 9/10. Room containment ==========


def test_object_is_attached_to_the_supplied_room():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    manager = _make_manager(sg=sg, room_manager=room_manager)
    room_id = _make_room(sg)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)), room_id=room_id)

    object_id = _object_nodes(manager)[0].id
    assert sg.query.has_edge(room_id, object_id, EdgeType.ROOM_CONTAINS)
    assert room_manager.get_room_id_for_direct_member(object_id) == room_id


def test_repeated_observations_do_not_duplicate_containment_edges():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    manager = _make_manager(sg=sg, room_manager=room_manager)
    room_id = _make_room(sg)

    for index in range(4):
        _apply(
            manager,
            _make_proposal_array(
                _make_proposal(1.0 + 0.02 * index, 1.0), stamp_sec=100 + index
            ),
            room_id=room_id,
        )

    assert len(_object_nodes(manager)) == 1
    object_id = _object_nodes(manager)[0].id
    containment = [
        edge
        for edge in sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)
        if int(edge.source_id) == room_id
    ]
    assert len(containment) == 1
    assert _object_nodes(manager)[0].attributes["observation_count"] == 4


def test_reassignment_removes_the_stale_containment_edge():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    manager = _make_manager(sg=sg, room_manager=room_manager)
    room_a = _make_room(sg, name="room_0")
    room_b = _make_room(sg, x=10.0, name="room_1")

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)), room_id=room_a)
    object_id = _object_nodes(manager)[0].id

    # Region-driven membership sync is the path that moves an object between rooms.
    room_manager.attach_direct_member_to_room(
        room_b, object_id, allow_reassignment=True, reason="test_reassignment"
    )

    owning_rooms = {
        int(edge.source_id)
        for edge in sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)
    }
    assert owning_rooms == {room_b}
    assert room_manager.get_room_id_for_direct_member(object_id) == room_b

    # Later observations from the new room keep updating the same node.
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.05, 1.0), stamp_sec=101),
        room_id=room_b,
    )
    assert stats["updated_object_ids"] == [object_id]
    assert len(_object_nodes(manager)) == 1


def test_objects_in_different_rooms_are_never_merged():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    manager = _make_manager(sg=sg, room_manager=room_manager)
    room_a = _make_room(sg, name="room_0")
    room_b = _make_room(sg, x=10.0, name="room_1")

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)), room_id=room_a)
    # Spatially close and semantically identical, but observed from another room.
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.02, 1.0), stamp_sec=101),
        room_id=room_b,
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


def test_unroomed_observation_does_not_bind_to_a_roomed_object():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    manager = _make_manager(sg=sg, room_manager=room_manager)
    room_id = _make_room(sg)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)), room_id=room_id)
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.02, 1.0), stamp_sec=101),
        room_id=None,
    )

    assert stats["new_objects"] == 1
    assert len(_object_nodes(manager)) == 2


def test_room_scoping_works_without_a_room_manager():
    sg = create_scene_graph_interface()
    manager = _make_manager(sg=sg)
    room_id = _make_room(sg)

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    object_id = _object_nodes(manager)[0].id
    sg.update.add_edge(Edge(room_id, object_id, type=EdgeType.ROOM_CONTAINS))

    # Without a room_manager the graph edges still scope association.
    stats = _apply(
        manager,
        _make_proposal_array(_make_proposal(1.02, 1.0), stamp_sec=101),
        room_id=room_id,
    )
    assert stats["updated_objects"] == 1
    assert len(_object_nodes(manager)) == 1


# ========== Observation edges / statistics ==========


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


def test_statistics_track_creation_association_and_rejection():
    manager = _make_manager()

    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))
    _apply(manager, _make_proposal_array(_make_proposal(1.1, 1.0), stamp_sec=101))
    _apply(
        manager,
        _make_proposal_array(_make_proposal(1.0, 1.0, valid_3d=False), stamp_sec=102),
    )

    stats = manager.get_statistics()
    assert stats["total_objects_created"] == 1
    assert stats["total_objects_updated"] == 1
    assert stats["total_objects_associated"] == 1
    assert stats["total_detections_accepted"] == 2
    assert stats["total_detections_rejected"] == 1
    assert stats["rejected_by_reason"] == {"invalid_3d_geometry": 1}
    assert stats["last_detection_stamp_sec"] == pytest.approx(102.0)


def test_similarity_helpers_agree_with_stored_representation():
    manager = _make_manager()
    _apply(manager, _make_proposal_array(_make_proposal(1.0, 1.0)))

    stored = _object_nodes(manager)[0].attributes["object_embedding"]
    assert cosine_similarity(stored, CHAIR_EMBEDDING) == pytest.approx(1.0, abs=1e-6)
    assert normalize_embedding(stored) is not None
