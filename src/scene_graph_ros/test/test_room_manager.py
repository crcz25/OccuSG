#!/usr/bin/env python3
"""Unit tests for room anchoring and direct room membership."""

from geometry_msgs.msg import Point32
from incremental_dude_msgs.msg import Region2D, Region2DArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import Edge, EdgeType, NavNode, NodeType, ObjectNode, PoseNode
from scene_graph_ros.managers.region_manager import RegionManager
from scene_graph_ros.managers.room_manager import RoomManager


class MockLogger:
    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass


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
    adjacent_ids: tuple[int, ...] = (),
) -> Region2D:
    region = Region2D()
    region.id = int(tracker_region_id)
    region.centroid.x = float(centroid_xy[0])
    region.centroid.y = float(centroid_xy[1])
    region.area = 1.0
    region.adjacent_ids = [int(region_id) for region_id in adjacent_ids]
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


def _make_managers():
    sg = create_scene_graph_interface()
    logger = MockLogger()
    return (
        sg,
        RegionManager(sg_interface=sg, logger=logger, z_offset=8.0),
        RoomManager(sg_interface=sg, logger=logger, z_offset=12.0),
    )


def _room_adjacency_pairs(sg) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(edge.source_id), int(edge.target_id))))
        for edge in sg.query.get_all_edges(EdgeType.ROOM_ADJACENCY)
    }


def test_room_association_mirrors_stable_region_identity_and_geometry():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(0.0, 0.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )

    assert room_id is not None
    room_manager.associate_room_with_tracker_region(
        room_id,
        11,
        prepared_regions[11],
        is_bootstrap_region=True,
    )
    room_node = sg.query.get_node(room_id)

    assert room_manager.get_room_id_for_tracker_region(11) == room_id
    assert room_node.attributes["stable_region_id"] == 11
    assert room_node.attributes["tracker_region_id"] == 11
    assert room_node.attributes["is_bootstrap_region"] is True
    assert room_node.attributes["polygon"][0] == {"x": 0.0, "y": 0.0}
    assert room_node.attributes["bounds"]["max_x"] == 2.0
    assert room_node.pose.position.x == 1.0
    assert sg.query.find_nodes_by_type(NodeType.REGION) == []


def test_sync_room_membership_from_region_attaches_only_geometry_filtered_members():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    inside_object_id = sg.update.add_node(_make_object_node("chair", 1.5, 1.5))
    outside_object_id = sg.update.add_node(_make_object_node("lamp", 5.0, 5.0))
    nav_id = sg.update.add_node(_make_nav_node(1.0, 0.5))
    room_id = room_manager.create_initial_room(pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(
        room_id,
        11,
        prepared_regions[11],
    )
    member_ids = region_manager.gather_region_member_ids(prepared_regions[11])

    stats = room_manager.sync_room_membership_from_region(room_id, member_ids)

    direct_members = room_manager.get_attached_direct_member_ids(room_id)
    assert stats["assigned"] == 3
    assert direct_members[NodeType.AGENT] == {pose_id}
    assert direct_members[NodeType.OBJECT] == {inside_object_id}
    assert direct_members[NodeType.NAVIGATION] == {nav_id}
    assert room_manager.get_room_id_for_direct_member(outside_object_id) is None
    assert sg.query.get_all_edges(EdgeType.REGION_CONTAINS) == []


def test_region_aware_room_adjacency_requires_dude_adjacency_and_nav_bridge():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_b_id = sg.update.add_node(_make_pose_node(5.0, 1.0))
    nav_a_id = sg.update.add_node(_make_nav_node(1.0, 1.0))
    nav_b_id = sg.update.add_node(_make_nav_node(5.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    room_b_id = room_manager.create_room_from_pose(pose_b_id, name="room_b")
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                adjacent_ids=(22,),
            ),
            _make_region(
                22,
                centroid_xy=(5.0, 1.0),
                polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
                adjacent_ids=(11,),
            ),
        )
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, prepared_regions[11])
    room_manager.associate_room_with_tracker_region(room_b_id, 22, prepared_regions[22])
    sg.update.add_edge(
        Edge(source_id=nav_a_id, target_id=nav_b_id, type=EdgeType.NAVIGABLE_PATH)
    )

    stats = room_manager.rebuild_room_adjacency(
        prepared_regions=prepared_regions,
        region_nav_ids_by_tracker_region={
            11: {nav_a_id},
            22: {nav_b_id},
        },
    )

    assert stats["dude_region_pairs"] == [(11, 22)]
    assert stats["candidate_cross_region_nav_bridges"] == [
        (nav_a_id, nav_b_id, 11, 22)
    ]
    assert stats["adjacent_room_pairs"] == [tuple(sorted((room_a_id, room_b_id)))]
    assert _room_adjacency_pairs(sg) == {tuple(sorted((room_a_id, room_b_id)))}


def test_region_aware_room_adjacency_rejects_adjacent_regions_without_nav_bridge():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_b_id = sg.update.add_node(_make_pose_node(5.0, 1.0))
    nav_a_id = sg.update.add_node(_make_nav_node(1.0, 1.0))
    nav_b_id = sg.update.add_node(_make_nav_node(5.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    room_b_id = room_manager.create_room_from_pose(pose_b_id, name="room_b")
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                adjacent_ids=(22,),
            ),
            _make_region(
                22,
                centroid_xy=(5.0, 1.0),
                polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
            ),
        )
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, prepared_regions[11])
    room_manager.associate_room_with_tracker_region(room_b_id, 22, prepared_regions[22])

    stats = room_manager.rebuild_room_adjacency(
        prepared_regions=prepared_regions,
        region_nav_ids_by_tracker_region={
            11: {nav_a_id},
            22: {nav_b_id},
        },
    )

    assert stats["adjacent_room_pairs"] == []
    assert stats["rejected_missing_nav_bridge_pairs"] == [(11, 22)]
    assert _room_adjacency_pairs(sg) == set()


def test_region_aware_room_adjacency_rejects_nav_bridge_without_dude_adjacency():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_b_id = sg.update.add_node(_make_pose_node(5.0, 1.0))
    nav_a_id = sg.update.add_node(_make_nav_node(1.0, 1.0))
    nav_b_id = sg.update.add_node(_make_nav_node(5.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    room_b_id = room_manager.create_room_from_pose(pose_b_id, name="room_b")
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
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
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, prepared_regions[11])
    room_manager.associate_room_with_tracker_region(room_b_id, 22, prepared_regions[22])
    sg.update.add_edge(
        Edge(source_id=nav_a_id, target_id=nav_b_id, type=EdgeType.NAVIGABLE_PATH)
    )

    stats = room_manager.rebuild_room_adjacency(
        prepared_regions=prepared_regions,
        region_nav_ids_by_tracker_region={
            11: {nav_a_id},
            22: {nav_b_id},
        },
    )

    assert stats["adjacent_room_pairs"] == []
    assert stats["rejected_missing_dude_adjacency_pairs"] == [(11, 22)]
    assert _room_adjacency_pairs(sg) == set()


def test_region_aware_room_adjacency_ignores_missing_unowned_and_ambiguous_nav():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_b_id = sg.update.add_node(_make_pose_node(5.0, 1.0))
    shared_nav_id = sg.update.add_node(_make_nav_node(2.0, 1.0))
    nav_b_id = sg.update.add_node(_make_nav_node(5.0, 1.0))
    unowned_nav_id = sg.update.add_node(_make_nav_node(3.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    room_b_id = room_manager.create_room_from_pose(pose_b_id, name="room_b")
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                adjacent_ids=(22,),
            ),
            _make_region(
                22,
                centroid_xy=(5.0, 1.0),
                polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
                adjacent_ids=(11,),
            ),
        )
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, prepared_regions[11])
    room_manager.associate_room_with_tracker_region(room_b_id, 22, prepared_regions[22])
    sg.update.add_edge(
        Edge(source_id=shared_nav_id, target_id=nav_b_id, type=EdgeType.NAVIGABLE_PATH)
    )
    sg.update.add_edge(
        Edge(source_id=unowned_nav_id, target_id=nav_b_id, type=EdgeType.NAVIGABLE_PATH)
    )

    stats = room_manager.rebuild_room_adjacency(
        prepared_regions=prepared_regions,
        region_nav_ids_by_tracker_region={
            11: {shared_nav_id, 999999},
            22: {shared_nav_id, nav_b_id},
        },
    )

    assert stats["adjacent_room_pairs"] == []
    assert stats["ambiguous_nav_node_ids"] == [shared_nav_id]
    assert stats["missing_nav_node_ids"] == [999999]
    assert stats["skipped_nav_edges"] == [
        (shared_nav_id, nav_b_id, None, 22),
        (unowned_nav_id, nav_b_id, None, 22),
    ]
    assert _room_adjacency_pairs(sg) == set()


def test_region_aware_room_adjacency_rejects_stale_room_anchor():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_stale_id = sg.update.add_node(_make_pose_node(9.0, 1.0))
    nav_a_id = sg.update.add_node(_make_nav_node(1.0, 1.0))
    nav_stale_id = sg.update.add_node(_make_nav_node(9.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    stale_room_id = room_manager.create_room_from_pose(pose_stale_id, name="room_stale")
    _, all_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
                adjacent_ids=(33,),
            ),
            _make_region(
                33,
                centroid_xy=(9.0, 1.0),
                polygon_xy=[(8.0, 0.0), (10.0, 0.0), (10.0, 2.0), (8.0, 2.0)],
                adjacent_ids=(11,),
            ),
        )
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, all_regions[11])
    room_manager.associate_room_with_tracker_region(stale_room_id, 33, all_regions[33])
    sg.update.add_edge(
        Edge(source_id=nav_a_id, target_id=nav_stale_id, type=EdgeType.NAVIGABLE_PATH)
    )

    stats = room_manager.rebuild_room_adjacency(
        prepared_regions={11: all_regions[11]},
        region_nav_ids_by_tracker_region={
            11: {nav_a_id},
            33: {nav_stale_id},
        },
    )

    assert stats["dude_region_pairs"] == []
    assert stats["region_to_room"] == {11: room_a_id}
    assert stats["adjacent_room_pairs"] == []
    assert _room_adjacency_pairs(sg) == set()


def test_room_signature_uses_direct_geometry_gated_objects():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    sg.update.add_node(_make_object_node("chair", 1.5, 1.5))
    sg.update.add_node(_make_object_node("lamp", 5.0, 5.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, prepared_regions[11])
    room_manager.sync_room_membership_from_region(
        room_id,
        region_manager.gather_region_member_ids(prepared_regions[11]),
    )

    signature_set = room_manager.build_room_region_signature_set(room_id, persist=True)
    room_node = sg.query.get_node(room_id)

    assert {item[0] for item in signature_set} == {"chair"}
    assert room_node.attributes["signature_set"][0]["class_name"] == "chair"


def test_room_persists_with_valid_anchor_even_when_direct_members_are_empty():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, prepared_regions[11])
    room_manager.sync_room_membership_from_region(room_id, {})

    pruned = room_manager.prune_rooms_without_valid_anchors(prepared_regions.keys())

    assert pruned == []
    assert sg.query.get_node(room_id) is not None
    assert room_manager.get_attached_direct_member_ids(room_id)[NodeType.OBJECT] == set()


def test_room_with_invalid_anchor_is_pruned_and_adjacency_is_removed():
    sg, region_manager, room_manager = _make_managers()
    pose_a_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_b_id = sg.update.add_node(_make_pose_node(5.0, 1.0))
    room_a_id = room_manager.create_room_from_pose(pose_a_id, name="room_a")
    room_b_id = room_manager.create_room_from_pose(pose_b_id, name="room_b")
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
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
    )
    room_manager.associate_room_with_tracker_region(room_a_id, 11, prepared_regions[11])
    room_manager.associate_room_with_tracker_region(room_b_id, 22, prepared_regions[22])
    sg.update.add_edge(
        Edge(source_id=room_a_id, target_id=room_b_id, type=EdgeType.ROOM_ADJACENCY),
        is_structural=False,
    )
    room_manager.room_adjacency_pairs.add(tuple(sorted((room_a_id, room_b_id))))

    pruned = room_manager.prune_rooms_without_valid_anchors({11})

    assert [item["room_node_id"] for item in pruned] == [room_b_id]
    assert sg.query.get_node(room_a_id) is not None
    assert sg.query.get_node(room_b_id) is None
    assert sg.query.get_all_edges(EdgeType.ROOM_ADJACENCY) == []


def _assert_invalid_anchor_room_is_pruned_despite_direct_member(member_node):
    sg, region_manager, room_manager = _make_managers()
    bootstrap_pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    member_id = sg.update.add_node(member_node)
    room_id = room_manager.create_initial_room(bootstrap_pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, prepared_regions[11])
    assert room_manager.attach_direct_member_to_room(room_id, member_id)
    room_manager.room_object_ids[room_id] = {member_id}
    room_manager.room_region_centroids[room_id] = [(1.0, 1.0)]
    room_manager.room_signature_sets[room_id] = set()
    room_manager.room_direct_signature_sets[room_id] = set()
    room_manager.dirty_room_ids.add(room_id)

    pruned = room_manager.prune_rooms_without_valid_anchors(set())

    assert pruned == [
        {
            "room_node_id": room_id,
            "name": "room_0",
            "stable_region_id": "11",
        }
    ]
    assert sg.query.get_node(room_id) is None
    assert room_manager.get_room_id_for_tracker_region(11) is None
    assert room_manager.get_room_id_for_direct_member(member_id) is None
    assert sg.query.get_incoming_edges(member_id, EdgeType.ROOM_CONTAINS) == []
    assert room_id not in room_manager.stable_region_to_room.values()
    assert room_id not in room_manager.room_to_stable_region_id
    assert room_id not in room_manager.room_to_tracker_region_id
    assert room_id not in room_manager.room_object_ids
    assert room_id not in room_manager.room_region_centroids
    assert room_id not in room_manager.room_signature_sets
    assert room_id not in room_manager.room_to_direct_members
    assert room_id not in room_manager.room_direct_signature_sets
    assert room_id not in room_manager.dirty_room_ids
    assert room_id not in room_manager.direct_member_to_room.values()


def test_room_with_invalid_anchor_and_object_member_is_pruned():
    _assert_invalid_anchor_room_is_pruned_despite_direct_member(
        _make_object_node("chair", 1.5, 1.5)
    )


def test_room_with_invalid_anchor_and_agent_member_is_pruned():
    _assert_invalid_anchor_room_is_pruned_despite_direct_member(
        _make_pose_node(1.5, 1.5)
    )


def test_room_with_invalid_anchor_and_navigation_member_is_pruned():
    _assert_invalid_anchor_room_is_pruned_despite_direct_member(
        _make_nav_node(1.5, 1.5)
    )


def test_bootstrap_room_with_invalid_anchor_is_pruned():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, prepared_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(
        room_id,
        11,
        prepared_regions[11],
        is_bootstrap_region=True,
    )

    pruned = room_manager.prune_rooms_without_valid_anchors(set())

    assert [item["room_node_id"] for item in pruned] == [room_id]
    assert sg.query.get_node(room_id) is None


def test_live_room_with_stale_anchor_relinks_to_overlapping_replacement_region():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    object_id = sg.update.add_node(_make_object_node("chair", 1.5, 1.5))
    room_id = room_manager.create_initial_room(pose_id)
    _, old_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, old_regions[11])
    room_manager.sync_room_membership_from_region(
        room_id,
        region_manager.gather_region_member_ids(old_regions[11]),
    )
    _, replacement_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                22,
                centroid_xy=(1.05, 1.0),
                polygon_xy=[(-0.1, 0.0), (2.1, 0.0), (2.1, 2.0), (-0.1, 2.0)],
            )
        )
    )

    relinked = room_manager.relink_rooms_to_replacement_regions(replacement_regions)

    room = sg.query.get_node(room_id)
    assert [item["room_node_id"] for item in relinked] == [room_id]
    assert room.attributes["stable_region_id"] == 22
    assert room.attributes["tracker_region_id"] == 22
    assert room_manager.get_room_id_for_tracker_region(11) is None
    assert room_manager.get_room_id_for_tracker_region(22) == room_id
    assert room_manager.get_room_id_for_direct_member(object_id) == room_id


def test_stale_room_anchor_does_not_relink_to_unrelated_region():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, old_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, old_regions[11])
    room_manager.sync_room_membership_from_region(
        room_id,
        region_manager.gather_region_member_ids(old_regions[11]),
    )
    _, replacement_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                22,
                centroid_xy=(5.0, 1.0),
                polygon_xy=[(4.0, 0.0), (6.0, 0.0), (6.0, 2.0), (4.0, 2.0)],
            )
        )
    )

    relinked = room_manager.relink_rooms_to_replacement_regions(replacement_regions)

    room = sg.query.get_node(room_id)
    assert relinked == []
    assert room.attributes["stable_region_id"] == 11
    assert room_manager.get_room_id_for_tracker_region(22) is None


def test_stale_room_anchor_refuses_ambiguous_replacement_regions():
    sg, region_manager, room_manager = _make_managers()
    pose_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    room_id = room_manager.create_initial_room(pose_id)
    _, old_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )
    room_manager.associate_room_with_tracker_region(room_id, 11, old_regions[11])
    room_manager.sync_room_membership_from_region(
        room_id,
        region_manager.gather_region_member_ids(old_regions[11]),
    )
    _, replacement_regions = region_manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                22,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            ),
            _make_region(
                33,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            ),
        )
    )

    relinked = room_manager.relink_rooms_to_replacement_regions(replacement_regions)

    room = sg.query.get_node(room_id)
    assert relinked == []
    assert room.attributes["stable_region_id"] == 11
