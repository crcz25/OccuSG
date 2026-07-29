#!/usr/bin/env python3
"""Object-to-room assignment from Incremental DUDE region polygons."""

import pytest
from geometry_msgs.msg import Point32
from incremental_dude_msgs.msg import Region2D, Region2DArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import EdgeType, ObjectNode, RoomNode
from scene_graph_ros.managers.region_manager import RegionManager
from scene_graph_ros.managers.room_manager import RoomManager


class MockLogger:
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


def _square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _region(region_id, polygon):
    message = Region2D()
    message.id = int(region_id)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    message.centroid.x = sum(xs) / len(xs)
    message.centroid.y = sum(ys) / len(ys)
    for x, y in polygon:
        point = Point32()
        point.x, point.y = float(x), float(y)
        message.polygon.points.append(point)
        hull_point = Point32()
        hull_point.x, hull_point.y = float(x), float(y)
        message.convex_hull.points.append(hull_point)
    message.area = float(abs(max(xs) - min(xs)) * abs(max(ys) - min(ys)))
    return message


def _snapshot(manager, *regions):
    message = Region2DArray()
    for region in regions:
        message.regions.append(region)
    valid, prepared = manager.prepare_region_snapshot(message)
    assert valid
    return prepared


def _make_region_manager(tolerance=0.10):
    return RegionManager(
        sg_interface=create_scene_graph_interface(),
        logger=MockLogger(),
        object_room_boundary_tolerance=tolerance,
    )


def _add_object(sg, x, y):
    node = ObjectNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    node.attributes = {"class_name": "chair"}
    node.id = sg.update.add_node(node)
    return int(node.id)


def _add_room(sg, name="room_0"):
    room = RoomNode(attributes={"name": name})
    room.pose.orientation.w = 1.0
    return int(sg.update.add_node(room))


# ========== strict containment ==========


def test_strictly_contained_object_is_assigned():
    manager = _make_region_manager()
    prepared = _snapshot(manager, _region(3, _square(0, 0, 4, 4)))

    assignment = manager.assign_object_region(2.0, 2.0, prepared)

    assert assignment.region_id == 3
    assert assignment.reason == "contained"


def test_object_outside_all_regions_is_unassigned():
    manager = _make_region_manager()
    prepared = _snapshot(manager, _region(1, _square(0, 0, 4, 4)))

    assignment = manager.assign_object_region(20.0, 20.0, prepared)

    assert assignment.region_id is None
    assert assignment.reason == "outside_all_regions"


def test_containment_beats_a_closer_neighbouring_boundary():
    manager = _make_region_manager()
    prepared = _snapshot(
        manager, _region(1, _square(0, 0, 4, 4)), _region(2, _square(4.02, 0, 8, 4))
    )

    # 0.02 m from region 2's edge, but strictly inside region 1.
    assignment = manager.assign_object_region(4.0 - 1e-6, 2.0, prepared)

    assert assignment.region_id == 1
    assert assignment.reason == "contained"


# ========== boundary tolerance ==========


def test_object_within_tolerance_attaches_to_the_nearest_boundary():
    manager = _make_region_manager(tolerance=0.10)
    prepared = _snapshot(manager, _region(5, _square(0, 0, 4, 4)))

    assignment = manager.assign_object_region(4.05, 2.0, prepared)

    assert assignment.region_id == 5
    assert assignment.reason == "boundary_tolerance"
    assert assignment.boundary_distance == pytest.approx(0.05, abs=1e-6)


def test_object_beyond_tolerance_is_never_forced_into_the_nearest_room():
    manager = _make_region_manager(tolerance=0.10)
    prepared = _snapshot(manager, _region(5, _square(0, 0, 4, 4)))

    assignment = manager.assign_object_region(4.5, 2.0, prepared)

    assert assignment.region_id is None
    assert assignment.boundary_distance == pytest.approx(0.5, abs=1e-6)


def test_tolerance_is_configurable():
    prepared_tight = _snapshot(
        _make_region_manager(0.01), _region(1, _square(0, 0, 4, 4))
    )
    tight = _make_region_manager(0.01)
    loose = _make_region_manager(0.50)

    assert tight.assign_object_region(4.2, 2.0, prepared_tight).region_id is None
    assert loose.assign_object_region(4.2, 2.0, prepared_tight).region_id == 1


def test_boundary_tolerance_picks_the_nearest_of_several_regions():
    manager = _make_region_manager(tolerance=0.50)
    prepared = _snapshot(
        manager, _region(1, _square(0, 0, 4, 4)), _region(2, _square(4.4, 0, 8, 4))
    )

    # 0.1 m from region 1's right edge, 0.3 m from region 2's left edge.
    assignment = manager.assign_object_region(4.1, 2.0, prepared)

    assert assignment.region_id == 1


# ========== overlap tie-breaking ==========


def test_overlapping_regions_resolve_deterministically():
    manager = _make_region_manager()
    prepared = _snapshot(
        manager, _region(9, _square(0, 0, 4, 4)), _region(2, _square(3, 0, 7, 4))
    )

    first = manager.assign_object_region(3.5, 2.0, prepared)
    second = manager.assign_object_region(3.5, 2.0, prepared)

    assert first.region_id == second.region_id
    assert first.region_id in (2, 9)
    assert first.candidate_ids == (2, 9)


def test_assignment_is_independent_of_region_iteration_order():
    manager = _make_region_manager()
    forward = _snapshot(
        manager, _region(1, _square(0, 0, 4, 4)), _region(2, _square(3, 0, 7, 4))
    )
    reverse_manager = _make_region_manager()
    reverse = _snapshot(
        reverse_manager, _region(2, _square(3, 0, 7, 4)), _region(1, _square(0, 0, 4, 4))
    )

    assert (
        manager.assign_object_region(3.5, 2.0, forward).region_id
        == reverse_manager.assign_object_region(3.5, 2.0, reverse).region_id
    )


def test_empty_region_snapshot_leaves_objects_unassigned():
    manager = _make_region_manager()
    assert manager.assign_object_region(1.0, 1.0, {}).region_id is None


# ========== containment edges ==========


def test_containment_edge_is_created_once_and_not_duplicated():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _add_room(sg)
    object_id = _add_object(sg, 1.0, 1.0)

    assert room_manager.attach_direct_member_to_room(room_id, object_id)
    # A second attach to the same room is a no-op.
    assert not room_manager.attach_direct_member_to_room(room_id, object_id)

    edges = list(sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS))
    assert len(edges) == 1


def test_reassignment_removes_the_stale_containment_edge():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_a = _add_room(sg, "room_0")
    room_b = _add_room(sg, "room_1")
    object_id = _add_object(sg, 1.0, 1.0)

    room_manager.attach_direct_member_to_room(room_a, object_id)
    room_manager.attach_direct_member_to_room(
        room_b, object_id, allow_reassignment=True, reason="region_geometry_update"
    )

    owning = {
        int(edge.source_id)
        for edge in sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)
    }
    assert owning == {room_b}
    assert room_manager.get_room_id_for_direct_member(object_id) == room_b


def test_object_never_ends_up_in_two_rooms():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_a = _add_room(sg, "room_0")
    room_b = _add_room(sg, "room_1")
    object_id = _add_object(sg, 1.0, 1.0)

    # Force a duplicate ownership edge, as a stale writer might.
    room_manager.attach_direct_member_to_room(room_a, object_id)
    from scene_graph_core.representation import Edge

    sg.update.add_edge(Edge(room_b, object_id, type=EdgeType.ROOM_CONTAINS))

    resolved = room_manager.get_room_id_for_direct_member(object_id)

    owning = {
        int(edge.source_id)
        for edge in sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)
    }
    assert owning == {resolved}
    assert len(owning) == 1


def test_detach_leaves_the_object_unassigned():
    sg = create_scene_graph_interface()
    room_manager = RoomManager(sg_interface=sg, logger=MockLogger())
    room_id = _add_room(sg)
    object_id = _add_object(sg, 1.0, 1.0)
    room_manager.attach_direct_member_to_room(room_id, object_id)

    room_manager.detach_direct_member_from_rooms(object_id)

    assert list(sg.query.get_incoming_edges(object_id, EdgeType.ROOM_CONTAINS)) == []
    assert room_manager.get_room_id_for_direct_member(object_id) is None


# ========== re-evaluation after change ==========


def test_reassignment_follows_the_object_position():
    manager = _make_region_manager()
    prepared = _snapshot(
        manager, _region(1, _square(0, 0, 4, 4)), _region(2, _square(6, 0, 10, 4))
    )

    assert manager.assign_object_region(2.0, 2.0, prepared).region_id == 1
    # The object moved into the second region.
    assert manager.assign_object_region(8.0, 2.0, prepared).region_id == 2
    # ...and then out of every region.
    assert manager.assign_object_region(5.0, 2.0, prepared).region_id is None


def test_reassignment_follows_updated_region_geometry():
    manager = _make_region_manager()
    before = _snapshot(manager, _region(1, _square(0, 0, 4, 4)))
    assert manager.assign_object_region(5.0, 2.0, before).region_id is None

    # The region grew to cover the object.
    after = _snapshot(manager, _region(1, _square(0, 0, 8, 4)))
    assert manager.assign_object_region(5.0, 2.0, after).region_id == 1


def test_object_membership_uses_position_only_not_footprint():
    manager = _make_region_manager(tolerance=0.0)
    prepared = _snapshot(manager, _region(1, _square(0, 0, 4, 4)))

    # A large object whose centre is outside the region is still unassigned; no
    # footprint-overlap rule can pull it in.
    assert manager.assign_object_region(4.5, 2.0, prepared).region_id is None
