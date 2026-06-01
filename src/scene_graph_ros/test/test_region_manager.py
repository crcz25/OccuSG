#!/usr/bin/env python3
"""Unit tests for runtime-only RegionManager geometry helpers."""

from geometry_msgs.msg import Point32
from incremental_dude_msgs.msg import Region2D, Region2DArray

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import Edge, EdgeType, NavNode, NodeType, ObjectNode, PoseNode
from scene_graph_ros.managers.region_manager import RegionManager


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


def _make_object_node(x: float, y: float) -> ObjectNode:
    node = ObjectNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    return node


def _make_nav_node(x: float, y: float, *, bounds: dict[str, float] | None = None) -> NavNode:
    node = NavNode()
    node.pose.position.x = float(x)
    node.pose.position.y = float(y)
    node.pose.orientation.w = 1.0
    node.attributes = {}
    if bounds is not None:
        node.attributes["bounds"] = {key: float(value) for key, value in bounds.items()}
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


def test_prepare_region_snapshot_returns_runtime_geometry_without_graph_nodes():
    sg = create_scene_graph_interface()
    manager = RegionManager(sg_interface=sg, logger=MockLogger(), z_offset=8.0)

    snapshot_valid, prepared_regions = manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )

    assert snapshot_valid is True
    assert sorted(prepared_regions) == [11]
    assert prepared_regions[11].bounds == (0.0, 0.0, 2.0, 2.0)
    assert sg.query.find_nodes_by_type(NodeType.REGION) == []


def test_gather_region_member_ids_filters_entities_by_runtime_region_geometry():
    sg = create_scene_graph_interface()
    manager = RegionManager(sg_interface=sg, logger=MockLogger(), z_offset=8.0)
    pose_inside_id = sg.update.add_node(_make_pose_node(1.0, 1.0))
    pose_outside_id = sg.update.add_node(_make_pose_node(5.0, 5.0))
    object_inside_id = sg.update.add_node(_make_object_node(1.5, 1.5))
    object_outside_id = sg.update.add_node(_make_object_node(6.0, 6.0))

    _, prepared_regions = manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )

    member_ids = manager.gather_region_member_ids(prepared_regions[11])

    assert member_ids[NodeType.AGENT] == {pose_inside_id}
    assert pose_outside_id not in member_ids[NodeType.AGENT]
    assert member_ids[NodeType.OBJECT] == {object_inside_id}
    assert object_outside_id not in member_ids[NodeType.OBJECT]
    assert sg.query.get_all_edges(EdgeType.REGION_CONTAINS) == []


def test_find_tracker_region_for_pose_and_boundary_fallback_do_not_create_region_nodes():
    sg = create_scene_graph_interface()
    manager = RegionManager(
        sg_interface=sg,
        logger=MockLogger(),
        z_offset=8.0,
        nav_region_boundary_epsilon_m=0.05,
    )
    pose_id = sg.update.add_node(_make_pose_node(2.02, 1.0))

    _, prepared_regions = manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(1.0, 1.0),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)],
            )
        )
    )

    assert manager.find_tracker_region_for_pose(pose_id, prepared_regions) == 11
    assert manager.current_tracker_region_id is None
    assert sg.query.find_nodes_by_type(NodeType.REGION) == []


def test_navigation_resolution_keeps_footprint_overlap_and_neighbor_tiebreak_logic():
    sg = create_scene_graph_interface()
    manager = RegionManager(
        sg_interface=sg,
        logger=MockLogger(),
        z_offset=8.0,
    )
    nav_a_id = sg.update.add_node(_make_nav_node(0.5, 0.5))
    nav_b_id = sg.update.add_node(
        _make_nav_node(
            2.02,
            0.5,
            bounds={"min_x": 1.9, "max_x": 2.1, "min_y": 0.4, "max_y": 0.6},
        )
    )
    sg.update.add_edge(
        Edge(source_id=nav_a_id, target_id=nav_b_id, type=EdgeType.NAVIGABLE_PATH),
        is_structural=False,
    )
    _, prepared_regions = manager.prepare_region_snapshot(
        _make_region_array(
            _make_region(
                11,
                centroid_xy=(0.5, 0.5),
                polygon_xy=[(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)],
            ),
            _make_region(
                22,
                centroid_xy=(2.5, 0.5),
                polygon_xy=[(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)],
            ),
        )
    )

    members_a = manager.gather_region_member_ids(
        prepared_regions[11],
        prepared_regions=prepared_regions,
        node_types=(NodeType.NAVIGATION,),
    )
    members_b = manager.gather_region_member_ids(
        prepared_regions[22],
        prepared_regions=prepared_regions,
        node_types=(NodeType.NAVIGATION,),
    )

    assert nav_a_id in members_a[NodeType.NAVIGATION]
    assert nav_b_id in members_b[NodeType.NAVIGATION]
    assert sg.query.get_all_edges(EdgeType.REGION_CONTAINS) == []
