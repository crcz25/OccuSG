#!/usr/bin/env python3
"""Unit tests for free-space navigation node maintenance."""

from nav_msgs.msg import OccupancyGrid
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import EdgeType, NodeType, ObjectNode
from scene_graph_ros.managers.free_space_manager import FreeSpaceNodeManager
from scene_graph_ros.managers.object_manager import ObjectNodeManager


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


def _make_grid(data, resolution=1.0, origin_x=0.5, origin_y=0.5) -> OccupancyGrid:
    grid = OccupancyGrid()
    grid.info.width = len(data)
    grid.info.height = 1
    grid.info.resolution = resolution
    grid.info.origin.position.x = origin_x
    grid.info.origin.position.y = origin_y
    grid.info.origin.orientation.w = 1.0
    grid.data = list(data)
    return grid


def _make_grid_2d(rows, resolution=1.0, origin_x=-0.5, origin_y=-0.5) -> OccupancyGrid:
    grid = OccupancyGrid()
    grid.info.height = len(rows)
    grid.info.width = len(rows[0]) if rows else 0
    grid.info.resolution = resolution
    grid.info.origin.position.x = origin_x
    grid.info.origin.position.y = origin_y
    grid.info.origin.orientation.w = 1.0
    grid.data = [cell for row in rows for cell in row]
    return grid


def _make_manager(
    max_distance=2.0,
    *,
    cell_stride=2,
    navigation_connectivity=8,
) -> FreeSpaceNodeManager:
    sg = create_scene_graph_interface()
    manager = FreeSpaceNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        cell_stride_cells=cell_stride,
        min_free_cell_count=1,
        nearest_link_max_distance_m=max_distance,
        navigation_connectivity=navigation_connectivity,
        enable_debug_logging=False,
    )
    return manager


def _add_object(sg, x: float, y: float) -> ObjectNode:
    obj = ObjectNode()
    obj.pose.position.x = x
    obj.pose.position.y = y
    obj.pose.orientation.w = 1.0
    obj_id = sg.update.add_node(obj)
    obj.id = obj_id
    return obj


def _nearest_edges_for_object(sg, obj_id: int):
    return sg.query.get_outgoing_edges(obj_id, edge_type=EdgeType.NEAREST_FREE_SPACE)


def _single_nearest_edge(sg, obj_id: int):
    edges = _nearest_edges_for_object(sg, obj_id)
    assert len(edges) <= 1
    return edges[0] if edges else None


def _nav_target_xy(sg, edge):
    nav_node = sg.query.get_node(edge.target_id)
    assert nav_node is not None
    return (float(nav_node.pose.position.x), float(nav_node.pose.position.y))


def _nav_block_edge_pairs(manager: FreeSpaceNodeManager):
    node_to_block = {
        int(node_id): block_id for block_id, node_id in manager.block_to_node_id.items()
    }
    return {
        (node_to_block[int(edge.source_id)], node_to_block[int(edge.target_id)])
        for edge in manager.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH)
        if int(edge.source_id) in node_to_block and int(edge.target_id) in node_to_block
    }


def _make_detection_array(x: float, y: float, frame_id: str = "odom") -> Detection3DArray:
    msg = Detection3DArray()
    msg.header.frame_id = frame_id

    detection = Detection3D()
    detection.id = "chair"
    detection.bbox.center.position.x = x
    detection.bbox.center.position.y = y
    detection.bbox.center.orientation.w = 1.0

    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = "chair"
    hypothesis.hypothesis.score = 0.95
    detection.results.append(hypothesis)

    msg.detections.append(detection)
    return msg


def test_free_space_update_creates_nav_nodes_and_links():
    manager = _make_manager()

    stats = manager.process_occupancy_grid_update(_make_grid([0, 0, 0]), None)

    nav_nodes = manager.sg.query.find_nodes_by_type(NodeType.NAVIGATION)
    nav_links = manager.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH)

    assert stats["total_nav_nodes"] == 2
    assert stats["new_nav_nodes"] == 2
    assert len(nav_nodes) == 2
    assert len(nav_links) == 2


def test_navigation_edges_connect_horizontal_neighbors():
    manager = _make_manager(cell_stride=1)

    manager.process_occupancy_grid_update(
        _make_grid([0, 0], origin_x=-0.5, origin_y=-0.5),
        None,
    )

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 0)) in edge_pairs
    assert ((1, 0), (0, 0)) in edge_pairs


def test_navigation_edges_connect_vertical_neighbors():
    manager = _make_manager(cell_stride=1)

    manager.process_occupancy_grid_update(_make_grid_2d([[0], [0]]), None)

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (0, 1)) in edge_pairs
    assert ((0, 1), (0, 0)) in edge_pairs


def test_navigation_edges_connect_diagonal_neighbors_in_8_connectivity():
    manager = _make_manager(cell_stride=1, navigation_connectivity=8)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 0], [0, 0]]), None)

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 1)) in edge_pairs
    assert ((1, 1), (0, 0)) in edge_pairs


def test_navigation_edges_allow_diagonal_links_without_orthogonal_side_checks():
    manager = _make_manager(cell_stride=1, navigation_connectivity=8)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 100], [100, 0]]), None)

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 1)) in edge_pairs
    assert ((1, 1), (0, 0)) in edge_pairs


def test_navigation_edges_skip_diagonals_in_4_connectivity():
    manager = _make_manager(cell_stride=1, navigation_connectivity=4)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 0], [0, 0]]), None)

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 1)) not in edge_pairs
    assert ((1, 1), (0, 0)) not in edge_pairs


def test_navigation_edges_keep_diagonal_links_when_only_corner_cells_change():
    manager = _make_manager(cell_stride=1, navigation_connectivity=8)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 0], [0, 0]]), None)
    assert ((0, 0), (1, 1)) in _nav_block_edge_pairs(manager)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 100], [100, 0]]), None)

    edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 1)) in edge_pairs
    assert ((1, 1), (0, 0)) in edge_pairs


def test_navigation_edges_are_symmetric_and_not_duplicated():
    manager = _make_manager(cell_stride=1, navigation_connectivity=8)

    manager.process_occupancy_grid_update(_make_grid_2d([[0, 0], [0, 0]]), None)
    manager.process_occupancy_grid_update(_make_grid_2d([[0, 0], [0, 0]]), None)

    nav_edges = manager.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH)
    edge_pairs = _nav_block_edge_pairs(manager)

    assert len(nav_edges) == len(edge_pairs)
    for source_block, target_block in edge_pairs:
        assert source_block != target_block
        assert (target_block, source_block) in edge_pairs


def test_navigation_edges_remove_and_restore_blocked_coarse_connection():
    manager = _make_manager(cell_stride=2, navigation_connectivity=4)

    manager.process_occupancy_grid_update(
        _make_grid_2d(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        None,
    )
    assert ((0, 0), (1, 0)) in _nav_block_edge_pairs(manager)
    assert ((1, 0), (0, 0)) in _nav_block_edge_pairs(manager)

    manager.process_occupancy_grid_update(
        _make_grid_2d(
            [
                [0, 100, 100, 0],
                [0, 100, 100, 0],
            ]
        ),
        None,
    )
    blocked_edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 0)) not in blocked_edge_pairs
    assert ((1, 0), (0, 0)) not in blocked_edge_pairs

    manager.process_occupancy_grid_update(
        _make_grid_2d(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
        None,
    )
    restored_edge_pairs = _nav_block_edge_pairs(manager)

    assert ((0, 0), (1, 0)) in restored_edge_pairs
    assert ((1, 0), (0, 0)) in restored_edge_pairs


def test_navigation_nodes_persist_when_block_dips_below_minimum_free_count():
    sg = create_scene_graph_interface()
    manager = FreeSpaceNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        cell_stride_cells=2,
        min_free_cell_count=2,
        nearest_link_max_distance_m=2.0,
        navigation_connectivity=8,
        enable_debug_logging=False,
    )

    manager.process_occupancy_grid_update(
        _make_grid([0, 0], origin_x=-0.5, origin_y=-0.5),
        None,
    )

    assert len(manager.block_to_node_id) == 1
    block_id = next(iter(manager.block_to_node_id))
    node_id = manager.block_to_node_id[block_id]

    stats = manager.process_occupancy_grid_update(
        _make_grid([0, 100], origin_x=-0.5, origin_y=-0.5),
        None,
    )

    nav_node = manager.sg.query.get_node(node_id)

    assert stats["deleted_nav_nodes"] == 0
    assert block_id in manager.block_to_node_id
    assert nav_node is not None
    assert nav_node.attributes["free_cell_count"] == 1
    assert nav_node.attributes["meets_minimum_free_cells"] is False


def test_try_initial_nearest_link_targets_closest_nav_node():
    manager = _make_manager()
    manager.process_occupancy_grid_update(_make_grid([0, 0, 0]), None)

    obj = _add_object(manager.sg, x=1.1, y=1.0)
    manager.queue_object_ids_for_nearest_link([obj.id])

    changed = manager.try_initial_nearest_link(obj)
    edge = _single_nearest_edge(manager.sg, obj.id)

    assert changed is True
    assert edge is not None
    assert edge.source_id == obj.id
    assert _nav_target_xy(manager.sg, edge) == (1.0, 1.0)
    assert manager.has_pending_nearest_link_work() is False


def test_merged_redetection_does_not_report_new_object_ids_again():
    sg = create_scene_graph_interface()
    fs_manager = FreeSpaceNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        cell_stride_cells=2,
        min_free_cell_count=1,
        nearest_link_max_distance_m=2.0,
        enable_debug_logging=False,
    )
    obj_manager = ObjectNodeManager(
        sg_interface=sg,
        logger=MockLogger(),
        spatial_merge_threshold=0.75,
        enable_debug_logging=False,
    )

    fs_manager.process_occupancy_grid_update(_make_grid([0, 0, 0]), None)

    first_stats = obj_manager.process_detections_update(
        _make_detection_array(1.1, 1.0),
        tf_buffer=None,
        fixed_frame_id="odom",
    )
    first_object_id = first_stats["new_object_ids"][0]
    fs_manager.queue_object_ids_for_nearest_link(first_stats["new_object_ids"])
    obj_node = sg.query.get_node(first_object_id)
    assert obj_node is not None
    fs_manager.try_initial_nearest_link(obj_node)
    first_edge = _single_nearest_edge(sg, first_object_id)

    second_stats = obj_manager.process_detections_update(
        _make_detection_array(1.12, 1.0),
        tf_buffer=None,
        fixed_frame_id="odom",
    )
    second_edge = _single_nearest_edge(sg, first_object_id)

    assert first_stats["new_object_ids"] == [first_object_id]
    assert second_stats["new_object_ids"] == []
    assert len(sg.query.find_nodes_by_type(NodeType.OBJECT)) == 1
    assert first_edge is not None
    assert second_edge is not None
    assert second_edge.target_id == first_edge.target_id


def test_created_nav_block_only_requeues_nearby_objects_and_improves_link():
    manager = _make_manager()
    manager.process_occupancy_grid_update(_make_grid([0, 100, 100, 100, 100]), None)

    near_obj = _add_object(manager.sg, x=4.6, y=1.0)
    far_obj = _add_object(manager.sg, x=12.6, y=1.0)
    manager.queue_object_ids_for_nearest_link([near_obj.id, far_obj.id])
    manager.try_initial_nearest_link(near_obj)
    manager.try_initial_nearest_link(far_obj)

    assert _single_nearest_edge(manager.sg, near_obj.id) is None
    assert _single_nearest_edge(manager.sg, far_obj.id) is None

    map_stats = manager.process_occupancy_grid_update(_make_grid([0, 0, 0, 0, 100]), None)
    queued_object_ids = manager.drain_queued_object_ids()
    object_nodes = [manager.sg.query.get_node(obj_id) for obj_id in sorted(queued_object_ids)]
    object_nodes = [node for node in object_nodes if node is not None]
    manager.update_nearest_freespace_links_for_objects(object_nodes)

    improved_edge = _single_nearest_edge(manager.sg, near_obj.id)

    assert map_stats["created_block_ids"] == {(1, 0), (2, 0)}
    assert queued_object_ids == {near_obj.id}
    assert improved_edge is not None
    assert _nav_target_xy(manager.sg, improved_edge) == (5.0, 1.0)
    assert _single_nearest_edge(manager.sg, far_obj.id) is None


def test_tie_candidate_does_not_replace_existing_link():
    manager = _make_manager()
    manager.process_occupancy_grid_update(_make_grid([100, 0, 0, 100, 100]), None)

    obj = _add_object(manager.sg, x=3.5, y=1.0)
    manager.queue_object_ids_for_nearest_link([obj.id])
    manager.try_initial_nearest_link(obj)
    original_edge = _single_nearest_edge(manager.sg, obj.id)
    assert original_edge is not None
    assert _nav_target_xy(manager.sg, original_edge) == (3.0, 1.0)

    manager.process_occupancy_grid_update(_make_grid([100, 0, 0, 0, 0]), None)
    queued_object_ids = manager.drain_queued_object_ids()
    object_nodes = [manager.sg.query.get_node(obj_id) for obj_id in queued_object_ids]
    object_nodes = [node for node in object_nodes if node is not None]
    manager.update_nearest_freespace_links_for_objects(object_nodes)

    retained_edge = _single_nearest_edge(manager.sg, obj.id)

    assert queued_object_ids == {obj.id}
    assert retained_edge is not None
    assert retained_edge.target_id == original_edge.target_id
    assert _nav_target_xy(manager.sg, retained_edge) == (3.0, 1.0)


def test_block_below_threshold_relinks_objects_to_remaining_qualifying_nav_nodes():
    manager = _make_manager()
    manager.process_occupancy_grid_update(_make_grid([100, 0, 0, 0, 0]), None)

    obj = _add_object(manager.sg, x=4.6, y=1.0)
    manager.queue_object_ids_for_nearest_link([obj.id])
    manager.try_initial_nearest_link(obj)
    initial_edge = _single_nearest_edge(manager.sg, obj.id)
    assert initial_edge is not None
    assert _nav_target_xy(manager.sg, initial_edge) == (5.0, 1.0)

    map_stats = manager.process_occupancy_grid_update(_make_grid([100, 0, 0, 100, 100]), None)
    queued_object_ids = manager.drain_queued_object_ids()
    object_nodes = [manager.sg.query.get_node(obj_id) for obj_id in queued_object_ids]
    object_nodes = [node for node in object_nodes if node is not None]
    manager.update_nearest_freespace_links_for_objects(object_nodes)

    relinked_edge = _single_nearest_edge(manager.sg, obj.id)

    assert map_stats["removed_linked_object_ids"] == set()
    assert queued_object_ids == {obj.id}
    assert relinked_edge is not None
    assert _nav_target_xy(manager.sg, relinked_edge) == (3.0, 1.0)


def test_no_candidate_removes_stale_nearest_link():
    manager = _make_manager()
    manager.process_occupancy_grid_update(_make_grid([100, 0, 0, 100, 100]), None)

    obj = _add_object(manager.sg, x=4.2, y=1.0)
    manager.queue_object_ids_for_nearest_link([obj.id])
    manager.try_initial_nearest_link(obj)
    assert _single_nearest_edge(manager.sg, obj.id) is not None

    map_stats = manager.process_occupancy_grid_update(
        _make_grid([100, 100, 100, 100, 100]),
        None,
    )
    queued_object_ids = manager.drain_queued_object_ids()
    object_nodes = [manager.sg.query.get_node(obj_id) for obj_id in queued_object_ids]
    object_nodes = [node for node in object_nodes if node is not None]
    manager.update_nearest_freespace_links_for_objects(object_nodes)

    assert map_stats["removed_linked_object_ids"] == set()
    assert queued_object_ids == {obj.id}
    assert _nearest_edges_for_object(manager.sg, obj.id) == []


def test_resolution_change_requests_full_relink_once():
    manager = _make_manager(max_distance=2.5)
    manager.process_occupancy_grid_update(_make_grid([0, 0, 0], resolution=1.0), None)

    obj = _add_object(manager.sg, x=1.1, y=1.0)
    manager.queue_object_ids_for_nearest_link([obj.id])
    manager.try_initial_nearest_link(obj)
    assert _single_nearest_edge(manager.sg, obj.id) is not None

    map_stats = manager.process_occupancy_grid_update(
        _make_grid([0, 0, 0, 0, 0, 0], resolution=0.5),
        None,
    )

    indexed_count = manager.rebuild_object_block_index()
    manager.pending_full_relink = False
    manager.drain_queued_object_ids()
    manager.update_nearest_freespace_links_for_objects()

    assert map_stats["full_rescan_required"] is True
    assert indexed_count == 1
    assert _single_nearest_edge(manager.sg, obj.id) is not None
    assert manager.has_pending_nearest_link_work() is False


def test_origin_change_keeps_world_aligned_nav_pose_stable():
    manager = _make_manager(cell_stride=1)

    first_stats = manager.process_occupancy_grid_update(
        _make_grid([0], origin_x=-0.5, origin_y=-0.5),
        None,
    )
    nav_node_id = next(iter(manager.block_to_node_id.values()))
    first_nav_node = manager.sg.query.get_node(nav_node_id)
    assert first_nav_node is not None
    first_nav_xy = (
        float(first_nav_node.pose.position.x),
        float(first_nav_node.pose.position.y),
    )

    second_stats = manager.process_occupancy_grid_update(
        _make_grid([0], origin_x=0.0, origin_y=0.0),
        None,
    )
    second_nav_node = manager.sg.query.get_node(nav_node_id)

    assert first_stats["full_rescan_required"] is False
    assert first_nav_xy == (0.5, 0.5)
    assert second_stats["full_rescan_required"] is True
    assert second_nav_node is not None
    assert (second_nav_node.pose.position.x, second_nav_node.pose.position.y) == (
        0.5,
        0.5,
    )


def test_origin_change_rekeys_and_prunes_stale_nav_blocks():
    manager = _make_manager(cell_stride=1)

    manager.process_occupancy_grid_update(
        _make_grid([0], origin_x=-0.5, origin_y=-0.5),
        None,
    )
    old_nav_node_ids = set(manager.block_to_node_id.values())

    stats = manager.process_occupancy_grid_update(
        _make_grid([0], origin_x=1.5, origin_y=-0.5),
        None,
    )

    assert stats["full_rescan_required"] is True
    assert set(manager.block_to_node_id) == {(2, 0)}
    assert old_nav_node_ids.isdisjoint(set(manager.block_to_node_id.values()))


def test_world_to_block_id_uses_world_aligned_blocks():
    manager = _make_manager(cell_stride=1)

    manager.process_occupancy_grid_update(
        _make_grid_2d([[0]], resolution=1.0, origin_x=0.25, origin_y=0.25),
        None,
    )

    block_id = next(iter(manager.block_to_node_id))

    assert block_id == (0, 0)
    assert manager._world_to_block_id(0.3, 0.3) == (0, 0)
    assert manager._world_to_block_id(1.2, 1.2) == (1, 1)
