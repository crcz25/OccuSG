#!/usr/bin/env python3
"""Tests for typed edge queries and index consistency in scene_graph_core."""

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    Edge,
    EdgeType,
    ObjectNode,
    PoseNode,
    RoomNode,
)


def test_get_all_edges_filtered_by_type():
    sg = create_scene_graph_interface()

    room = RoomNode()
    pose = PoseNode()
    obj = ObjectNode()

    room_id = sg.update.add_node(room)
    pose_id = sg.update.add_node(pose)
    obj_id = sg.update.add_node(obj)

    sg.update.add_edge(
        Edge(source_id=room_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    )
    sg.update.add_edge(
        Edge(source_id=room_id, target_id=pose_id, type=EdgeType.ROOM_CONTAINS)
    )

    room_contains = sg.query.get_all_edges(EdgeType.ROOM_CONTAINS)
    region_contains = sg.query.get_all_edges(EdgeType.REGION_CONTAINS)
    room_adj = sg.query.get_all_edges(EdgeType.ROOM_ADJACENCY)

    assert len(room_contains) == 2
    assert {(edge.source_id, edge.target_id) for edge in room_contains} == {
        (room_id, obj_id),
        (room_id, pose_id),
    }
    assert region_contains == []
    assert room_adj == []


def test_edge_type_index_consistent_after_node_removal():
    sg = create_scene_graph_interface()

    room_a = RoomNode()
    room_b = RoomNode()
    room_a_id = sg.update.add_node(room_a)
    room_b_id = sg.update.add_node(room_b)

    sg.update.add_edge(
        Edge(source_id=room_a_id, target_id=room_b_id, type=EdgeType.ROOM_ADJACENCY),
        is_structural=False,
    )
    assert len(sg.query.get_all_edges(EdgeType.ROOM_ADJACENCY)) == 1

    sg.update.remove_node(room_b_id)
    assert sg.query.get_all_edges(EdgeType.ROOM_ADJACENCY) == []
    assert not sg.query.has_node(room_b_id)
    assert sg.query.get_node(room_b_id) is None


def test_query_outgoing_incoming_with_edge_type_filter():
    sg = create_scene_graph_interface()

    room = RoomNode()
    pose = PoseNode()
    obj = ObjectNode()

    room_id = sg.update.add_node(room)
    pose_id = sg.update.add_node(pose)
    obj_id = sg.update.add_node(obj)

    sg.update.add_edge(
        Edge(source_id=room_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    )
    sg.update.add_edge(
        Edge(source_id=room_id, target_id=pose_id, type=EdgeType.ROOM_CONTAINS)
    )

    out_edges = sg.query.get_outgoing_edges(room_id, EdgeType.ROOM_CONTAINS)
    in_edges = sg.query.get_incoming_edges(obj_id, EdgeType.ROOM_CONTAINS)

    assert len(out_edges) == 2
    assert {edge.target_id for edge in out_edges} == {obj_id, pose_id}
    assert len(in_edges) == 1
    assert in_edges[0].source_id == room_id
