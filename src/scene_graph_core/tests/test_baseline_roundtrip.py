"""
Baseline test for scene graph round-trip serialization.

This test constructs a graph with nodes and edges, serializes it to a dictionary,
loads it back, and asserts that the node/edge counts match.

This serves as a baseline to ensure refactoring doesn't break basic functionality.
"""

import pytest

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import BaseNode, Edge, EdgeType, NodeLayer, NodeType
from scene_graph_core.representation.geometry import Pose


def test_roundtrip_serialization():
    """Test that a graph can be serialized and deserialized without data loss."""
    # Create a scene graph
    sg = SceneGraphInterface()

    # Create some nodes
    room_node = BaseNode(
        node_type=NodeType.ROOM,
        layer=NodeLayer.SEMANTIC,
    )
    room_node.pose = Pose()
    room_node.pose.position.x = 5.0
    room_node.pose.position.y = 5.0
    room_node.pose.position.z = 0.0
    room_node.attributes = {"name": "Kitchen"}

    obj_node = BaseNode(
        node_type=NodeType.OBJECT,
        layer=NodeLayer.OBJECT,
    )
    obj_node.pose = Pose()
    obj_node.pose.position.x = 5.2
    obj_node.pose.position.y = 5.3
    obj_node.pose.position.z = 0.5
    obj_node.attributes = {"class": "chair"}

    pose_node = BaseNode(
        node_type=NodeType.AGENT,
        layer=NodeLayer.MOTION,
    )
    pose_node.pose = Pose()
    pose_node.pose.position.x = 5.1
    pose_node.pose.position.y = 5.1
    pose_node.pose.position.z = 0.0

    # Add nodes to graph
    room_id = sg.update.add_node(room_node)
    obj_id = sg.update.add_node(obj_node)
    pose_id = sg.update.add_node(pose_node)

    # Create edges
    contains_edge = Edge(
        source_id=room_id,
        target_id=obj_id,
        type=EdgeType.ROOM_CONTAINS,
    )
    sg.update.add_edge(contains_edge)

    pose_in_room_edge = Edge(
        source_id=pose_id,
        target_id=room_id,
        type=EdgeType.TEMPORAL_LINK,
    )
    sg.update.add_edge(pose_in_room_edge)

    # Verify initial state
    assert len(sg.query.graph.get_all_nodes()) == 3
    assert len(sg.query.graph.get_all_edges()) == 2

    # Serialize to dictionary
    data = sg.serialize.to_dict()

    # Verify serialized data structure
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2

    # Create new scene graph and load data
    sg2 = SceneGraphInterface()
    sg2.serialize.from_dict(data)

    # Verify loaded state matches original
    assert len(sg2.query.graph.get_all_nodes()) == 3
    assert len(sg2.query.graph.get_all_edges()) == 2

    # Verify node types are preserved
    loaded_room = sg2.query.graph.get_node(room_id)
    assert loaded_room.node_type == NodeType.ROOM
    assert loaded_room.layer == NodeLayer.SEMANTIC
    assert loaded_room.attributes.get("name") == "Kitchen"

    loaded_obj = sg2.query.graph.get_node(obj_id)
    assert loaded_obj.node_type == NodeType.OBJECT
    assert loaded_obj.attributes.get("class") == "chair"

    loaded_pose = sg2.query.graph.get_node(pose_id)
    assert loaded_pose.node_type == NodeType.AGENT

    # Verify edges are preserved
    loaded_edges = sg2.query.graph.get_all_edges()
    edge_types = [e.type for e in loaded_edges]
    assert EdgeType.ROOM_CONTAINS in edge_types
    assert EdgeType.TEMPORAL_LINK in edge_types


def test_empty_graph_serialization():
    """Test serialization of an empty graph."""
    sg = SceneGraphInterface()

    # Serialize empty graph
    data = sg.serialize.to_dict()
    assert len(data["nodes"]) == 0
    assert len(data["edges"]) == 0

    # Load into new graph
    sg2 = SceneGraphInterface()
    sg2.serialize.from_dict(data)
    assert len(sg2.query.graph.get_all_nodes()) == 0
    assert len(sg2.query.graph.get_all_edges()) == 0


def test_node_id_consistency():
    """Test that node IDs are preserved through serialization."""
    sg = SceneGraphInterface()

    # Create nodes
    node1 = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)
    node2 = BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT)

    id1 = sg.update.add_node(node1)
    id2 = sg.update.add_node(node2)

    # Serialize and reload
    data = sg.serialize.to_dict()
    sg2 = SceneGraphInterface()
    sg2.serialize.from_dict(data)

    # Verify IDs are preserved
    assert sg2.query.graph.get_node(id1) is not None
    assert sg2.query.graph.get_node(id2) is not None
    assert sg2.query.graph.get_node(id1).node_type == NodeType.ROOM
    assert sg2.query.graph.get_node(id2).node_type == NodeType.OBJECT


if __name__ == "__main__":
    # Run tests directly for quick validation
    print("Running baseline round-trip tests...")

    print("\n1. Testing round-trip serialization...")
    test_roundtrip_serialization()
    print("   ✓ Round-trip serialization passed")

    print("\n2. Testing empty graph serialization...")
    test_empty_graph_serialization()
    print("   ✓ Empty graph serialization passed")

    print("\n3. Testing node ID consistency...")
    test_node_id_consistency()
    print("   ✓ Node ID consistency passed")

    print("\n✅ All baseline tests passed!")
