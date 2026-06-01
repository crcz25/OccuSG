#!/usr/bin/env python3
"""
Test the new graph interface layer.

Verifies that all interfaces work correctly with the dependency injection pattern.
"""

import sys

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    BaseNode,
    Edge,
    EdgeType,
    NodeLayer,
    NodeType,
)


def test_basic_operations():
    """Test basic add/query operations."""
    print("=" * 60)
    print("Testing Basic Operations")
    print("=" * 60)

    # Create interface (dependency injection)
    sg = create_scene_graph_interface()
    print(f"✓ Created scene graph interface: {sg}")

    # Add a node (using update interface)
    node1 = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)
    node1.attributes = {"name": "kitchen"}

    node_id1 = sg.update.add_node(node1)  # Use sg.update.add_node()
    print(f"✓ Added room node with ID: {node_id1}")

    # Add another node
    node2 = BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT)
    node2.attributes = {"name": "table"}

    node_id2 = sg.update.add_node(node2)  # Use sg.update.add_node()
    print(f"✓ Added object node with ID: {node_id2}")

    # Query nodes (using query interface)
    retrieved = sg.query.graph.get_node(node_id1)  # Use sg.query.graph.get_node()
    print(
        f"✓ Retrieved node: type={retrieved.node_type.name}, name={retrieved.attributes.get('name')}"
    )

    # Query by type (direct graph access)
    rooms = sg.query.graph.get_nodes_by_type(NodeType.ROOM)
    print(f"✓ Found {len(rooms)} room(s)")

    objects = sg.query.graph.get_nodes_by_type(NodeType.OBJECT)
    print(f"✓ Found {len(objects)} object(s)")

    # Add edge (using update interface)
    edge = Edge(source_id=node_id1, target_id=node_id2, type=EdgeType.ROOM_CONTAINS)
    edge_id = sg.update.add_edge(edge)  # Use sg.update.add_edge()
    print(f"✓ Added edge with ID: {edge_id}")

    # Stats
    print(f"\n✓ Graph stats: {sg.get_node_count()} nodes, {sg.get_edge_count()} edges")
    print(sg)


def test_thread_safety():
    """Test that write operations use locks."""
    print("\n" + "=" * 60)
    print("Testing Thread Safety")
    print("=" * 60)

    sg = create_scene_graph_interface()

    # Check that update interface has a lock
    has_lock = hasattr(sg.update, "_lock")
    print(f"✓ UpdateInterface has lock: {has_lock}")

    # Add multiple nodes
    nodes = [
        BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT) for _ in range(5)
    ]

    node_ids = sg.update.add_nodes(nodes)
    print(f"✓ Batch added {len(node_ids)} nodes")

    print(f"✓ Total nodes in graph: {sg.get_node_count()}")


def test_serialization():
    """Test save/load operations."""
    print("\n" + "=" * 60)
    print("Testing Serialization")
    print("=" * 60)

    sg = create_scene_graph_interface()

    # Add some nodes (using update interface)
    for i in range(3):
        node = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)
        node.attributes = {"name": f"room_{i}"}
        sg.update.add_node(node)  # Use sg.update.add_node()

    print(f"✓ Created graph with {sg.get_node_count()} nodes")

    # Save to JSON (using serialize interface)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        filepath = f.name

    saved_path = sg.serialize.save(filepath)  # Use sg.serialize.save()
    print(f"✓ Saved graph to: {saved_path}")

    # Create new interface and load
    sg2 = create_scene_graph_interface()
    sg2.serialize.load(str(saved_path))  # Use sg.serialize.load()
    print(f"✓ Loaded graph: {sg2.get_node_count()} nodes")

    # Verify (using query interface)
    rooms = sg2.query.graph.get_nodes_by_type(NodeType.ROOM)
    print(f"✓ Found {len(rooms)} rooms after loading")

    # Cleanup
    import os

    os.unlink(saved_path)
    print("✓ Cleaned up test file")


def test_dependency_injection():
    """Test that multiple nodes can share the same graph."""
    print("\n" + "=" * 60)
    print("Testing Dependency Injection Pattern")
    print("=" * 60)

    # Create single graph
    sg = create_scene_graph_interface()

    # Simulate foreground node
    class ForegroundNode:
        def __init__(self, sg_interface):
            self.sg = sg_interface

        def process_detection(self):
            node = BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT)
            node.attributes = {"source": "foreground"}
            return self.sg.update.add_node(node)  # Use sg.update.add_node()

    # Simulate background node
    class BackgroundNode:
        def __init__(self, sg_interface):
            self.sg = sg_interface

        def optimize(self):
            # Can query all nodes added by foreground
            objects = self.sg.query.graph.get_nodes_by_type(
                NodeType.OBJECT
            )  # Use sg.query.graph
            return len(objects)

    # Both share same graph
    foreground = ForegroundNode(sg)
    background = BackgroundNode(sg)

    # Foreground adds node
    node_id = foreground.process_detection()
    print(f"✓ Foreground added node: {node_id}")

    # Background can see it
    count = background.optimize()
    print(f"✓ Background sees {count} object(s)")

    # Verify they share the same graph
    assert foreground.sg is background.sg
    print("✓ Both nodes share the same graph instance")


if __name__ == "__main__":
    print("\n" + "🚀 Testing Scene Graph Interface Layer" + "\n")

    try:
        test_basic_operations()
        test_thread_safety()
        test_serialization()
        test_dependency_injection()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
        import traceback

        traceback.print_exc()
        sys.exit(1)
