"""
Test GraphPatch for atomic batch updates.
"""

from scene_graph_core.representation import (
    Edge,
    EdgeType,
    ObjectNode,
    PoseNode,
    RoomNode,
    SceneGraph,
)
from scene_graph_core.services import GraphPatch, QueryService, UpdateService


def test_graph_patch_basic():
    """Test basic GraphPatch operations."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Create a patch
    patch = GraphPatch()

    # Add nodes
    room = RoomNode()
    obj = ObjectNode()
    patch.add_node(room).add_node(obj)

    assert len(patch.nodes_to_add) == 2
    assert not patch.is_empty()
    assert patch.size() == 2

    # Apply patch
    update.apply_patch(patch)

    # Verify nodes were added
    assert query.graph.node_count() == 2


def test_graph_patch_with_edges():
    """Test GraphPatch with nodes and edges."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # First, add some nodes directly
    room = RoomNode()
    obj = ObjectNode()
    room_id = update.add_node(room)
    obj_id = update.add_node(obj)

    # Create patch to add edge
    patch = GraphPatch()
    edge = Edge(source_id=room_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    patch.add_edge(edge)

    # Apply patch
    update.apply_patch(patch)

    # Verify edge was added
    assert query.graph.has_edge(room_id, obj_id)


def test_graph_patch_remove_operations():
    """Test GraphPatch removal operations."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Add some nodes and edges
    room1 = RoomNode()
    room2 = RoomNode()
    room1_id = update.add_node(room1)
    room2_id = update.add_node(room2)

    edge = Edge(source_id=room1_id, target_id=room2_id, type=EdgeType.ROOM_CONTAINS)
    update.add_edge(edge)

    # Create patch to remove edge and node
    patch = GraphPatch()
    patch.remove_edge(room1_id, room2_id)
    patch.remove_node(room2_id)

    # Apply patch
    update.apply_patch(patch)

    # Verify removals
    assert not query.graph.has_edge(room1_id, room2_id)
    assert not query.graph.has_node(room2_id)
    assert query.graph.has_node(room1_id)


def test_graph_patch_attribute_updates():
    """Test GraphPatch attribute updates."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Add a room
    room = RoomNode()
    room.attributes = {"name": "Kitchen"}
    room_id = update.add_node(room)

    # Create patch to update attributes
    patch = GraphPatch()
    patch.update_node_attributes(room_id, {"name": "Living Room", "size": "large"})

    # Apply patch
    update.apply_patch(patch)

    # Verify attributes were updated
    updated_room = query.graph.get_node(room_id)
    assert updated_room.attributes["name"] == "Living Room"
    assert updated_room.attributes["size"] == "large"


def test_graph_patch_validation():
    """Test GraphPatch validation."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Create patch with invalid operations
    patch = GraphPatch()
    patch.remove_node(999)  # Non-existent node

    # Should fail validation
    try:
        update.apply_patch(patch, validate=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "non-existent" in str(e).lower()


def test_graph_patch_atomic_application():
    """Test that patch application is atomic (all or nothing)."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Add a room
    room = RoomNode()
    room_id = update.add_node(room)

    # Create patch with one valid and one invalid operation
    patch = GraphPatch()
    patch.remove_node(room_id)  # Valid
    patch.remove_edge(999, 888)  # Invalid

    # Should fail validation and not apply any changes
    try:
        update.apply_patch(patch, validate=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Room should still exist (atomic failure)
    assert query.graph.has_node(room_id)


def test_graph_patch_complex_scenario():
    """Test a complex scenario with multiple operations."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Setup: Add some initial nodes and edges
    room1 = RoomNode()
    room2 = RoomNode()
    obj1 = ObjectNode()
    obj2 = ObjectNode()

    room1_id = update.add_node(room1)
    room2_id = update.add_node(room2)
    obj1_id = update.add_node(obj1)
    obj2_id = update.add_node(obj2)

    # room1 contains obj1
    update.add_edge(
        Edge(source_id=room1_id, target_id=obj1_id, type=EdgeType.ROOM_CONTAINS)
    )

    # Create patch to:
    # 1. Remove obj1 from room1
    # 2. Add obj1 to room2
    # 3. Add new pose node
    # 4. Update room1 attributes
    patch = GraphPatch()
    patch.remove_edge(room1_id, obj1_id)
    patch.add_edge(
        Edge(source_id=room2_id, target_id=obj1_id, type=EdgeType.ROOM_CONTAINS)
    )

    pose = PoseNode()
    patch.add_node(pose)

    patch.update_node_attributes(room1_id, {"occupancy": "empty"})

    # Apply atomically
    update.apply_patch(patch)

    # Verify all changes
    assert not query.graph.has_edge(room1_id, obj1_id)
    assert query.graph.has_edge(room2_id, obj1_id)
    assert query.graph.node_count() == 5  # 2 rooms + 2 objects + 1 pose

    room1_updated = query.graph.get_node(room1_id)
    assert room1_updated.attributes.get("occupancy") == "empty"


def test_graph_patch_string_representation():
    """Test GraphPatch string representation."""
    patch = GraphPatch()
    assert "empty" in str(patch)

    patch.add_node(RoomNode())
    patch.remove_node(123)
    assert "+1 nodes" in str(patch)
    assert "-1 nodes" in str(patch)


def test_graph_patch_preserves_relational_edge_semantics():
    """Patch-added relational edges should remain non-structural."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    pose = PoseNode()
    obj = ObjectNode()
    pose_id = update.add_node(pose)
    obj_id = update.add_node(obj)

    patch = GraphPatch()
    patch.add_edge(
        Edge(
            source_id=pose_id,
            target_id=obj_id,
            type=EdgeType.OBSERVATION_ANCHOR,
            is_structural=False,
        ),
        is_structural=False,
    )
    update.apply_patch(patch)

    edge = query.graph.get_edge(pose_id, obj_id)
    assert edge.type == EdgeType.OBSERVATION_ANCHOR
    assert edge.is_structural is False


def test_graph_patch_removes_typed_edges():
    """Typed edge removal should remove only the matching edge type."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    room = RoomNode()
    obj = ObjectNode()
    room_id = update.add_node(room)
    obj_id = update.add_node(obj)

    update.add_edge(
        Edge(source_id=room_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    )
    patch = GraphPatch()
    patch.remove_edge(room_id, obj_id, EdgeType.ROOM_CONTAINS)
    update.apply_patch(patch)
    assert not query.graph.has_edge(room_id, obj_id)


if __name__ == "__main__":
    print("Testing GraphPatch...")

    print("\n1. Testing basic GraphPatch operations...")
    test_graph_patch_basic()
    print("   ✓ Basic operations passed")

    print("\n2. Testing patch with edges...")
    test_graph_patch_with_edges()
    print("   ✓ Edge operations passed")

    print("\n3. Testing removal operations...")
    test_graph_patch_remove_operations()
    print("   ✓ Removal operations passed")

    print("\n4. Testing attribute updates...")
    test_graph_patch_attribute_updates()
    print("   ✓ Attribute updates passed")

    print("\n5. Testing validation...")
    test_graph_patch_validation()
    print("   ✓ Validation passed")

    print("\n6. Testing atomic application...")
    test_graph_patch_atomic_application()
    print("   ✓ Atomic application passed")

    print("\n7. Testing complex scenario...")
    test_graph_patch_complex_scenario()
    print("   ✓ Complex scenario passed")

    print("\n8. Testing string representation...")
    test_graph_patch_string_representation()
    print("   ✓ String representation passed")

    print("\n✅ All GraphPatch tests passed!")
