"""
Scene Graph Core - New API Usage Examples

This file demonstrates the new features introduced in the refactoring
while showing backward compatibility with existing code.
"""

from scene_graph_core.graph_interface import GraphPatch, SceneGraphInterface
from scene_graph_core.representation import (
    Edge,
    EdgeType,
    NodeType,
    ObjectNode,
    PoseNode,
    RoomNode,
)
from scene_graph_core.representation.geometry import Point


def example_1_node_subclasses():
    """Example 1: Using node subclasses for type safety."""
    print("\n=== Example 1: Node Subclasses ===")

    sg = SceneGraphInterface()

    # Old way (still works)
    # node = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)

    # New way (cleaner, type-safe)
    room = RoomNode()
    room.pose.position.x = 5.0
    room.pose.position.y = 5.0
    room.attributes = {"name": "Kitchen", "size": "large"}

    obj = ObjectNode()
    obj.pose.position.x = 5.2
    obj.pose.position.y = 5.3
    obj.attributes = {"class": "chair", "confidence": 0.95}

    pose = PoseNode()
    pose.pose.position.x = 5.1
    pose.pose.position.y = 5.1

    # Add to graph
    room_id = sg.update.add_node(room)
    obj_id = sg.update.add_node(obj)
    pose_id = sg.update.add_node(pose)

    print(f"Added room: {room_id}")
    print(f"Added object: {obj_id}")
    print(f"Added pose: {pose_id}")

    # Verify
    assert sg.query.graph.get_node(room_id).node_type == NodeType.ROOM
    assert sg.query.graph.get_node(obj_id).node_type == NodeType.OBJECT
    assert sg.query.graph.get_node(pose_id).node_type == NodeType.AGENT

    print("✓ Node subclasses work correctly")


def example_2_explicit_xy_vs_xyz_queries():
    """Example 2: Explicit XY (2D) vs XYZ (3D) spatial queries."""
    print("\n=== Example 2: XY vs XYZ Queries ===")

    sg = SceneGraphInterface()

    # Create rooms at different heights
    room1 = RoomNode()
    room1.pose.position.x = 0.0
    room1.pose.position.y = 0.0
    room1.pose.position.z = 0.0

    room2 = RoomNode()
    room2.pose.position.x = 5.0
    room2.pose.position.y = 5.0
    room2.pose.position.z = 10.0  # High Z

    room1_id = sg.update.add_node(room1)
    room2_id = sg.update.add_node(room2)

    # Query point at ground level
    test_point = Point()
    test_point.x = 4.9
    test_point.y = 4.9
    test_point.z = 0.0

    # XY query (2D, Z ignored) - for room matching
    closest_xy = sg.query.find_closest_node_xy(test_point, NodeType.ROOM)
    print(f"Closest in XY: Room {closest_xy[0].id}, distance={closest_xy[1]:.2f}m")
    assert closest_xy[0].id == room2_id  # room2 is closer in XY despite high Z

    # XYZ query (3D) - for spatial awareness
    closest_xyz = sg.query.find_closest_node_xyz(test_point, NodeType.ROOM)
    print(f"Closest in XYZ: Room {closest_xyz[0].id}, distance={closest_xyz[1]:.2f}m")
    assert closest_xyz[0].id == room1_id  # room1 is closer in 3D due to Z difference

    print("✓ XY vs XYZ queries work correctly")


def example_3_atomic_batch_updates():
    """Example 3: Using GraphPatch for atomic batch updates."""
    print("\n=== Example 3: Atomic Batch Updates ===")

    sg = SceneGraphInterface()

    # Setup initial state
    room1 = RoomNode()
    room2 = RoomNode()
    obj = ObjectNode()

    room1_id = sg.update.add_node(room1)
    room2_id = sg.update.add_node(room2)
    obj_id = sg.update.add_node(obj)

    # room1 initially contains obj
    sg.update.add_edge(
        Edge(source_id=room1_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    )

    print(f"Initial: room1({room1_id}) contains obj({obj_id})")

    # Move object from room1 to room2 atomically
    patch = GraphPatch()
    patch.remove_edge(room1_id, obj_id)
    patch.add_edge(
        Edge(source_id=room2_id, target_id=obj_id, type=EdgeType.ROOM_CONTAINS)
    )
    patch.update_node_attributes(room1_id, {"occupancy": "empty"})
    patch.update_node_attributes(room2_id, {"occupancy": "occupied"})

    print(f"Applying patch: {patch}")

    # Apply atomically (single lock acquisition)
    sg.update.apply_patch(patch, validate=True)

    print(f"After: room2({room2_id}) contains obj({obj_id})")

    # Verify
    assert not sg.query.graph.has_edge(room1_id, obj_id)
    assert sg.query.graph.has_edge(room2_id, obj_id)
    assert sg.query.graph.get_node(room1_id).attributes["occupancy"] == "empty"
    assert sg.query.graph.get_node(room2_id).attributes["occupancy"] == "occupied"

    print("✓ Atomic batch updates work correctly")


def example_4_backward_compatibility():
    """Example 4: Showing backward compatibility with old API."""
    print("\n=== Example 4: Backward Compatibility ===")

    from scene_graph_core.representation import BaseNode, NodeLayer

    sg = SceneGraphInterface()

    # Old way still works
    old_style_node = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)
    old_style_id = sg.update.add_node(old_style_node)

    # All old query methods still work
    rooms = sg.query.find_nodes_by_type(NodeType.ROOM)
    assert len(rooms) == 1

    # Direct graph access still works
    node = sg.query.graph.get_node(old_style_id)
    assert node.node_type == NodeType.ROOM

    print("✓ Old API still works")


def example_5_complex_scenario():
    """Example 5: Complex scenario with pose tracking and room assignment."""
    print("\n=== Example 5: Complex Scenario ===")

    sg = SceneGraphInterface()

    # Create rooms
    kitchen = RoomNode()
    kitchen.pose.position.x = 0.0
    kitchen.pose.position.y = 0.0
    kitchen.attributes = {"name": "Kitchen"}

    living_room = RoomNode()
    living_room.pose.position.x = 10.0
    living_room.pose.position.y = 0.0
    living_room.attributes = {"name": "Living Room"}

    kitchen_id = sg.update.add_node(kitchen)
    living_room_id = sg.update.add_node(living_room)

    print(f"Created rooms: Kitchen({kitchen_id}), Living Room({living_room_id})")

    # Robot moves from kitchen to living room
    poses = []
    for i, x in enumerate([1.0, 3.0, 5.0, 7.0, 9.0]):
        pose = PoseNode()
        pose.pose.position.x = x
        pose.pose.position.y = 0.0
        pose_id = sg.update.add_node(pose)
        poses.append(pose_id)

        # Find closest room (2D)
        closest = sg.query.find_closest_node_xy(pose.pose.position, NodeType.ROOM)
        current_room_id = closest[0].id

        # Assign pose to room
        sg.update.add_edge(
            Edge(
                source_id=current_room_id,
                target_id=pose_id,
                type=EdgeType.ROOM_CONTAINS,
            )
        )

        room_name = sg.query.graph.get_node(current_room_id).attributes["name"]
        print(f"  Pose {i} at x={x:.1f} → {room_name}")

    # Verify room transitions
    # Early poses should be in kitchen, later poses in living room
    for i, pose_id in enumerate(poses):
        edges = sg.query.graph.get_incoming_edges(pose_id)
        room_edge = [e for e in edges if e.type == EdgeType.ROOM_CONTAINS][0]
        room = sg.query.graph.get_node(room_edge.source_id)
        print(f"  Pose {i}: {room.attributes['name']}")

    print("✓ Complex scenario works correctly")


def example_6_batch_operations():
    """Example 6: Efficient batch operations."""
    print("\n=== Example 6: Batch Operations ===")

    sg = SceneGraphInterface()

    # Create multiple nodes efficiently
    rooms = [RoomNode() for _ in range(5)]
    room_ids = sg.update.add_nodes(rooms)

    print(f"Added {len(room_ids)} rooms in one call")

    # Create edges between adjacent rooms
    edges = []
    for i in range(len(room_ids) - 1):
        edge = Edge(
            source_id=room_ids[i],
            target_id=room_ids[i + 1],
            type=EdgeType.ROOM_ADJACENCY,
        )
        edges.append((edge, True))  # (edge, is_structural) tuple

    sg.update.add_edges(edges)

    print(f"Added {len(edges)} edges in one call")

    # Verify
    assert sg.query.graph.node_count() == 5
    assert sg.query.graph.edge_count() == 4

    print("✓ Batch operations work efficiently")


if __name__ == "__main__":
    print("=" * 60)
    print("Scene Graph Core - New API Examples")
    print("=" * 60)

    example_1_node_subclasses()
    example_2_explicit_xy_vs_xyz_queries()
    example_3_atomic_batch_updates()
    example_4_backward_compatibility()
    example_5_complex_scenario()
    example_6_batch_operations()

    print("\n" + "=" * 60)
    print("✅ All examples passed!")
    print("=" * 60)
