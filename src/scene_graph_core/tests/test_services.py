"""
Test the new QueryService and UpdateService.

This test verifies the new services layer with explicit XY vs XYZ semantics.
"""

from scene_graph_core.representation import (
    Edge,
    EdgeType,
    ObjectNode,
    PoseNode,
    RoomNode,
    SceneGraph,
)
from scene_graph_core.representation.geometry import Point
from scene_graph_core.services import QueryService, UpdateService


def test_query_service_basics():
    """Test basic query service operations."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Add some nodes
    room = RoomNode()
    room.pose.position.x = 5.0
    room.pose.position.y = 5.0

    obj = ObjectNode()
    obj.pose.position.x = 5.2
    obj.pose.position.y = 5.3

    room_id = update.add_node(room)
    obj_id = update.add_node(obj)

    # Query by type
    rooms = query.find_nodes_by_type(room.node_type)
    assert len(rooms) == 1
    assert rooms[0].id == room_id

    # Direct graph access
    assert query.graph.has_node(room_id)
    assert query.graph.has_node(obj_id)


def test_spatial_queries_xy():
    """Test 2D spatial queries (XY only)."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Create rooms at different positions
    room1 = RoomNode()
    room1.pose.position.x = 0.0
    room1.pose.position.y = 0.0
    room1.pose.position.z = 0.0

    room2 = RoomNode()
    room2.pose.position.x = 5.0
    room2.pose.position.y = 5.0
    room2.pose.position.z = 10.0  # High Z, but should be ignored in XY queries

    room3 = RoomNode()
    room3.pose.position.x = 10.0
    room3.pose.position.y = 10.0
    room3.pose.position.z = 0.0

    room1_id = update.add_node(room1)
    room2_id = update.add_node(room2)
    room3_id = update.add_node(room3)

    # Find closest room using XY distance only
    test_point = Point()
    test_point.x = 4.9
    test_point.y = 4.9
    test_point.z = 0.0  # At ground level

    closest = query.find_closest_node_xy(test_point, room1.node_type)
    assert closest is not None
    assert closest[0].id == room2_id  # room2 is closest in XY despite high Z

    # XY distance should be ~0.14, not affected by Z=10
    assert closest[1] < 0.2

    # Find nodes in 2D radius
    nearby = query.find_nodes_by_position_xy(test_point, max_range=6.0)
    # room1 is at (0,0), distance = sqrt(4.9^2 + 4.9^2) = ~6.9 > 6.0
    # room2 is at (5,5), distance = sqrt(0.1^2 + 0.1^2) = ~0.14 < 6.0
    # room3 is at (10,10), distance = sqrt(5.1^2 + 5.1^2) = ~7.2 > 6.0
    assert len(nearby) == 1  # Only room2
    assert nearby[0][0].id == room2_id  # Closest


def test_spatial_queries_xyz():
    """Test 3D spatial queries (XYZ)."""
    graph = SceneGraph(enable_spatial_index=True)  # Enable spatial index
    query = QueryService(graph)
    update = UpdateService(graph)

    # Create objects at different positions
    obj1 = ObjectNode()
    obj1.pose.position.x = 0.0
    obj1.pose.position.y = 0.0
    obj1.pose.position.z = 0.0

    obj2 = ObjectNode()
    obj2.pose.position.x = 1.0
    obj2.pose.position.y = 1.0
    obj2.pose.position.z = 10.0  # High Z

    obj1_id = update.add_node(obj1)
    obj2_id = update.add_node(obj2)

    # Find closest using 3D distance
    test_point = Point()
    test_point.x = 0.9
    test_point.y = 0.9
    test_point.z = 0.0

    closest = query.find_closest_node_xyz(test_point, obj1.node_type)
    assert closest is not None
    # obj1 is closer in 3D because obj2 is at Z=10
    assert closest[0].id == obj1_id

    # Find in 3D radius
    nearby = query.find_nodes_by_position_xyz(test_point, max_range=2.0)
    assert len(nearby) == 1  # Only obj1, obj2 is too far in Z
    assert nearby[0][0].id == obj1_id


def test_update_service_edge_operations():
    """Test edge operations in UpdateService."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Create nodes
    pose = PoseNode()
    room1 = RoomNode()
    room2 = RoomNode()

    pose_id = update.add_node(pose)
    room1_id = update.add_node(room1)
    room2_id = update.add_node(room2)

    # Add edge
    edge = Edge(
        source_id=pose_id,
        target_id=room1_id,
        type=EdgeType.ROOM_CONTAINS,
    )
    update.add_edge(edge)

    # Verify edge exists
    assert query.graph.has_edge(pose_id, room1_id)

    # Replace edge (move pose to room2)
    update.replace_edge_of_type(pose_id, EdgeType.ROOM_CONTAINS, room2_id)

    # Verify old edge removed, new edge added
    assert not query.graph.has_edge(pose_id, room1_id)
    assert query.graph.has_edge(pose_id, room2_id)

    # Remove edge
    removed = update.remove_edge(pose_id, room2_id, EdgeType.ROOM_CONTAINS)
    assert removed
    assert not query.graph.has_edge(pose_id, room2_id)


def test_update_service_batch_operations():
    """Test batch operations for efficiency."""
    graph = SceneGraph()
    query = QueryService(graph)
    update = UpdateService(graph)

    # Add multiple nodes at once
    nodes = [RoomNode() for _ in range(5)]
    node_ids = update.add_nodes(nodes)

    assert len(node_ids) == 5
    assert query.graph.node_count() == 5

    # Add multiple edges
    edges = []
    for i in range(4):
        edge = Edge(
            source_id=node_ids[i],
            target_id=node_ids[i + 1],
            type=EdgeType.ROOM_CONTAINS,
        )
        edges.append(edge)

    update.add_edges(edges)
    assert query.graph.edge_count() == 4

    # Remove multiple nodes
    update.remove_nodes([node_ids[0], node_ids[1]])
    assert query.graph.node_count() == 3


def test_update_service_typed_edge_helpers():
    """Typed edge helpers should use graph fast paths without changing semantics."""
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

    outgoing = query.get_outgoing_edges(room_id, EdgeType.ROOM_CONTAINS)
    incoming = query.get_incoming_edges(obj_id, EdgeType.ROOM_CONTAINS)

    assert len(outgoing) == 1
    assert outgoing[0].target_id == obj_id
    assert len(incoming) == 1
    assert incoming[0].source_id == room_id


def test_thread_safety():
    """Test that update operations are thread-safe."""
    import threading

    graph = SceneGraph()
    update = UpdateService(graph)

    def add_nodes():
        for _ in range(100):
            node = RoomNode()
            update.add_node(node)

    # Run multiple threads
    threads = [threading.Thread(target=add_nodes) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have 300 nodes with no corruption
    query = QueryService(graph)
    assert query.graph.node_count() == 300


if __name__ == "__main__":
    print("Testing QueryService and UpdateService...")

    print("\n1. Testing query service basics...")
    test_query_service_basics()
    print("   ✓ Query service basics passed")

    print("\n2. Testing spatial queries (XY)...")
    test_spatial_queries_xy()
    print("   ✓ XY spatial queries passed")

    print("\n3. Testing spatial queries (XYZ)...")
    test_spatial_queries_xyz()
    print("   ✓ XYZ spatial queries passed")

    print("\n4. Testing edge operations...")
    test_update_service_edge_operations()
    print("   ✓ Edge operations passed")

    print("\n5. Testing batch operations...")
    test_update_service_batch_operations()
    print("   ✓ Batch operations passed")

    print("\n6. Testing thread safety...")
    test_thread_safety()
    print("   ✓ Thread safety passed")

    print("\n✅ All service tests passed!")
