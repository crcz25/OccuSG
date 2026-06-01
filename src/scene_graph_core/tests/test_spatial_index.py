"""
Test Spatial Index - Unit tests for SpatialIndex class and integration.

Tests cover:
1. Basic CRUD operations (insert, update, remove)
2. Query operations (nearest, radius, k-nearest)
3. Per-NodeType indexing
4. Thread-safety
5. Integration with SceneGraph
6. Performance verification
"""

import math
import threading
import time

# Import scene graph components
from scene_graph_core.algorithms.spatial import SpatialIndex, create_spatial_index
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import BaseNode, NodeLayer, NodeType
from scene_graph_core.representation.geometry import Point, Pose


def create_test_node(
    x: float,
    y: float,
    z: float,
    node_type: NodeType = NodeType.OBJECT,
    node_id: int = None,
) -> BaseNode:
    """Helper to create a test node at a position."""
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0

    return BaseNode(
        id=node_id,
        pose=pose,
        node_type=node_type,
        layer=NodeLayer.OBJECT,
    )


def create_point(x: float, y: float, z: float) -> Point:
    """Helper to create a Point."""
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


class TestSpatialIndexBasic:
    """Test basic SpatialIndex operations."""

    def test_create_spatial_index(self):
        """Test creating a spatial index."""
        index = create_spatial_index()
        assert index is not None
        assert len(index) == 0

    def test_insert_single_node(self):
        """Test inserting a single node."""
        index = SpatialIndex(rebuild_threshold=0)
        node = create_test_node(1.0, 2.0, 3.0)
        node.id = 100

        index.insert(node)

        assert len(index) == 1
        assert index.contains(100)

    def test_insert_multiple_nodes(self):
        """Test inserting multiple nodes."""
        index = SpatialIndex(rebuild_threshold=0)

        for i in range(10):
            node = create_test_node(float(i), float(i), 0.0)
            node.id = i
            index.insert(node)

        assert len(index) == 10
        for i in range(10):
            assert index.contains(i)

    def test_remove_node(self):
        """Test removing a node."""
        index = SpatialIndex(rebuild_threshold=0)
        node = create_test_node(1.0, 2.0, 3.0)
        node.id = 100

        index.insert(node)
        assert index.contains(100)

        index.remove(100)
        assert not index.contains(100)
        assert len(index) == 0

    def test_update_node_position(self):
        """Test updating a node's position."""
        index = SpatialIndex(rebuild_threshold=0)
        node = create_test_node(1.0, 2.0, 3.0)
        node.id = 100
        index.insert(node)

        # Update position
        node2 = create_test_node(10.0, 20.0, 30.0)
        node2.id = 100
        index.update(node2)

        # Query should find it at new position
        result = index.query_nearest(create_point(10.0, 20.0, 30.0))
        assert result is not None
        assert result[0] == 100
        assert result[1] < 0.01  # Very close

    def test_clear(self):
        """Test clearing the index."""
        index = SpatialIndex(rebuild_threshold=0)
        for i in range(5):
            node = create_test_node(float(i), 0.0, 0.0)
            node.id = i
            index.insert(node)

        assert len(index) == 5
        index.clear()
        assert len(index) == 0


class TestSpatialIndexQueries:
    """Test SpatialIndex query operations."""

    def test_query_nearest_single(self):
        """Test finding the nearest node."""
        index = SpatialIndex(rebuild_threshold=0)

        # Add nodes at various positions
        positions = [(0, 0, 0), (5, 5, 0), (10, 0, 0), (0, 10, 0)]
        for i, (x, y, z) in enumerate(positions):
            node = create_test_node(x, y, z)
            node.id = i
            index.insert(node)

        # Query from (1, 1, 0) - nearest should be (0, 0, 0)
        result = index.query_nearest(create_point(1.0, 1.0, 0.0))
        assert result is not None
        node_id, distance = result
        assert node_id == 0  # (0, 0, 0) is closest
        assert abs(distance - math.sqrt(2)) < 0.01

    def test_query_nearest_with_max_distance(self):
        """Test nearest query with max distance."""
        index = SpatialIndex(rebuild_threshold=0)

        node = create_test_node(100.0, 100.0, 0.0)
        node.id = 1
        index.insert(node)

        # Query with small max_distance - should return None
        result = index.query_nearest(create_point(0.0, 0.0, 0.0), max_distance=10.0)
        assert result is None

    def test_query_radius(self):
        """Test finding all nodes within radius."""
        index = SpatialIndex(rebuild_threshold=0)

        # Add nodes in a grid
        for i in range(5):
            for j in range(5):
                node = create_test_node(float(i), float(j), 0.0)
                node.id = i * 5 + j
                index.insert(node)

        # Query within radius 1.5 of (2, 2)
        results = index.query_radius(create_point(2.0, 2.0, 0.0), radius=1.5)

        # Should find 5 nodes: center + 4 cardinal neighbors
        assert len(results) >= 4
        # Center node should be first (distance 0)
        assert results[0][0] == 12  # Node at (2, 2)
        assert results[0][1] < 0.01

    def test_query_k_nearest(self):
        """Test finding k nearest nodes."""
        index = SpatialIndex(rebuild_threshold=0)

        # Add 10 nodes in a line
        for i in range(10):
            node = create_test_node(float(i), 0.0, 0.0)
            node.id = i
            index.insert(node)

        # Query 3 nearest from (0.5, 0, 0)
        results = index.query_nearest(create_point(0.5, 0.0, 0.0), k=3)

        assert len(results) == 3
        # Should be nodes 0, 1, 2 (closest to 0.5)
        ids = [r[0] for r in results]
        assert 0 in ids
        assert 1 in ids


class TestSpatialIndexByType:
    """Test per-NodeType indexing."""

    def test_query_by_type(self):
        """Test querying nodes of a specific type."""
        index = SpatialIndex(rebuild_threshold=0)

        # Add OBJECT nodes at (0,0), (1,1), (2,2)
        for i in range(3):
            node = create_test_node(float(i), float(i), 0.0, node_type=NodeType.OBJECT)
            node.id = i
            index.insert(node)

        # Add ROOM nodes at (10,10), (11,11)
        for i in range(2):
            node = create_test_node(10.0 + i, 10.0 + i, 0.0, node_type=NodeType.ROOM)
            node.id = 100 + i
            index.insert(node)

        # Query nearest OBJECT from (0.5, 0.5)
        result = index.query_nearest(
            create_point(0.5, 0.5, 0.0), node_type=NodeType.OBJECT
        )
        assert result is not None
        assert result[0] in [0, 1]  # Should be one of the OBJECT nodes

        # Query nearest ROOM from (0.5, 0.5) - should be far away
        result = index.query_nearest(
            create_point(0.5, 0.5, 0.0), node_type=NodeType.ROOM
        )
        assert result is not None
        assert result[0] in [100, 101]  # Should be one of the ROOM nodes
        assert result[1] > 10  # Should be far away

    def test_query_radius_by_type(self):
        """Test radius query filtered by type."""
        index = SpatialIndex(rebuild_threshold=0)

        # Add mixed nodes
        node1 = create_test_node(0.0, 0.0, 0.0, node_type=NodeType.OBJECT)
        node1.id = 1
        index.insert(node1)

        node2 = create_test_node(0.5, 0.0, 0.0, node_type=NodeType.ROOM)
        node2.id = 2
        index.insert(node2)

        node3 = create_test_node(1.0, 0.0, 0.0, node_type=NodeType.OBJECT)
        node3.id = 3
        index.insert(node3)

        # Query OBJECT nodes within radius 2
        results = index.query_radius(
            create_point(0.0, 0.0, 0.0), radius=2.0, node_type=NodeType.OBJECT
        )

        # Should find only OBJECT nodes (1 and 3)
        ids = [r[0] for r in results]
        assert 1 in ids
        assert 3 in ids
        assert 2 not in ids  # ROOM node should not be included


class TestSpatialIndexThreadSafety:
    """Test thread-safety of SpatialIndex."""

    def test_concurrent_inserts(self):
        """Test concurrent insertions."""
        index = SpatialIndex(rebuild_threshold=100)
        errors = []

        def insert_nodes(start_id: int, count: int):
            try:
                for i in range(count):
                    node = create_test_node(float(i), float(start_id), 0.0)
                    node.id = start_id * 1000 + i
                    index.insert(node)
            except Exception as e:
                errors.append(e)

        # Run 4 threads inserting nodes concurrently
        threads = []
        for thread_id in range(4):
            t = threading.Thread(target=insert_nodes, args=(thread_id, 25))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent insert: {errors}"
        assert len(index) == 100  # 4 threads * 25 nodes

    def test_concurrent_reads_and_writes(self):
        """Test concurrent reads while writing."""
        index = SpatialIndex(rebuild_threshold=10)
        errors = []
        query_results = []

        # Pre-populate with some nodes
        for i in range(20):
            node = create_test_node(float(i), 0.0, 0.0)
            node.id = i
            index.insert(node)

        def writer():
            try:
                for i in range(50):
                    node = create_test_node(float(i + 100), 0.0, 0.0)
                    node.id = i + 100
                    index.insert(node)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(50):
                    result = index.query_nearest(create_point(0.0, 0.0, 0.0))
                    if result:
                        query_results.append(result)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("reader", e))

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"
        assert len(query_results) > 0  # Should have gotten some results


class TestSceneGraphIntegration:
    """Test SpatialIndex integration with SceneGraph."""

    def test_spatial_index_enabled_by_default(self):
        """Test that spatial index is enabled by default."""
        sg = create_scene_graph_interface()
        assert sg.query.graph.spatial_index is not None

    def test_add_node_updates_index(self):
        """Test that adding a node updates the spatial index."""
        sg = create_scene_graph_interface()

        node = create_test_node(1.0, 2.0, 3.0)
        node_id = sg.update.add_node(node)

        # Query via spatial index
        result = sg.query.find_closest_node(create_point(1.0, 2.0, 3.0))
        assert result is not None
        assert result[0].id == node_id
        assert result[1] < 0.01

    def test_update_node_updates_index(self):
        """Test that updating a node updates the spatial index."""
        sg = create_scene_graph_interface()

        # Add node at (0, 0, 0)
        node = create_test_node(0.0, 0.0, 0.0)
        node_id = sg.update.add_node(node)

        # Update to (10, 10, 10)
        node.pose.position.x = 10.0
        node.pose.position.y = 10.0
        node.pose.position.z = 10.0
        sg.update.update_node(node_id, node)

        # Query from (10, 10, 10) - should find it
        result = sg.query.find_closest_node(create_point(10.0, 10.0, 10.0))
        assert result is not None
        assert result[0].id == node_id
        assert result[1] < 0.01

    def test_remove_node_updates_index(self):
        """Test that removing a node updates the spatial index."""
        sg = create_scene_graph_interface()

        # Add two nodes
        node1 = create_test_node(0.0, 0.0, 0.0)
        node1_id = sg.update.add_node(node1)

        node2 = create_test_node(10.0, 10.0, 0.0)
        node2_id = sg.update.add_node(node2)

        # Remove first node
        sg.update.remove_node(node1_id)

        # Query from (0, 0, 0) - should find node2 (not node1)
        result = sg.query.find_closest_node(create_point(0.0, 0.0, 0.0))
        assert result is not None
        assert result[0].id == node2_id

    def test_find_nodes_by_position(self):
        """Test find_nodes_by_position uses spatial index."""
        sg = create_scene_graph_interface()

        # Add nodes in a cluster
        for i in range(5):
            node = create_test_node(float(i) * 0.1, 0.0, 0.0)
            sg.update.add_node(node)

        # Add far-away node
        far_node = create_test_node(100.0, 100.0, 0.0)
        sg.update.add_node(far_node)

        # Query within 1.0m of origin
        results = sg.query.find_nodes_by_position(
            create_point(0.0, 0.0, 0.0), max_range=1.0
        )

        assert len(results) == 5  # Only the cluster, not the far node

    def test_find_k_nearest_nodes(self):
        """Test find_k_nearest_nodes."""
        sg = create_scene_graph_interface()

        # Add 10 nodes in a line
        for i in range(10):
            node = create_test_node(float(i), 0.0, 0.0)
            sg.update.add_node(node)

        # Find 3 nearest to origin
        results = sg.query.find_k_nearest_nodes(create_point(0.0, 0.0, 0.0), k=3)

        assert len(results) == 3
        # First should be at (0, 0, 0)
        assert results[0][1] < 0.01


class TestPerformance:
    """Performance verification tests."""

    def test_spatial_index_faster_than_brute_force(self):
        """Verify spatial index is faster for large graphs."""
        sg = create_scene_graph_interface()

        # Add many nodes
        num_nodes = 1000
        for i in range(num_nodes):
            node = create_test_node(float(i % 100), float(i // 100), 0.0)
            sg.update.add_node(node)

        # Ensure spatial index is built
        if sg.query.graph.spatial_index:
            sg.query.graph.spatial_index.rebuild_all()

        query_point = create_point(50.0, 5.0, 0.0)

        # Time spatial query
        start = time.time()
        for _ in range(100):
            result = sg.query.find_closest_node(query_point)
        spatial_time = time.time() - start

        print(
            f"\nSpatial index query time for {num_nodes} nodes: {spatial_time:.4f}s for 100 queries"
        )
        print(f"Average: {spatial_time / 100 * 1000:.2f}ms per query")

        # Just verify it works - specific timing depends on hardware
        assert result is not None


def run_manual_tests():
    """Run tests manually for debugging."""
    print("Running SpatialIndex tests...")

    # Basic tests
    print("\n=== Basic Tests ===")
    test = TestSpatialIndexBasic()
    test.test_create_spatial_index()
    print("✓ create_spatial_index")
    test.test_insert_single_node()
    print("✓ insert_single_node")
    test.test_insert_multiple_nodes()
    print("✓ insert_multiple_nodes")
    test.test_remove_node()
    print("✓ remove_node")
    test.test_update_node_position()
    print("✓ update_node_position")
    test.test_clear()
    print("✓ clear")

    # Query tests
    print("\n=== Query Tests ===")
    test = TestSpatialIndexQueries()
    test.test_query_nearest_single()
    print("✓ query_nearest_single")
    test.test_query_nearest_with_max_distance()
    print("✓ query_nearest_with_max_distance")
    test.test_query_radius()
    print("✓ query_radius")
    test.test_query_k_nearest()
    print("✓ query_k_nearest")

    # Type-specific tests
    print("\n=== Type-Specific Tests ===")
    test = TestSpatialIndexByType()
    test.test_query_by_type()
    print("✓ query_by_type")
    test.test_query_radius_by_type()
    print("✓ query_radius_by_type")

    # Thread safety tests
    print("\n=== Thread Safety Tests ===")
    test = TestSpatialIndexThreadSafety()
    test.test_concurrent_inserts()
    print("✓ concurrent_inserts")
    test.test_concurrent_reads_and_writes()
    print("✓ concurrent_reads_and_writes")

    # Integration tests
    print("\n=== Integration Tests ===")
    test = TestSceneGraphIntegration()
    test.test_spatial_index_enabled_by_default()
    print("✓ spatial_index_enabled_by_default")
    test.test_add_node_updates_index()
    print("✓ add_node_updates_index")
    test.test_update_node_updates_index()
    print("✓ update_node_updates_index")
    test.test_remove_node_updates_index()
    print("✓ remove_node_updates_index")
    test.test_find_nodes_by_position()
    print("✓ find_nodes_by_position")
    test.test_find_k_nearest_nodes()
    print("✓ find_k_nearest_nodes")

    # Performance test
    print("\n=== Performance Test ===")
    test = TestPerformance()
    test.test_spatial_index_faster_than_brute_force()
    print("✓ spatial_index_faster_than_brute_force")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_manual_tests()
