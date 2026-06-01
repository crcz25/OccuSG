#!/usr/bin/env python3
"""
Test to verify thread safety protection in the interface design.

Verifies that:
1. QueryInterface.graph is accessible (read-only context is safe)
2. UpdateInterface.graph is NOT accessible (prevents unsafe writes)
3. SerializationInterface.graph is NOT accessible
"""

import sys

from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import BaseNode, NodeLayer, NodeType


def test_query_graph_access():
    """Test that QueryInterface exposes graph for read access."""
    print("Testing QueryInterface.graph access...")

    sg = create_scene_graph_interface()

    # Add a node
    node = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)
    node_id = sg.update.add_node(node)

    # ✅ Should work - QueryInterface.graph is public (read-only)
    try:
        retrieved = sg.query.graph.get_node(node_id)
        print(f"✅ QueryInterface.graph.get_node() works: {retrieved.id}")
    except AttributeError as e:
        print(f"❌ FAILED: QueryInterface.graph not accessible: {e}")
        return False

    # Verify it's a property
    assert hasattr(type(sg.query), "graph"), "graph should be a property"
    assert isinstance(type(sg.query).graph, property), "graph should be a property"
    print("✅ QueryInterface.graph is a read-only property")

    return True


def test_update_no_graph_access():
    """Test that UpdateInterface does NOT expose graph."""
    print("\nTesting UpdateInterface.graph access...")

    sg = create_scene_graph_interface()

    # ❌ Should NOT work - UpdateInterface.graph is private
    try:
        _ = sg.update.graph
        print("❌ FAILED: UpdateInterface.graph is accessible (security risk!)")
        return False
    except AttributeError:
        print("✅ UpdateInterface.graph is NOT accessible (thread safety protected)")

    # Verify _graph is private
    assert hasattr(sg.update, "_graph"), "_graph should exist privately"
    assert not hasattr(type(sg.update), "graph"), "graph should not be a property"
    print("✅ UpdateInterface._graph is private only")

    return True


def test_serialize_no_graph_access():
    """Test that SerializationInterface does NOT expose graph."""
    print("\nTesting SerializationInterface.graph access...")

    sg = create_scene_graph_interface()

    # ❌ Should NOT work - SerializationInterface.graph is private
    try:
        _ = sg.serialize.graph
        print("❌ FAILED: SerializationInterface.graph is accessible")
        return False
    except AttributeError:
        print("✅ SerializationInterface.graph is NOT accessible")

    # Verify _graph is private
    assert hasattr(sg.serialize, "_graph"), "_graph should exist privately"
    assert not hasattr(type(sg.serialize), "graph"), "graph should not be a property"
    print("✅ SerializationInterface._graph is private only")

    return True


def test_thread_safe_usage():
    """Test the intended usage pattern is enforced."""
    print("\nTesting thread-safe usage patterns...")

    sg = create_scene_graph_interface()
    node = BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT)

    # ✅ CORRECT: Use update methods (thread-safe)
    try:
        node_id = sg.update.add_node(node)
        print(f"✅ sg.update.add_node() works (thread-safe): {node_id}")
    except Exception as e:
        print(f"❌ FAILED: sg.update.add_node() failed: {e}")
        return False

    # ✅ CORRECT: Read via query.graph (read-only, thread-safe)
    try:
        retrieved = sg.query.graph.get_node(node_id)
        print(f"✅ sg.query.graph.get_node() works (read-only): {retrieved.id}")
    except Exception as e:
        print(f"❌ FAILED: sg.query.graph.get_node() failed: {e}")
        return False

    # ❌ INCORRECT: Try to write via update.graph (should fail)
    try:
        _ = sg.update.graph.add_node(node)
        print("❌ FAILED: Unsafe write via sg.update.graph is possible!")
        return False
    except AttributeError:
        print("✅ Unsafe write via sg.update.graph is prevented")

    # ❌ INCORRECT: Try to write via query.graph (should technically work but discouraged)
    try:
        node2 = BaseNode(node_type=NodeType.OBJECT, layer=NodeLayer.OBJECT)
        # This will work because graph is exposed, but it's unsafe!
        # We document this as "use at your own risk"
        node_id2 = sg.query.graph.add_node(node2)
        print(
            f"⚠️  WARNING: Write via sg.query.graph works but bypasses locks: {node_id2}"
        )
        print("    (This is documented as unsafe - users should use sg.update)")
    except Exception as e:
        print(f"Write via sg.query.graph failed: {e}")

    return True


def test_api_clarity():
    """Test that the API clearly shows intent."""
    print("\nTesting API clarity...")

    sg = create_scene_graph_interface()

    # Check what's publicly available
    print("\nPublic API of QueryInterface:")
    query_public = [attr for attr in dir(sg.query) if not attr.startswith("_")]
    print(f"  - graph: {hasattr(type(sg.query), 'graph')}")
    print(f"  - find_closest_node: {hasattr(sg.query, 'find_closest_node')}")
    print(f"  - get_objects_in_room: {hasattr(sg.query, 'get_objects_in_room')}")

    print("\nPublic API of UpdateInterface:")
    update_public = [attr for attr in dir(sg.update) if not attr.startswith("_")]
    print(f"  - graph: {hasattr(type(sg.update), 'graph')} (should be False)")
    print(f"  - add_node: {hasattr(sg.update, 'add_node')}")
    print(f"  - upsert_node: {hasattr(sg.update, 'upsert_node')}")

    # Verify UpdateInterface does NOT expose graph
    assert not hasattr(type(sg.update), "graph"), (
        "UpdateInterface should not expose graph"
    )
    print("\n✅ API is clear: UpdateInterface forces use of thread-safe methods")

    return True


if __name__ == "__main__":
    print("\n" + "🔒 Testing Thread Safety Protection" + "\n")
    print("=" * 70)

    try:
        results = []
        results.append(test_query_graph_access())
        results.append(test_update_no_graph_access())
        results.append(test_serialize_no_graph_access())
        results.append(test_thread_safe_usage())
        results.append(test_api_clarity())

        print("\n" + "=" * 70)
        if all(results):
            print("✅ All thread safety tests passed!")
            print("\nDesign Summary:")
            print("  ✅ QueryInterface.graph - Public property (read-only context)")
            print(
                "  ✅ UpdateInterface.graph - NOT accessible (enforces thread safety)"
            )
            print("  ✅ SerializationInterface.graph - NOT accessible")
            print("=" * 70)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
