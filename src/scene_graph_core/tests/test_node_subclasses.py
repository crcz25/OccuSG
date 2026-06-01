"""
Test domain-specific node subclasses.

This test verifies that the new PoseNode, RoomNode, ObjectNode, and NavNode
classes properly enforce their type and layer invariants.
"""

import pytest
from scene_graph_core.representation import (
    BaseNode,
    NavNode,
    NodeLayer,
    NodeType,
    ObjectNode,
    PoseNode,
    RoomNode,
)
from scene_graph_core.representation.geometry import Pose


def test_pose_node_creation():
    """Test PoseNode creation and invariants."""
    # Create pose node
    pose_node = PoseNode()

    # Verify type and layer are automatically set
    assert pose_node.node_type == NodeType.AGENT
    assert pose_node.layer == NodeLayer.MOTION

    # Verify pose is initialized
    assert pose_node.pose is not None
    assert isinstance(pose_node.pose, Pose)


def test_room_node_creation():
    """Test RoomNode creation and invariants."""
    # Create room node
    room_node = RoomNode()

    # Verify type and layer are automatically set
    assert room_node.node_type == NodeType.ROOM
    assert room_node.layer == NodeLayer.SEMANTIC

    # Test with attributes
    room_node_with_attrs = RoomNode(attributes={"name": "Kitchen"})
    assert room_node_with_attrs.attributes["name"] == "Kitchen"


def test_object_node_creation():
    """Test ObjectNode creation and invariants."""
    # Create object node
    obj_node = ObjectNode()

    # Verify type and layer are automatically set
    assert obj_node.node_type == NodeType.OBJECT
    assert obj_node.layer == NodeLayer.OBJECT

    # Test with attributes
    obj_node_with_attrs = ObjectNode(attributes={"class": "chair", "confidence": 0.95})
    assert obj_node_with_attrs.attributes["class"] == "chair"
    assert obj_node_with_attrs.attributes["confidence"] == 0.95


def test_nav_region_node_creation():
    """Test NavNode creation and invariants."""
    # Create nav region node
    nav_node = NavNode()

    # Verify type and layer are automatically set
    assert nav_node.node_type == NodeType.NAVIGATION
    assert nav_node.layer == NodeLayer.NAVIGATION


def test_pose_node_type_enforcement():
    """Test that PoseNode enforces correct type/layer."""
    # This should fail - wrong node_type
    with pytest.raises(ValueError, match="PoseNode must have node_type=AGENT"):
        PoseNode(node_type=NodeType.OBJECT)

    # This should fail - wrong layer
    with pytest.raises(ValueError, match="PoseNode must have layer=MOTION"):
        PoseNode(layer=NodeLayer.SEMANTIC)


def test_room_node_type_enforcement():
    """Test that RoomNode enforces correct type/layer."""
    # This should fail - wrong node_type
    with pytest.raises(ValueError, match="RoomNode must have node_type=ROOM"):
        RoomNode(node_type=NodeType.OBJECT)

    # This should fail - wrong layer
    with pytest.raises(ValueError, match="RoomNode must have layer=SEMANTIC"):
        RoomNode(layer=NodeLayer.MOTION)


def test_object_node_type_enforcement():
    """Test that ObjectNode enforces correct type/layer."""
    # This should fail - wrong node_type
    with pytest.raises(ValueError, match="ObjectNode must have node_type=OBJECT"):
        ObjectNode(node_type=NodeType.ROOM)

    # This should fail - wrong layer
    with pytest.raises(ValueError, match="ObjectNode must have layer=OBJECT"):
        ObjectNode(layer=NodeLayer.SEMANTIC)


def test_nav_region_node_type_enforcement():
    """Test that NavNode enforces correct type/layer."""
    # This should fail - wrong node_type
    with pytest.raises(ValueError, match="NavNode must have node_type=NAVIGATION"):
        NavNode(node_type=NodeType.OBJECT)

    # This should fail - wrong layer
    with pytest.raises(ValueError, match="NavNode must have layer=NAVIGATION"):
        NavNode(layer=NodeLayer.SEMANTIC)


def test_subclass_serialization():
    """Test that subclasses can be serialized and deserialized."""
    # Create a pose node
    pose_node = PoseNode()
    pose_node.pose.position.x = 1.0
    pose_node.pose.position.y = 2.0
    pose_node.attributes = {"velocity": 0.5}

    # Serialize
    data = pose_node.to_dict()

    # Verify serialized data
    assert data["node_type"] == "AGENT"
    assert data["layer"] == "MOTION"
    assert data["pose"]["position"]["x"] == 1.0
    assert data["attributes"]["velocity"] == 0.5

    # Deserialize (as BaseNode for now - proper factory would be in serialization)
    loaded_node = BaseNode.from_dict(data)
    assert loaded_node.node_type == NodeType.AGENT
    assert loaded_node.layer == NodeLayer.MOTION
    assert loaded_node.pose.position.x == 1.0


def test_backwards_compatibility_with_base_node():
    """Test that BaseNode can still be used directly if needed."""
    # Create a BaseNode with explicit type/layer (old way)
    base_node = BaseNode(node_type=NodeType.ROOM, layer=NodeLayer.SEMANTIC)

    # Should work fine
    assert base_node.node_type == NodeType.ROOM
    assert base_node.layer == NodeLayer.SEMANTIC

    # Serialize and deserialize
    data = base_node.to_dict()
    loaded = BaseNode.from_dict(data)
    assert loaded.node_type == NodeType.ROOM
    assert loaded.layer == NodeLayer.SEMANTIC


if __name__ == "__main__":
    # Run tests directly for quick validation
    print("Testing domain-specific node subclasses...")

    print("\n1. Testing PoseNode creation...")
    test_pose_node_creation()
    print("   ✓ PoseNode creation passed")

    print("\n2. Testing RoomNode creation...")
    test_room_node_creation()
    print("   ✓ RoomNode creation passed")

    print("\n3. Testing ObjectNode creation...")
    test_object_node_creation()
    print("   ✓ ObjectNode creation passed")

    print("\n4. Testing NavNode creation...")
    test_nav_region_node_creation()
    print("   ✓ NavNode creation passed")

    print("\n5. Testing PoseNode type enforcement...")
    test_pose_node_type_enforcement()
    print("   ✓ PoseNode type enforcement passed")

    print("\n6. Testing RoomNode type enforcement...")
    test_room_node_type_enforcement()
    print("   ✓ RoomNode type enforcement passed")

    print("\n7. Testing ObjectNode type enforcement...")
    test_object_node_type_enforcement()
    print("   ✓ ObjectNode type enforcement passed")

    print("\n8. Testing NavNode type enforcement...")
    test_nav_region_node_type_enforcement()
    print("   ✓ NavNode type enforcement passed")

    print("\n9. Testing subclass serialization...")
    test_subclass_serialization()
    print("   ✓ Subclass serialization passed")

    print("\n10. Testing backwards compatibility...")
    test_backwards_compatibility_with_base_node()
    print("   ✓ Backwards compatibility passed")

    print("\n✅ All subclass tests passed!")
