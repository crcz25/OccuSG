"""
Representation module - Core data structures for Scene Graphs.

This module contains the fundamental building blocks:
- BaseNode: Node representation with pose, attributes, and layer information
- Subclasses: PoseNode, RoomNode, ObjectNode, NavNode (domain-specific nodes)
- Edge: Edge representation with type and attributes
- Pose: scene_graph_core Pose (position and orientation)
- NodeType, NodeLayer, EdgeType: Type enumerations
- SceneGraph: Storage wrapper around NetworkX graph
- Helper functions: pose_to_dict, pose_from_dict for serialization
"""

from .edge import Edge, EdgeType
from .free_space_region import FreeSpaceRegion
from .geometry import Point, Pose, Quaternion
from .graph import SceneGraph
from .node import (
    BaseNode,
    NavNode,
    NodeLayer,
    NodeType,
    ObjectNode,
    PoseNode,
    RegionNode,
    RoomNode,
    get_global_id,
    get_type_scoped_id,
    pose_from_dict,
    pose_to_dict,
)

__all__ = [
    "BaseNode",
    "PoseNode",
    "RoomNode",
    "ObjectNode",
    "NavNode",
    "RegionNode",
    "NodeType",
    "NodeLayer",
    "Point",
    "Quaternion",
    "Pose",
    "pose_to_dict",
    "pose_from_dict",
    "Edge",
    "EdgeType",
    "SceneGraph",
    "FreeSpaceRegion",
    "get_type_scoped_id",
    "get_global_id",
]
