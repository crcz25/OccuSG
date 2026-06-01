"""
Scene Graph Core Library

A modular library for 3D Scene Graph construction, manipulation, and querying.
Designed for robotics applications with foreground (real-time) and background
(optimization) processing separation.

Main components:
- representation: Core data structures (Node, Edge, Graph)
- graph_interface: Shared interface for concurrent access
- foreground: Real-time perception and incremental updates
- background: Asynchronous optimization and consistency checking
- algorithms: Reusable algorithms (spatial, graph operations)
- queries: High-level query API

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Carlos Cueto Zumaya"

from scene_graph_core.representation.edge import Edge, EdgeType
from scene_graph_core.serialization import SceneGraphJsonSerializer

# Import main classes for convenient access
from scene_graph_core.representation.geometry import Point, Pose, Quaternion
from scene_graph_core.representation.node import BaseNode, NodeLayer, NodeType

__all__ = [
    "BaseNode",
    "NodeType",
    "NodeLayer",
    "Point",
    "Pose",
    "Quaternion",
    "Edge",
    "EdgeType",
    "SceneGraphJsonSerializer",
    "__version__",
]
