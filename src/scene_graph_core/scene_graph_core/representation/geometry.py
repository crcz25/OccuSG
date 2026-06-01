"""Small geometry primitives used by scene_graph_core.

These classes intentionally mirror the field shape of ROS geometry messages
without importing ROS packages.
"""

from dataclasses import dataclass, field


@dataclass
class Point:
    """3D point with ROS-compatible x/y/z fields."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    """Quaternion with ROS-compatible x/y/z/w fields."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    """Pose with ROS-compatible position/orientation fields."""

    position: Point = field(default_factory=Point)
    orientation: Quaternion = field(default_factory=Quaternion)
