from typing import Optional

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose
from rclpy.duration import Duration
from visualization_msgs.msg import Marker


def _make_marker(
    ns: str,
    m_id: int,
    mtype: int,
    stamp: Time,
    frame: str,
    pose: Pose,
    scale: tuple[float, float, float],
    color: tuple[float, float, float, float],
    z_offset: float = 0.0,
    lifetime: float = 0.0,
    text: Optional[str] = None,
) -> Marker:
    """
    Create a visualization marker with the specified parameters.
    Args:
        ns (str): Namespace for the marker.
        m_id (int): Unique identifier for the marker.
        mtype (int): Type of the marker (e.g., Marker.SPHERE, Marker.CUBE).
        stamp (Time): Timestamp for the marker.
        frame (str): Frame ID for the marker.
        pose (Pose): Pose of the marker (ROS geometry_msgs/Pose).
        scale (tuple[float, float, float]): Scale of the marker.
        color (tuple[float, float, float, float]): Color of the marker in RGBA format.
        z_offset (float): Offset to apply to the z-coordinate of the marker's position.
        lifetime (float): Lifetime of the marker in seconds.
        text (Optional[str]): Text to display on the marker, if applicable.
    Returns:
        Marker: A visualization marker with the specified parameters.
    """
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = m_id
    m.type = mtype
    m.action = Marker.ADD
    m.pose = pose
    m.pose.position.z += z_offset
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.text = text if text is not None else ""
    m.lifetime = Duration(seconds=int(lifetime)).to_msg()
    return m


def _make_edge_marker(
    ns: str,
    m_id: int,
    stamp: Time,
    frame: str,
    points: list[Point],
    scale: float,
    color: tuple[float, float, float, float],
    lifetime: float = 0.0,
) -> Marker:
    """
    Create a line strip marker with the specified parameters.
    Args:
        ns (str): Namespace for the marker.
        m_id (int): Unique identifier for the marker.
        stamp (Time): Timestamp for the marker.
        frame (str): Frame ID for the marker.
        points (list[Point]): List of points defining the line strip.
        scale (float): Scale of the line width.
        color (tuple[float, float, float, float]): Color of the marker in RGBA format.
        lifetime (float): Lifetime of the marker in seconds.
    Returns:
        Marker: A line strip marker with the specified parameters.
    """
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns = ns
    m.id = m_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = scale
    m.points = points
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.lifetime = Duration(seconds=int(lifetime)).to_msg()
    return m
