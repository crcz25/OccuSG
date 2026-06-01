"""Lightweight runtime diagnostics for scene-graph pose ingestion."""

from __future__ import annotations

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class PoseDiagnosticsNode(Node):
    """Report odometry receipt and visible pose-marker counts."""

    def __init__(self):
        super().__init__("scene_graph_pose_diagnostics")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scene_graph_topic", "/dsg/scene_graph")
        self.declare_parameter("report_hz", 1.0)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.scene_graph_topic = self.get_parameter("scene_graph_topic").value
        report_hz = max(0.1, float(self.get_parameter("report_hz").value))

        self.odom_count = 0
        self.last_odom_stamp_sec = None
        self.last_odom_frame_id = ""
        self.last_child_frame_id = ""
        self.pose_marker_count = 0
        self.pose_edge_marker_count = 0
        self.marker_msg_count = 0

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            10,
        )
        self.create_subscription(
            MarkerArray,
            self.scene_graph_topic,
            self._marker_callback,
            10,
        )
        self.create_timer(1.0 / report_hz, self._report)

        self.get_logger().info(
            "Pose diagnostics listening on "
            f"odom='{self.odom_topic}' scene_graph='{self.scene_graph_topic}'"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_count += 1
        self.last_odom_stamp_sec = (
            float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1e9
        )
        self.last_odom_frame_id = str(msg.header.frame_id)
        self.last_child_frame_id = str(msg.child_frame_id)

    def _marker_callback(self, msg: MarkerArray) -> None:
        self.marker_msg_count += 1
        pose_count = 0
        pose_edge_count = 0
        for marker in msg.markers:
            if marker.action == Marker.DELETE:
                continue
            if marker.ns == "pose_layer":
                pose_count += len(marker.points) if marker.points else 1
            elif marker.ns == "pose_link":
                pose_edge_count += max(1, len(marker.points) // 2)
        self.pose_marker_count = pose_count
        self.pose_edge_marker_count = pose_edge_count

    def _report(self) -> None:
        self.get_logger().info(
            "[pose_diagnostics] "
            f"odom_received={self.odom_count} "
            f"last_odom_stamp={self.last_odom_stamp_sec} "
            f"frame_id='{self.last_odom_frame_id}' "
            f"child_frame_id='{self.last_child_frame_id}' "
            f"marker_msgs={self.marker_msg_count} "
            f"visible_pose_markers={self.pose_marker_count} "
            f"visible_pose_edges={self.pose_edge_marker_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PoseDiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
