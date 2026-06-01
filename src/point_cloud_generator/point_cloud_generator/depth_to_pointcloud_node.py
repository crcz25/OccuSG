#!/usr/bin/env python3
"""
ROS 2 Node: Depth Image to PointCloud2 Converter.

Converts synchronized depth images and camera info into 3D point clouds.
Supports both 16UC1 (mm) and 32FC1 (meters) depth encodings.
"""

from array import array
import signal
import time
from typing import Optional

import message_filters
import numpy as np
import rclpy
import rclpy.logging
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from scene_graph_ros.profiling import ProfilingRecorder
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


class DepthToPointCloudNode(Node):
    """Convert depth images into PointCloud2 messages in real time."""

    def __init__(self):
        """Initialize ROS interfaces and optimized projection pipeline."""
        super().__init__("depth_to_pointcloud_node")

        # Core topics and projection parameters.
        self.declare_parameter("depth_image_topic", "/camera/depth/image_raw")
        self.declare_parameter(
            "camera_info_topic",
            "/camera/depth/camera_info",
        )
        self.declare_parameter("output_frame", "camera_depth_optical_frame")
        self.declare_parameter("pointcloud_topic", "/pointcloud")
        self.declare_parameter("stride", 1)
        self.declare_parameter("min_depth", 0.1)
        self.declare_parameter("max_depth", 10.0)
        self.declare_parameter("target_hz", 3.0)  # 0 => unlimited

        # Synchronization and QoS knobs for better real-time behavior.
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("qos_depth", 10)
        self.declare_parameter(
            "input_qos_reliability",
            "best_effort",
        )
        self.declare_parameter("output_qos_reliability", "reliable")
        self.declare_parameter("enable_profiling", False)
        self.declare_parameter("profiling_output_path", "")
        self.declare_parameter("profiling_run_name", "run")
        self.declare_parameter("profiling_save_on_shutdown", True)
        self.declare_parameter("profiling_discard_first_n", 5)

        self.depth_topic = self.get_parameter("depth_image_topic").value
        self.camera_info_topic = self.get_parameter(
            "camera_info_topic"
        ).value
        self.output_frame = self.get_parameter("output_frame").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.stride = int(self.get_parameter("stride").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.target_hz = float(self.get_parameter("target_hz").value)
        self.sync_queue_size = int(
            self.get_parameter("sync_queue_size").value
        )
        self.sync_slop_sec = float(
            self.get_parameter("sync_slop_sec").value
        )
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.input_qos_reliability = str(
            self.get_parameter("input_qos_reliability").value
        )
        self.output_qos_reliability = str(
            self.get_parameter("output_qos_reliability").value
        )
        self.profiler = ProfilingRecorder(
            node_name="point_cloud_generator_node",
            package_name="point_cloud_generator",
            run_name=str(self.get_parameter("profiling_run_name").value),
            output_path=str(self.get_parameter("profiling_output_path").value),
            enabled=bool(self.get_parameter("enable_profiling").value),
            save_on_shutdown=bool(
                self.get_parameter("profiling_save_on_shutdown").value
            ),
            discard_first_n=int(
                self.get_parameter("profiling_discard_first_n").value
            ),
            file_tag="point_cloud_generator",
        )
        if self.profiler.enabled:
            self.get_logger().info(
                "Runtime profiling enabled: "
                f"{self.profiler.output_path}/"
                f"{self.profiler.run_name}.point_cloud_generator.json"
            )

        if self.stride < 1:
            self.get_logger().error(
                f"Invalid stride: {self.stride}. Must be >= 1. Setting to 1."
            )
            self.stride = 1

        if self.min_depth < 0.0:
            self.get_logger().error(
                f"Invalid min_depth: {self.min_depth}. "
                "Must be >= 0. Setting to 0.0."
            )
            self.min_depth = 0.0

        if self.max_depth <= self.min_depth:
            self.get_logger().error(
                f"Invalid depth range [{self.min_depth}, {self.max_depth}]. "
                f"Setting max_depth={self.min_depth + 10.0}."
            )
            self.max_depth = self.min_depth + 10.0

        if self.target_hz < 0.0:
            self.get_logger().error(
                f"Invalid target_hz: {self.target_hz}. "
                "Must be >= 0. Setting to 0."
            )
            self.target_hz = 0.0

        if self.sync_queue_size < 1:
            self.get_logger().error(
                f"Invalid sync_queue_size: {self.sync_queue_size}. "
                "Must be >= 1. Setting to 10."
            )
            self.sync_queue_size = 10

        if self.sync_slop_sec < 0.0:
            self.get_logger().error(
                f"Invalid sync_slop_sec: {self.sync_slop_sec}. "
                "Must be >= 0. Setting to 0.05."
            )
            self.sync_slop_sec = 0.05

        if self.qos_depth < 1:
            self.get_logger().error(
                f"Invalid qos_depth: {self.qos_depth}. "
                "Must be >= 1. Setting to 10."
            )
            self.qos_depth = 10

        # Enforce >0 depth even if min_depth is configured to zero.
        self.min_valid_depth = float(
            max(self.min_depth, np.nextafter(np.float32(0.0), np.float32(1.0)))
        )
        self.mm_to_m = np.float32(0.001)

        # Rate limiter.
        self.last_publish_time = self.get_clock().now()
        if self.target_hz > 0.0:
            self.min_publish_interval = 1.0 / self.target_hz
        else:
            self.min_publish_interval = 0.0

        # Camera intrinsics.
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.camera_info_received = False

        # Projection cache (recomputed only on shape/intrinsics changes).
        self._projection_cache_valid = False
        self._cache_width = -1
        self._cache_height = -1
        self._x_factors = np.empty((0,), dtype=np.float32)
        self._y_factors = np.empty((0,), dtype=np.float32)

        # Static PointCloud2 schema for XYZ32.
        self._point_fields = [
            PointField(
                name="x", offset=0, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="y", offset=4, datatype=PointField.FLOAT32, count=1
            ),
            PointField(
                name="z", offset=8, datatype=PointField.FLOAT32, count=1
            ),
        ]
        self._point_step_bytes = 12

        # Fallback converter for malformed depth payloads.
        self.bridge = CvBridge()

        input_reliability = self._parse_reliability(
            self.input_qos_reliability,
            fallback=QoSReliabilityPolicy.BEST_EFFORT,
            param_name="input_qos_reliability",
        )
        output_reliability = self._parse_reliability(
            self.output_qos_reliability,
            fallback=QoSReliabilityPolicy.RELIABLE,
            param_name="output_qos_reliability",
        )
        input_qos_depth = max(self.qos_depth, self.sync_queue_size)

        input_qos = QoSProfile(
            reliability=input_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=input_qos_depth,
        )
        output_qos = QoSProfile(
            reliability=output_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
        )

        self.pc_pub = self.create_publisher(
            PointCloud2,
            self.pointcloud_topic,
            output_qos,
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=input_qos,
        )
        self.camera_info_sub = message_filters.Subscriber(
            self,
            CameraInfo,
            self.camera_info_topic,
            qos_profile=input_qos,
        )
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.depth_sub, self.camera_info_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.msg_count = 0
        self._profiling_samples_since_checkpoint = 0
        self.last_log_time = self.get_clock().now()

        self._log_configuration()
        self.get_logger().info(
            "Depth to PointCloud node initialized. Waiting for messages..."
        )

    def _log_configuration(self) -> None:
        """Log startup parameters."""
        self.get_logger().info(
            "=== Depth to PointCloud Node Configuration ==="
        )
        self.get_logger().info(f"  Depth topic:       {self.depth_topic}")
        self.get_logger().info(
            f"  Camera info topic: {self.camera_info_topic}"
        )
        self.get_logger().info(f"  Output frame:      {self.output_frame}")
        self.get_logger().info(f"  PointCloud topic:  {self.pointcloud_topic}")
        self.get_logger().info(f"  Stride:            {self.stride}")
        self.get_logger().info(
            f"  Depth range:       [{self.min_depth}, {self.max_depth}] meters"
        )
        self.get_logger().info(
            f"  Target rate:       {self.target_hz:.1f} Hz"
            if self.target_hz > 0.0
            else "  Target rate:       unlimited"
        )
        self.get_logger().info(
            "  Sync:              "
            f"queue={self.sync_queue_size}, slop={self.sync_slop_sec:.3f}s"
        )
        self.get_logger().info(
            "  QoS:               "
            f"in={self.input_qos_reliability}, "
            f"out={self.output_qos_reliability}, "
            f"depth={self.qos_depth}"
        )
        self.get_logger().info("=" * 47)

    def save_profiling(self) -> None:
        """Flush profiling data if profiling is enabled."""
        if not getattr(self, "profiler", None) or not self.profiler.enabled:
            return

        path = self.profiler.save()
        if path is not None:
            self.get_logger().info(f"Runtime profiling saved: {path}")

    def _parse_reliability(
        self,
        value: str,
        fallback: QoSReliabilityPolicy,
        param_name: str,
    ) -> QoSReliabilityPolicy:
        """Parse reliability policy from string parameter."""
        normalized = value.strip().lower()
        if normalized in ("reliable",):
            return QoSReliabilityPolicy.RELIABLE
        if normalized in ("best_effort", "best-effort", "besteffort"):
            return QoSReliabilityPolicy.BEST_EFFORT

        self.get_logger().error(
            f"Invalid {param_name}: '{value}'. "
            "Supported values: reliable, best_effort. "
            f"Using fallback '{fallback.name.lower()}'."
        )
        return fallback

    def synchronized_callback(
        self,
        depth_msg: Image,
        camera_info_msg: CameraInfo,
    ):
        """Process synchronized depth + camera info messages."""
        callback_start = time.perf_counter()
        if (
            not self.camera_info_received
            or self._camera_info_changed(camera_info_msg)
        ):
            self._update_camera_intrinsics(camera_info_msg)

        if not self.camera_info_received:
            return

        current_time = self.get_clock().now()
        time_since_last = (
            current_time - self.last_publish_time
        ).nanoseconds / 1e9
        if (
            self.target_hz > 0.0
            and time_since_last < self.min_publish_interval
        ):
            return

        try:
            conversion_start = time.perf_counter()
            pointcloud_msg = self._depth_to_pointcloud(depth_msg)
            conversion_ms = (time.perf_counter() - conversion_start) * 1000.0
            if pointcloud_msg is None:
                return

            self.pc_pub.publish(pointcloud_msg)
            callback_ms = (time.perf_counter() - callback_start) * 1000.0
            self.profiler.record(
                "point_cloud_generation_ms",
                conversion_ms,
                metadata={"published_points": int(pointcloud_msg.width)},
            )
            self.profiler.record(
                "point_cloud_callback_total_ms",
                callback_ms,
                metadata={"published_points": int(pointcloud_msg.width)},
            )
            if self.profiler.enabled:
                self._profiling_samples_since_checkpoint += 1
                if self._profiling_samples_since_checkpoint <= 5:
                    self.profiler.save_checkpoint()
                elif self._profiling_samples_since_checkpoint % 25 == 0:
                    self.profiler.save_checkpoint()
            self.last_publish_time = current_time
            self._update_stats(current_time)

        except Exception as exc:
            self.get_logger().error(
                f"Error processing depth image: {exc}",
                throttle_duration_sec=2.0,
            )

    def _update_stats(self, current_time) -> None:
        """Emit periodic runtime statistics."""
        self.msg_count += 1
        time_diff = (current_time - self.last_log_time).nanoseconds / 1e9
        if time_diff < 5.0:
            return

        hz = self.msg_count / time_diff
        self.get_logger().debug(
            "Publishing point clouds at "
            f"{hz:.1f} Hz (frame: {self.output_frame})"
        )
        self.msg_count = 0
        self.last_log_time = current_time

    def _camera_info_changed(self, camera_info_msg: CameraInfo) -> bool:
        """Check whether intrinsics changed."""
        K = camera_info_msg.k
        return (
            self.fx != K[0]
            or self.fy != K[4]
            or self.cx != K[2]
            or self.cy != K[5]
        )

    def _update_camera_intrinsics(self, camera_info_msg: CameraInfo) -> None:
        """Extract and validate camera intrinsics."""
        K = camera_info_msg.k
        fx = float(K[0])
        fy = float(K[4])
        cx = float(K[2])
        cy = float(K[5])

        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warning(
                "Invalid camera intrinsics: "
                f"fx={fx}, fy={fy}. Skipping frame.",
                throttle_duration_sec=2.0,
            )
            self.camera_info_received = False
            self._projection_cache_valid = False
            return

        intrinsics_changed = (
            not self.camera_info_received
            or self.fx != fx
            or self.fy != fy
            or self.cx != cx
            or self.cy != cy
        )

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.camera_info_received = True

        if intrinsics_changed:
            self._projection_cache_valid = False
            self.get_logger().debug(
                f"Camera intrinsics updated: "
                f"fx={self.fx:.3f}, fy={self.fy:.3f}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f}"
            )

    def _decode_image_buffer(
        self,
        depth_msg: Image,
        dtype: np.dtype,
        bytes_per_pixel: int,
    ) -> Optional[np.ndarray]:
        """Decode ROS Image buffer into a 2D array with zero-copy view."""
        height = int(depth_msg.height)
        width = int(depth_msg.width)
        step = int(depth_msg.step)

        if height < 1 or width < 1:
            self.get_logger().warning(
                "Invalid depth image shape: "
                f"{width}x{height}. Skipping frame.",
                throttle_duration_sec=2.0,
            )
            return None

        row_bytes = width * bytes_per_pixel
        if step < row_bytes:
            self.get_logger().error(
                "Invalid step="
                f"{step} for width={width}, bpp={bytes_per_pixel}.",
                throttle_duration_sec=2.0,
            )
            return None

        required_bytes = step * height
        if len(depth_msg.data) < required_bytes:
            self.get_logger().error(
                "Depth payload too small: "
                f"{len(depth_msg.data)} < {required_bytes}.",
                throttle_duration_sec=2.0,
            )
            return None

        endian = ">" if depth_msg.is_bigendian else "<"
        view_dtype = np.dtype(dtype).newbyteorder(endian)
        try:
            image = np.ndarray(
                shape=(height, width),
                dtype=view_dtype,
                buffer=depth_msg.data,
                strides=(step, bytes_per_pixel),
            )
        except Exception as exc:
            self.get_logger().warning(
                f"Zero-copy decode failed ({exc}). Falling back to cv_bridge.",
                throttle_duration_sec=5.0,
            )
            return None

        if not image.dtype.isnative:
            image = image.byteswap().newbyteorder()

        return image

    def _decode_with_cv_bridge(self, depth_msg: Image) -> Optional[np.ndarray]:
        """Fallback decoder for malformed payload layouts."""
        try:
            if depth_msg.encoding == "16UC1":
                depth_image = self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding="16UC1",
                )
                return depth_image.astype(np.float32) * self.mm_to_m

            if depth_msg.encoding == "32FC1":
                depth_image = self.bridge.imgmsg_to_cv2(
                    depth_msg,
                    desired_encoding="32FC1",
                )
                return depth_image.astype(np.float32, copy=False)
        except Exception as exc:
            self.get_logger().error(
                f"cv_bridge conversion failed: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        return None

    def _decode_depth_image(self, depth_msg: Image) -> Optional[np.ndarray]:
        """Decode depth image to float32 meters."""
        if depth_msg.encoding == "16UC1":
            depth_u16 = self._decode_image_buffer(
                depth_msg=depth_msg,
                dtype=np.uint16,
                bytes_per_pixel=2,
            )
            if depth_u16 is None:
                return self._decode_with_cv_bridge(depth_msg)
            return depth_u16.astype(np.float32) * self.mm_to_m

        if depth_msg.encoding == "32FC1":
            depth_f32 = self._decode_image_buffer(
                depth_msg=depth_msg,
                dtype=np.float32,
                bytes_per_pixel=4,
            )
            if depth_f32 is None:
                return self._decode_with_cv_bridge(depth_msg)
            return depth_f32.astype(np.float32, copy=False)

        self.get_logger().error(
            f"Unsupported depth encoding: {depth_msg.encoding}. "
            "Supported encodings: 16UC1, 32FC1.",
            throttle_duration_sec=2.0,
        )
        return None

    def _ensure_projection_cache(self, width: int, height: int) -> None:
        """Build cached (u-cx)/fx and (v-cy)/fy arrays."""
        if (
            self._projection_cache_valid
            and self._cache_width == width
            and self._cache_height == height
        ):
            return

        u_coords = np.arange(0, width, self.stride, dtype=np.float32)
        v_coords = np.arange(0, height, self.stride, dtype=np.float32)
        if u_coords.size == 0 or v_coords.size == 0:
            self._x_factors = np.empty((0,), dtype=np.float32)
            self._y_factors = np.empty((0,), dtype=np.float32)
            self._projection_cache_valid = True
            self._cache_width = width
            self._cache_height = height
            return

        u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing="xy")
        inv_fx = np.float32(1.0 / self.fx)
        inv_fy = np.float32(1.0 / self.fy)
        cx = np.float32(self.cx)
        cy = np.float32(self.cy)

        self._x_factors = ((u_grid - cx) * inv_fx).ravel()
        self._y_factors = ((v_grid - cy) * inv_fy).ravel()
        self._projection_cache_valid = True
        self._cache_width = width
        self._cache_height = height

    def _build_pointcloud_msg(
        self,
        depth_msg: Image,
        points: np.ndarray,
    ) -> PointCloud2:
        """Build PointCloud2 message from C-contiguous float32 XYZ array."""
        cloud = PointCloud2()
        cloud.header.stamp = depth_msg.header.stamp
        cloud.header.frame_id = self.output_frame
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = self._point_fields
        cloud.is_bigendian = False
        cloud.is_dense = False
        cloud.point_step = self._point_step_bytes
        cloud.row_step = cloud.point_step * cloud.width

        cloud_data = array("B")
        cloud_data.frombytes(memoryview(points).cast("B"))
        cloud.data = cloud_data
        return cloud

    def _depth_to_pointcloud(self, depth_msg: Image) -> Optional[PointCloud2]:
        """Convert one depth image to PointCloud2."""
        depth_image = self._decode_depth_image(depth_msg)
        if depth_image is None:
            return None

        height, width = depth_image.shape
        self._ensure_projection_cache(width, height)
        if self._x_factors.size == 0:
            return None

        depth_sampled = depth_image[:: self.stride, :: self.stride]
        depth_flat = depth_sampled.ravel()

        if depth_flat.shape[0] != self._x_factors.shape[0]:
            # Safety net in case incoming dimensions changed between callbacks.
            self._projection_cache_valid = False
            self._ensure_projection_cache(width, height)
            if depth_flat.shape[0] != self._x_factors.shape[0]:
                self.get_logger().error(
                    f"Projection cache mismatch: depth={depth_flat.shape[0]} "
                    f"vs cache={self._x_factors.shape[0]}",
                    throttle_duration_sec=2.0,
                )
                return None

        valid_mask = np.isfinite(depth_flat)
        valid_mask &= depth_flat >= self.min_valid_depth
        valid_mask &= depth_flat <= self.max_depth
        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count == 0:
            self.get_logger().warning(
                "No valid depth points found in image.",
                throttle_duration_sec=5.0,
            )
            return None

        depth_valid = depth_flat[valid_mask].astype(np.float32, copy=False)
        points = np.empty((valid_count, 3), dtype=np.float32)
        np.multiply(self._x_factors[valid_mask], depth_valid, out=points[:, 0])
        np.multiply(self._y_factors[valid_mask], depth_valid, out=points[:, 1])
        points[:, 2] = depth_valid

        pointcloud_msg = self._build_pointcloud_msg(depth_msg, points)
        if (
            self.get_logger().get_effective_level()
            <= rclpy.logging.LoggingSeverity.DEBUG
        ):
            self.get_logger().debug(
                f"Generated cloud with {valid_count} points "
                f"(image={width}x{height}, stride={self.stride})"
            )
        return pointcloud_msg


def main(args=None):
    """Run the depth-to-pointcloud node."""
    rclpy.init(args=args)
    node = None

    def _shutdown_from_signal(signum, _frame):
        if node is not None:
            node.save_profiling()
        if rclpy.ok():
            rclpy.try_shutdown()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))

    try:
        node = DepthToPointCloudNode()
        signal.signal(signal.SIGINT, _shutdown_from_signal)
        signal.signal(signal.SIGTERM, _shutdown_from_signal)
        rclpy.get_default_context().on_shutdown(node.save_profiling)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        rclpy.logging.get_logger("depth_to_pointcloud_node").error(
            f"Unhandled exception in node: {e}"
        )
    finally:
        if node is not None:
            node.save_profiling()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
