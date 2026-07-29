"""ROS subscriptions, synchronization, message conversion, and publication."""

from __future__ import annotations

import collections
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from semantic_perception.debug_image import render_debug_image
from semantic_perception.projection import (
    ProjectionDiagnostics,
    depth_statistics,
    transform_to_matrix,
)
from semantic_perception.worker import Frame, FrameResult, WorkerPool
from semantic_perception_msgs.msg import ObjectProposal3D, ObjectProposal3DArray


@dataclass
class _PendingFrame:
    """One synchronized frame waiting for its own transform to arrive."""

    sequence: int
    stamp: object
    camera_frame: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: tuple
    received_monotonic: float
    deadline: object


class SemanticPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_perception")
        self._bridge = CvBridge()
        self._sequence = itertools.count()
        self._declare_parameters()
        config = self._configuration()
        self._validate_configuration(config)

        self._publisher = self.create_publisher(
            ObjectProposal3DArray, str(config["proposal_topic"]), 10
        )
        self._debug_mask_alpha = float(config["debug_mask_alpha"])
        self._debug_publisher = None
        if bool(config["publish_debug_image"]):
            self._debug_publisher = self.create_publisher(
                Image, str(config["debug_image_topic"]), 1
            )
        self._shutdown_on_worker_failure = bool(
            config["shutdown_when_all_workers_failed"]
        )
        self._shutdown_initiated = False
        self._target_frame = str(config["target_frame"]).strip()
        self._tf_timeout = Duration(seconds=max(0.0, float(config["tf_timeout_sec"])))
        self._publish_projection_diagnostics = bool(
            config["publish_projection_diagnostics"]
        )
        self._max_depth_m = float(config["max_depth_m"])
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_rejections = 0
        self._pending_frames: collections.deque = collections.deque()
        self._pending_frame_limit = max(1, int(config["frame_queue_size"]))
        self._workers = WorkerPool(
            config,
            lambda message: self.get_logger().warning(message),
            lambda message: self.get_logger().info(message),
            self._log_error,
        )

        self._rgb_sub = Subscriber(
            self, Image, str(config["rgb_topic"]), qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = Subscriber(
            self, Image, str(config["depth_topic"]), qos_profile=qos_profile_sensor_data
        )
        self._info_sub = Subscriber(
            self,
            CameraInfo,
            str(config["camera_info_topic"]),
            qos_profile=qos_profile_sensor_data,
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(config["sync_queue_size"]),
            slop=float(config["sync_slop_seconds"]),
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._synchronized_callback)
        self._result_timer = self.create_timer(0.01, self._publish_ready_results)
        self._stats_timer = None
        stats_interval = float(config["stats_report_interval_sec"])
        if stats_interval > 0.0:
            self._stats_timer = self.create_timer(stats_interval, self._report_statistics)
        self.get_logger().info(
            f"Listening for synchronized RGB-D frames on {config['rgb_topic']} and "
            f"{config['depth_topic']}"
        )
        if self._debug_publisher is not None:
            self.get_logger().info(
                f"Publishing inference debug images on {config['debug_image_topic']}"
            )

    def _declare_parameters(self) -> None:
        defaults = {
            "rgb_topic": "/camera/color/image_raw",
            "depth_topic": "/camera/depth/image_raw",
            "camera_info_topic": "/camera/color/camera_info",
            "proposal_topic": "/semantic_perception/object_proposals",
            "target_frame": "odom",
            "tf_timeout_sec": 0.2,
            "publish_projection_diagnostics": False,
            "publish_debug_image": False,
            "debug_image_topic": "/semantic_perception/debug_image",
            "debug_mask_alpha": 0.45,
            "class_labels_path": "models/labels/HM3D_CountsOfObjectTypes.csv",
            "class_embedding_cache_path": "models/hm3d_openclip_embedding_cache.bin",
            "openclip_model": "ViT-H-14",
            "openclip_checkpoint_path": "models/laion2b_s32b_b79k.bin",
            "text_embedding_batch_size": 64,
            "groundingdino_config_path": "models/groundingdino/GroundingDINO_SwinT_OGC.py",
            "groundingdino_model": "models/groundingdino/groundingdino_swint_ogc.pth",
            "sam_model": "models/mobilesam/mobile_sam.pt",
            "sam_model_type": "vit_t",
            "detector_vocabulary_size": 64,
            "excluded_class_labels": "",
            "detector_merge_iou_threshold": 0.7,
            "device": "cuda",
            "devices": ["cuda:0", "cuda:1"],
            "num_worker_threads": 2,
            "frame_queue_size": 4,
            "drop_report_every": 30,
            "result_queue_size": 8,
            "drop_stale_results": True,
            "shutdown_when_all_workers_failed": True,
            "gpu_memory_budget_gb": 0.0,
            "stats_report_interval_sec": 10.0,
            "detection_threshold": 0.35,
            "text_threshold": 0.25,
            "sync_queue_size": 10,
            "sync_slop_seconds": 0.1,
            "min_valid_depth_points": 20,
            "max_depth_m": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _configuration(self) -> dict:
        names = [parameter.name for parameter in self._parameters.values()]
        config = {name: self.get_parameter(name).value for name in names}
        for key in (
            "class_labels_path",
            "class_embedding_cache_path",
            "openclip_checkpoint_path",
            "groundingdino_config_path",
            "groundingdino_model",
            "sam_model",
        ):
            config[key] = self._resolve_path(
                str(config[key]), key == "class_embedding_cache_path"
            )
        return config

    @staticmethod
    def _resolve_path(value: str, output: bool = False) -> str:
        if not value:
            return ""
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        package_share = Path(get_package_share_directory("semantic_perception"))
        roots = [Path.cwd(), *Path.cwd().parents, package_share, *package_share.parents]
        candidates = [root / path for root in roots]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        if output:
            for candidate in candidates:
                if candidate.parent.is_dir():
                    return str(candidate.resolve())
        return str(package_share / path)

    @staticmethod
    def _validate_configuration(config: dict) -> None:
        for name in ("detection_threshold", "text_threshold"):
            value = float(config[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not str(config["target_frame"]).strip():
            raise ValueError("target_frame must name the graph coordinate frame")
        if int(config["detector_vocabulary_size"]) < 0:
            raise ValueError("detector_vocabulary_size must be 0 (all) or positive")
        if not 0.0 <= float(config["detector_merge_iou_threshold"]) <= 1.0:
            raise ValueError("detector_merge_iou_threshold must be between 0 and 1")
        if float(config["tf_timeout_sec"]) < 0.0:
            raise ValueError("tf_timeout_sec must be non-negative")
        if int(config["text_embedding_batch_size"]) <= 0:
            raise ValueError("text_embedding_batch_size must be positive")
        if not 0.0 <= float(config["debug_mask_alpha"]) <= 1.0:
            raise ValueError("debug_mask_alpha must be between 0 and 1")
        if (
            int(config["sync_queue_size"]) <= 0
            or float(config["sync_slop_seconds"]) < 0.0
        ):
            raise ValueError(
                "Synchronization queue size must be positive and slop non-negative"
            )
        if int(config["drop_report_every"]) <= 0:
            raise ValueError("drop_report_every must be positive")
        if float(config["gpu_memory_budget_gb"]) < 0.0:
            raise ValueError("gpu_memory_budget_gb must be 0 (unlimited) or positive")
        if float(config["stats_report_interval_sec"]) < 0.0:
            raise ValueError("stats_report_interval_sec must be 0 (disabled) or positive")

    def _synchronized_callback(
        self, rgb_message: Image, depth_message: Image, camera_info: CameraInfo
    ) -> None:
        """Convert ROS data and enqueue only; no model inference occurs here."""
        try:
            rgb = np.asarray(
                self._bridge.imgmsg_to_cv2(rgb_message, desired_encoding="rgb8"),
                dtype=np.uint8,
            )
            raw_depth = np.asarray(
                self._bridge.imgmsg_to_cv2(
                    depth_message, desired_encoding="passthrough"
                )
            )
            depth = self._depth_in_metres(raw_depth, depth_message.encoding)
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.get_logger().warning(
                f"Discarded frame with invalid image encoding: {exc}"
            )
            return
        fx, fy, cx, cy = (
            camera_info.k[0],
            camera_info.k[4],
            camera_info.k[2],
            camera_info.k[5],
        )
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warning(
                "CameraInfo is uncalibrated; proposals will have invalid 3D data"
            )
        if rgb.shape[:2] != depth.shape:
            self.get_logger().warning(
                "RGB and depth resolutions differ; 3D geometry will be invalid "
                "until aligned depth is used"
            )
            return
        # TF for an image's own timestamp usually arrives a few milliseconds
        # after the image itself, so the frame waits in a short bounded queue
        # until its transform exists rather than being resolved against a newer
        # robot pose or dropped outright.
        self._pending_frames.append(
            _PendingFrame(
                sequence=next(self._sequence),
                stamp=rgb_message.header.stamp,
                camera_frame=str(rgb_message.header.frame_id or ""),
                rgb=np.ascontiguousarray(rgb),
                depth_m=np.ascontiguousarray(depth),
                intrinsics=(float(fx), float(fy), float(cx), float(cy)),
                received_monotonic=time.monotonic(),
                deadline=self.get_clock().now() + self._tf_timeout,
            )
        )
        while len(self._pending_frames) > self._pending_frame_limit:
            dropped = self._pending_frames.popleft()
            self._tf_rejections += 1
            self.get_logger().warning(
                f"Dropped frame {dropped.sequence}: transform queue overflow",
                throttle_duration_sec=5.0,
            )
        self._drain_pending_frames()

    def _drain_pending_frames(self) -> None:
        """Submit frames whose transform is now available; drop expired ones."""
        if not self._pending_frames:
            return
        now = self.get_clock().now()
        remaining = collections.deque()
        while self._pending_frames:
            pending = self._pending_frames.popleft()
            camera_to_world, transform_stamp = self._lookup_camera_to_world(
                pending.camera_frame, pending.stamp
            )
            if camera_to_world is not None:
                self._workers.submit(
                    Frame(
                        sequence=pending.sequence,
                        stamp_sec=pending.stamp.sec,
                        stamp_nanosec=pending.stamp.nanosec,
                        frame_id=self._target_frame,
                        rgb=pending.rgb,
                        depth_m=pending.depth_m,
                        intrinsics=pending.intrinsics,
                        camera_to_world=camera_to_world,
                        received_monotonic=pending.received_monotonic,
                        source_frame_id=pending.camera_frame,
                        transform_stamp_sec=transform_stamp,
                    )
                )
                continue
            if now >= pending.deadline:
                self._tf_rejections += 1
                self.get_logger().warning(
                    f"Dropped frame {pending.sequence}: no "
                    f"{pending.camera_frame}->{self._target_frame} transform at the "
                    "image timestamp within tf_timeout_sec",
                    throttle_duration_sec=5.0,
                )
                continue
            remaining.append(pending)
        self._pending_frames = remaining

    def _lookup_camera_to_world(self, camera_frame: str, stamp):
        """Resolve the camera-to-target transform at the image's own timestamp.

        Returns ``(matrix, transform_stamp_seconds)``. A missing, stale, or
        invalid transform returns ``(None, 0.0)`` and the frame is dropped: no
        proposal is ever published in camera-local coordinates.
        """
        if not camera_frame:
            self.get_logger().warning(
                "Dropped frame: RGB header has no frame_id, so it cannot be "
                f"transformed into {self._target_frame}",
                throttle_duration_sec=5.0,
            )
            self._tf_rejections += 1
            return None, 0.0
        if camera_frame == self._target_frame:
            return np.eye(4, dtype=np.float64), Time.from_msg(stamp).nanoseconds / 1e9
        try:
            transform_stamped = self._tf_buffer.lookup_transform(
                self._target_frame, camera_frame, Time.from_msg(stamp)
            )
        except Exception:
            # Not available yet, or no longer available; the caller retries until
            # its deadline and then drops the frame.
            return None, 0.0
        matrix = transform_to_matrix(transform_stamped)
        if matrix is None:
            self._tf_rejections += 1
            self.get_logger().warning(
                f"Dropped frame: {camera_frame}->{self._target_frame} transform is "
                "not a valid rigid transform",
                throttle_duration_sec=5.0,
            )
            return None, 0.0
        return matrix, Time.from_msg(transform_stamped.header.stamp).nanoseconds / 1e9

    @staticmethod
    def _depth_in_metres(depth: np.ndarray, encoding: str) -> np.ndarray:
        if depth.ndim != 2:
            raise ValueError(
                f"Depth image must be single-channel, got shape {depth.shape}"
            )
        normalized = encoding.upper()
        if normalized in ("16UC1", "MONO16"):
            return depth.astype(np.float32) * 0.001
        if normalized == "32FC1":
            return depth.astype(np.float32, copy=False)
        raise ValueError(
            f"Unsupported depth encoding '{encoding}'; expected 16UC1 or 32FC1"
        )

    def _publish_ready_results(self) -> None:
        self._drain_pending_frames()
        while True:
            result = self._workers.get_result()
            if result is None:
                break
            self._publish_result(result)
        self._check_worker_health()

    def _check_worker_health(self) -> None:
        """Shut down once every inference worker thread has stopped running."""
        if (
            not self._shutdown_on_worker_failure
            or self._shutdown_initiated
            or not self._workers.all_workers_failed()
        ):
            return
        self._shutdown_initiated = True
        self.get_logger().error(
            "All inference workers have stopped; shutting the node down so a "
            "supervisor can restart it."
        )
        rclpy.try_shutdown()

    def _log_error(self, message: str) -> None:
        self.get_logger().error(message)

    def _report_statistics(self) -> None:
        info, warning = self._workers.stats_report()
        if info:
            self.get_logger().info(info)
        if warning:
            self.get_logger().warning(warning)

    def _publish_result(self, result: FrameResult) -> None:
        message = ObjectProposal3DArray()
        message.header.stamp.sec = result.frame.stamp_sec
        message.header.stamp.nanosec = result.frame.stamp_nanosec
        # Geometry was already projected into the target frame at the image
        # timestamp, so the published frame is the graph frame.
        message.header.frame_id = result.frame.frame_id
        diagnostics = None
        if self._publish_projection_diagnostics:
            diagnostics = ProjectionDiagnostics(
                sequence=result.frame.sequence,
                source_frame=result.frame.source_frame_id,
                target_frame=result.frame.frame_id,
                frame_stamp_sec=result.frame.stamp_sec
                + result.frame.stamp_nanosec * 1e-9,
                transform_stamp_sec=result.frame.transform_stamp_sec,
                transform=result.frame.camera_to_world,
            )
        for identifier, proposal in enumerate(result.proposals):
            if not self._proposal_is_publishable(proposal, identifier):
                continue
            item = ObjectProposal3D()
            item.id = identifier
            item.class_name = proposal.detection.class_name
            item.class_id = int(proposal.detection.class_index or 0)
            x1, y1, x2, y2 = proposal.detection.box_xyxy.tolist()
            item.bbox_2d.center.position.x = float((x1 + x2) * 0.5)
            item.bbox_2d.center.position.y = float((y1 + y2) * 0.5)
            item.bbox_2d.center.theta = 0.0
            item.bbox_2d.size_x = float(max(0.0, x2 - x1))
            item.bbox_2d.size_y = float(max(0.0, y2 - y1))
            item.mask_embedding = proposal.embeddings.mask_embedding.tolist()
            item.bbox_embedding = proposal.embeddings.bbox_embedding.tolist()
            item.label_embedding = proposal.embeddings.label_embedding.tolist()
            item.fused_embedding = proposal.embeddings.fused_embedding.tolist()
            item.detection_confidence = proposal.detection.confidence
            item.detector_source = proposal.detection.source
            item.valid_3d = proposal.geometry.valid
            if proposal.geometry.valid:
                centroid = proposal.geometry.centroid
                minimum, maximum = proposal.geometry.minimum, proposal.geometry.maximum
                item.centroid_3d.x, item.centroid_3d.y, item.centroid_3d.z = map(
                    float, centroid
                )
                center = (minimum + maximum) * 0.5
                size = maximum - minimum
                item.bbox_3d.center.position.x = float(center[0])
                item.bbox_3d.center.position.y = float(center[1])
                item.bbox_3d.center.position.z = float(center[2])
                item.bbox_3d.center.orientation.w = 1.0
                item.bbox_3d.size.x = float(size[0])
                item.bbox_3d.size.y = float(size[1])
                item.bbox_3d.size.z = float(size[2])
            if diagnostics is not None:
                diagnostics.add_detection(
                    identifier,
                    proposal.detection.class_name,
                    self._camera_frame_centroid(proposal, result.frame),
                    (
                        tuple(float(value) for value in proposal.geometry.centroid)
                        if proposal.geometry.valid
                        else None
                    ),
                    proposal.mask_pixels,
                    depth_statistics(
                        result.frame.depth_m,
                        proposal.mask,
                        proposal.detection.box_xyxy,
                        self._max_depth_m,
                    ),
                )
            message.proposals.append(item)
        if diagnostics is not None:
            self.get_logger().debug(diagnostics.render())
        self._publisher.publish(message)
        if self._debug_publisher is not None:
            self._publish_debug_result(result, message)

    def _proposal_is_publishable(self, proposal, identifier: int) -> bool:
        """Reject proposals lacking a class, a full embedding, or valid geometry."""
        if not proposal.detection.class_name or proposal.detection.class_index is None:
            self.get_logger().debug(
                f"Dropped proposal {identifier}: unresolved class label",
                throttle_duration_sec=5.0,
            )
            return False
        if proposal.embeddings.fused_embedding.size == 0:
            self.get_logger().debug(
                f"Dropped proposal {identifier} ({proposal.detection.class_name}): "
                "incomplete fused embedding",
                throttle_duration_sec=5.0,
            )
            return False
        if not proposal.geometry.valid:
            self.get_logger().debug(
                f"Dropped proposal {identifier} ({proposal.detection.class_name}): "
                "no valid 3D geometry",
                throttle_duration_sec=5.0,
            )
            return False
        return True

    @staticmethod
    def _camera_frame_centroid(proposal, frame) -> tuple[float, float, float] | None:
        """Recover the pre-transform centroid for diagnostics only."""
        if not proposal.geometry.valid or frame.camera_to_world is None:
            return None
        try:
            inverse = np.linalg.inv(np.asarray(frame.camera_to_world, dtype=np.float64))
        except np.linalg.LinAlgError:
            return None
        point = np.append(np.asarray(proposal.geometry.centroid, dtype=np.float64), 1.0)
        camera_point = inverse @ point
        if not np.isfinite(camera_point).all() or abs(camera_point[3]) <= 1e-12:
            return None
        return tuple(float(value) for value in camera_point[:3] / camera_point[3])

    def _publish_debug_result(
        self, result: FrameResult, proposal_message: ObjectProposal3DArray
    ) -> None:
        try:
            rendered = render_debug_image(
                result.frame.rgb,
                result.proposals,
                self._workers.best_class,
                self._debug_mask_alpha,
            )
            debug_message = self._bridge.cv2_to_imgmsg(rendered, encoding="bgr8")
            debug_message.header = proposal_message.header
            self._debug_publisher.publish(debug_message)
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to render debug image for frame {result.frame.sequence}: {exc}"
            )

    def destroy_node(self) -> bool:
        if hasattr(self, "_workers"):
            self._workers.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SemanticPerceptionNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
