"""ROS subscriptions, synchronization, message conversion, and publication."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from semantic_perception_msgs.msg import ObjectProposal3D, ObjectProposal3DArray

from semantic_perception.debug_image import render_debug_image
from semantic_perception.inference.crop_embeddings import validate_fusion_weights
from semantic_perception.worker import Frame, FrameResult, WorkerPool


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
        self._workers = WorkerPool(
            config,
            lambda message: self.get_logger().warning(message),
            lambda message: self.get_logger().info(message),
        )

        self._rgb_sub = Subscriber(
            self, Image, str(config["rgb_topic"]), qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = Subscriber(
            self, Image, str(config["depth_topic"]), qos_profile=qos_profile_sensor_data
        )
        self._info_sub = Subscriber(
            self, CameraInfo, str(config["camera_info_topic"]), qos_profile=qos_profile_sensor_data
        )
        self._synchronizer = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub, self._info_sub],
            queue_size=int(config["sync_queue_size"]),
            slop=float(config["sync_slop_seconds"]),
            allow_headerless=False,
        )
        self._synchronizer.registerCallback(self._synchronized_callback)
        self._result_timer = self.create_timer(0.01, self._publish_ready_results)
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
            "publish_debug_image": False,
            "debug_image_topic": "/semantic_perception/debug_image",
            "debug_mask_alpha": 0.45,
            "prompt_csv_path": "models/HM3D_CountsOfObjectTypes.csv",
            "class_embedding_cache_path": "models/hm3d_openclip_embedding_cache.bin",
            "openclip_model": "ViT-H-14",
            "openclip_checkpoint_path": "models/laion2b_s32b_b79k.bin",
            "text_embedding_batch_size": 64,
            "groundingdino_config_path": "models/GroundingDINO_SwinT_OGC.py",
            "groundingdino_model": "models/groundingdino_swint_ogc.pth",
            "sam_model": "models/mobile_sam.pt",
            "sam_model_type": "vit_t",
            "groundingdino_prompt": "object",
            "device": "cuda",
            "devices": ["cuda:0", "cuda:1"],
            "num_worker_threads": 2,
            "frame_queue_size": 4,
            "result_queue_size": 8,
            "detection_threshold": 0.35,
            "text_threshold": 0.25,
            "bbox_embedding_weight": 0.5,
            "masked_embedding_weight": 0.5,
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
            "prompt_csv_path",
            "class_embedding_cache_path",
            "openclip_checkpoint_path",
            "groundingdino_config_path",
            "groundingdino_model",
            "sam_model",
        ):
            config[key] = self._resolve_path(str(config[key]), key == "class_embedding_cache_path")
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
        validate_fusion_weights(
            float(config["bbox_embedding_weight"]),
            float(config["masked_embedding_weight"]),
        )
        for name in ("detection_threshold", "text_threshold"):
            value = float(config[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not str(config["groundingdino_prompt"]).strip():
            raise ValueError("groundingdino_prompt must not be empty")
        if int(config["text_embedding_batch_size"]) <= 0:
            raise ValueError("text_embedding_batch_size must be positive")
        if not 0.0 <= float(config["debug_mask_alpha"]) <= 1.0:
            raise ValueError("debug_mask_alpha must be between 0 and 1")
        if int(config["sync_queue_size"]) <= 0 or float(config["sync_slop_seconds"]) < 0.0:
            raise ValueError("Synchronization queue size must be positive and slop non-negative")

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
                self._bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
            )
            depth = self._depth_in_metres(raw_depth, depth_message.encoding)
        except (CvBridgeError, ValueError, TypeError) as exc:
            self.get_logger().warning(f"Discarded frame with invalid image encoding: {exc}")
            return
        fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warning(
                "CameraInfo is uncalibrated; proposals will have invalid 3D data"
            )
        if rgb.shape[:2] != depth.shape:
            self.get_logger().warning(
                "RGB and depth resolutions differ; 3D geometry will be invalid "
                "until aligned depth is used"
            )
        frame = Frame(
            sequence=next(self._sequence),
            stamp_sec=rgb_message.header.stamp.sec,
            stamp_nanosec=rgb_message.header.stamp.nanosec,
            frame_id=rgb_message.header.frame_id,
            rgb=np.ascontiguousarray(rgb),
            depth_m=np.ascontiguousarray(depth),
            intrinsics=(float(fx), float(fy), float(cx), float(cy)),
        )
        self._workers.submit(frame)

    @staticmethod
    def _depth_in_metres(depth: np.ndarray, encoding: str) -> np.ndarray:
        if depth.ndim != 2:
            raise ValueError(f"Depth image must be single-channel, got shape {depth.shape}")
        normalized = encoding.upper()
        if normalized in ("16UC1", "MONO16"):
            return depth.astype(np.float32) * 0.001
        if normalized == "32FC1":
            return depth.astype(np.float32, copy=False)
        raise ValueError(f"Unsupported depth encoding '{encoding}'; expected 16UC1 or 32FC1")

    def _publish_ready_results(self) -> None:
        while True:
            result = self._workers.get_result()
            if result is None:
                return
            self._publish_result(result)

    def _publish_result(self, result: FrameResult) -> None:
        message = ObjectProposal3DArray()
        message.header.stamp.sec = result.frame.stamp_sec
        message.header.stamp.nanosec = result.frame.stamp_nanosec
        message.header.frame_id = result.frame.frame_id
        if result.error:
            self.get_logger().warning(
                f"Publishing empty proposal array for failed frame "
                f"{result.frame.sequence}: {result.error}"
            )
        for identifier, proposal in enumerate(result.proposals):
            item = ObjectProposal3D()
            item.id = identifier
            # Deliberately unset: semantic classification belongs to a later module.
            item.class_name = ""
            item.class_id = 0
            x1, y1, x2, y2 = proposal.detection.box_xyxy.tolist()
            item.bbox_2d.center.position.x = float((x1 + x2) * 0.5)
            item.bbox_2d.center.position.y = float((y1 + y2) * 0.5)
            item.bbox_2d.center.theta = 0.0
            item.bbox_2d.size_x = float(max(0.0, x2 - x1))
            item.bbox_2d.size_y = float(max(0.0, y2 - y1))
            if proposal.mask is not None:
                item.mask = self._bridge.cv2_to_imgmsg(
                    proposal.mask.astype(np.uint8) * 255, encoding="mono8"
                )
                item.mask.header = message.header
            item.mask_embedding = proposal.embeddings.mask_embedding.tolist()
            item.bbox_embedding = proposal.embeddings.bbox_embedding.tolist()
            item.fused_embedding = proposal.embeddings.fused_embedding.tolist()
            item.detection_confidence = proposal.detection.confidence
            item.detector_source = proposal.detection.source
            item.valid_3d = proposal.geometry.valid
            if proposal.geometry.valid:
                centroid = proposal.geometry.centroid
                minimum, maximum = proposal.geometry.minimum, proposal.geometry.maximum
                item.centroid_3d.x, item.centroid_3d.y, item.centroid_3d.z = map(float, centroid)
                center = (minimum + maximum) * 0.5
                size = maximum - minimum
                item.bbox_3d.center.position.x = float(center[0])
                item.bbox_3d.center.position.y = float(center[1])
                item.bbox_3d.center.position.z = float(center[2])
                item.bbox_3d.center.orientation.w = 1.0
                item.bbox_3d.size.x = float(size[0])
                item.bbox_3d.size.y = float(size[1])
                item.bbox_3d.size.z = float(size[2])
            # Similarity and entropy retain their zero defaults by design.
            message.proposals.append(item)
        self._publisher.publish(message)
        if self._debug_publisher is not None:
            self._publish_debug_result(result, message)

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


from rclpy.executors import ExternalShutdownException  # noqa: E402
