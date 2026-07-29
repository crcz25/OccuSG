"""
Object Manager - Manage semantic-perception object proposals and observation links.

This manager handles the creation and tracking of OBJECT nodes in the OBJECT layer,
representing detected objects in the environment. Observation edges are created
during occupancy-grid-based line-of-sight updates.

Features:
- Spatial and embedding-based object association
- Online running-mean representation embeddings (``object_embedding``)
- Room-scoped association through the existing ROOM_CONTAINS ownership
- OBSERVATION_ANCHOR edge creation to poses via LoS updates
- Thread-safe scene graph updates
- Line-of-sight validation using occupancy-grid raycasting

Usage:
    obj_mgr = ObjectNodeManager(sg_interface, logger, room_manager=room_mgr, ...)
    obj_mgr.process_detections_update(proposals_msg, tf_buffer, room_id=room_id)
"""

import math
import time
import traceback
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose

from scene_graph_core.algorithms.semantic import (
    cosine_similarity,
    normalize_embedding,
    running_mean_embedding,
)
from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import (
    BaseNode,
    Edge,
    EdgeType,
    NodeType,
    ObjectNode,
)
from scene_graph_core.services import GraphPatch
from semantic_perception_msgs.msg import ObjectProposal3DArray

# Sentinel distinguishing "no transform needed" from "transform lookup failed".
_TRANSFORM_FAILED = object()


class ObjectNodeManager:
    """
    Manages OBJECT nodes representing detected objects in the OBJECT layer.

    This class processes ``semantic_perception`` object proposals and associates
    each one with an existing object node when it is both spatially close and
    semantically similar, otherwise creating a new node. It also performs
    occupancy-grid line-of-sight checks and writes observation edges to visible
    objects.
    """

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        # Object association parameters
        spatial_association_threshold: float = 0.75,  # meters
        semantic_similarity_threshold: float = 0.7,  # cosine similarity
        position_update_policy: str = "latest",  # "latest" or "running_mean"
        room_manager=None,
        # Line-of-sight parameters
        max_los_range: float = 10.0,  # meters — maximum sensing range for LoS
        los_fov_deg: float = 360.0,  # degrees — field of view (360 = omnidirectional)
        los_unknown_is_occupied: bool = False,  # treat unknown grid cells as obstacles
        # Debug logging
        enable_debug_logging: bool = True,
        debug_log_interval: int = 10,  # Log every N detections
    ):
        """
        Initialize the ObjectNodeManager.

        Args:
            sg_interface: Shared scene graph interface for thread-safe access
            logger: ROS logger for debug/info messages
            spatial_association_threshold: Maximum distance (m) for candidate objects
            semantic_similarity_threshold: Minimum cosine similarity for association
            position_update_policy: "latest" keeps the newest observed pose,
                "running_mean" averages the associated observations
            room_manager: Optional RoomManager used to keep association inside one
                room and to own the ROOM_CONTAINS attachment of new objects
            max_los_range: Maximum sensing range (m) for line-of-sight checks
            los_fov_deg: Field-of-view angle (degrees) centred on robot heading;
                         360 means full omnidirectional visibility
            los_unknown_is_occupied: If True, unknown occupancy grid cells (-1)
                                     are treated as obstacles when raycasting
            enable_debug_logging: Enable detailed debug logging
            debug_log_interval: Log detailed stats every N updates
        """
        self.sg = sg_interface
        self.logger = logger

        # Configuration
        self.spatial_association_threshold = max(0.0, float(spatial_association_threshold))
        self.semantic_similarity_threshold = float(semantic_similarity_threshold)
        self.position_update_policy = str(position_update_policy).strip().lower()
        self.room_manager = room_manager
        self.max_los_range = max_los_range
        self.los_fov_deg = los_fov_deg
        self.los_unknown_is_occupied = los_unknown_is_occupied

        # Debug logging
        self.enable_debug_logging = enable_debug_logging
        self.debug_log_interval = debug_log_interval
        self.update_counter = 0

        # Occupancy grid for Bresenham raycasting
        self.occupancy_grid = None
        self.grid_resolution: Optional[float] = None
        self.grid_origin: Optional[Tuple[float, float]] = None
        self.grid_width: Optional[int] = None
        self.grid_height: Optional[int] = None

        # Statistics (for monitoring)
        self.stats = {
            "total_detection_messages": 0,
            "total_detections_received": 0,
            "total_detections_accepted": 0,
            "total_detections_rejected": 0,
            "rejected_by_reason": {},
            "total_objects_created": 0,
            "total_objects_updated": 0,
            "total_objects_associated": 0,
            "total_observation_edges_created": 0,
            "last_detection_stamp_sec": None,
            "last_object_create_stamp_sec": None,
            "last_object_update_stamp_sec": None,
        }
        self._rejected_by_reason = Counter()

        self.logger.debug("ObjectNodeManager initialized:")
        self.logger.debug(
            f"  - spatial_association_threshold: {self.spatial_association_threshold}m"
        )
        self.logger.debug(
            f"  - semantic_similarity_threshold: {self.semantic_similarity_threshold}"
        )
        self.logger.debug(f"  - position_update_policy: {self.position_update_policy}")
        self.logger.debug(f"  - max_los_range: {self.max_los_range}m")
        self.logger.debug(f"  - los_fov_deg: {self.los_fov_deg}°")
        self.logger.debug(
            f"  - los_unknown_is_occupied: {self.los_unknown_is_occupied}"
        )
        self.logger.debug(f"  - debug_logging: {self.enable_debug_logging}")

    def _initialize_pose_attributes(self, pose_node: BaseNode) -> None:
        """Initialize tracking attributes for a pose node if not present.

        Args:
            pose_node: Pose node to initialize attributes for
        """
        if pose_node.attributes is None:
            pose_node.attributes = {}

        # Initialize object_in_los tracking
        if "object_in_los" not in pose_node.attributes:
            pose_node.attributes["object_in_los"] = []

    def _log_update_stats(self, objects_processed: int, new_objects: int):
        """Log detailed update statistics if debug logging is enabled."""
        if not self.enable_debug_logging:
            return

        self.update_counter += 1
        if self.update_counter % self.debug_log_interval != 0:
            return

        self.logger.debug(f"=== ObjectNodeManager Update #{self.update_counter} ===")
        self.logger.debug(
            f"  Total objects created: {self.stats['total_objects_created']}"
        )
        self.logger.debug(
            f"  Total objects updated: {self.stats['total_objects_updated']}"
        )
        self.logger.debug(
            f"  Total objects associated: {self.stats['total_objects_associated']}"
        )
        self.logger.debug(
            f"  Total observation edges: {self.stats['total_observation_edges_created']}"
        )
        self.logger.debug(
            f"  Last batch: {objects_processed} processed, {new_objects} new"
        )

    def update_occupancy_grid(self, grid_msg: OccupancyGrid):
        """
        Update stored occupancy grid for raycast-based LoS checks.

        Args:
            grid_msg: OccupancyGrid message
        """
        # Store grid data as 2D numpy array
        width = grid_msg.info.width
        height = grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.int8).reshape((height, width))

        self.occupancy_grid = data
        self.grid_resolution = grid_msg.info.resolution
        self.grid_origin = (
            grid_msg.info.origin.position.x,
            grid_msg.info.origin.position.y,
        )
        self.grid_width = width
        self.grid_height = height

        self.logger.debug(
            f"Updated occupancy grid: {width}x{height}, res={self.grid_resolution:.3f}m"
        )

    def process_detections_update(
        self,
        proposals_msg: ObjectProposal3DArray,
        tf_buffer,
        fixed_frame_id: str = "world",
        transform_stamped=None,
        room_id: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Create or update objects from semantic-perception proposals.

        For each proposal:
        1. Reject malformed geometry, then transform the centroid to the fixed frame
        2. Find nearby OBJECT nodes and score them against ``fused_embedding``
        3. Associate with the best candidate, or create a new object node
        4. Attach the object to the supplied room through ROOM_CONTAINS
        5. Report newly created object IDs for derived nearest-link maintenance

        Args:
            proposals_msg: ObjectProposal3DArray published by semantic_perception
            tf_buffer: TF2 buffer for coordinate transforms
            fixed_frame_id: Target frame for object poses (default: "world")
            transform_stamped: Optional pre-fetched transform from proposal frame
                to fixed frame. Supplying this avoids TF lookups inside SG critical sections.
            room_id: Room that currently owns newly observed objects, when known

        Returns:
            Dictionary with update statistics
        """
        proposals = (
            list(getattr(proposals_msg, "proposals", ()) or ())
            if proposals_msg is not None
            else []
        )
        self.stats["total_detection_messages"] += 1
        if not proposals:
            return self._empty_update_stats()

        detection_count = len(proposals)
        self.stats["total_detections_received"] += detection_count

        header = getattr(proposals_msg, "header", None)
        detection_frame = str(getattr(header, "frame_id", "") or "")
        transform_stamped = self._resolve_transform(
            tf_buffer, detection_frame, fixed_frame_id, transform_stamped
        )
        if transform_stamped is _TRANSFORM_FAILED:
            self._record_detection_rejection("tf_lookup_failed", detection_count)
            self.stats["total_detections_rejected"] += detection_count
            return self._empty_update_stats(
                rejected=detection_count,
                rejected_by_reason={"tf_lookup_failed": detection_count},
            )

        timestamp = self._message_timestamp(header)
        self.stats["last_detection_stamp_sec"] = timestamp

        new_objects = 0
        updated_objects = 0
        accepted_detections = 0
        rejected_by_reason = Counter()
        new_object_ids: List[int] = []
        updated_object_ids: List[int] = []

        for detection_index, proposal in enumerate(proposals):
            obj_node, is_new, reason = self._process_single_detection(
                proposal,
                timestamp,
                transform_stamped,
                detection_index=detection_index,
                room_id=room_id,
            )

            if obj_node is None:
                rejected_by_reason[str(reason or "unknown")] += 1
                continue

            accepted_detections += 1
            if is_new:
                new_objects += 1
                if obj_node.id is not None:
                    new_object_ids.append(int(obj_node.id))
            else:
                updated_objects += 1
                if obj_node.id is not None:
                    updated_object_ids.append(int(obj_node.id))

        rejected_detections = int(sum(rejected_by_reason.values()))
        self.stats["total_detections_accepted"] += accepted_detections
        self.stats["total_detections_rejected"] += rejected_detections
        for reason, count in rejected_by_reason.items():
            self._record_detection_rejection(reason, count)
        self.stats["rejected_by_reason"] = dict(self._rejected_by_reason)

        result_stats = {
            "objects_processed": detection_count,
            "accepted_detections": accepted_detections,
            "rejected_detections": rejected_detections,
            "rejected_by_reason": dict(rejected_by_reason),
            "new_objects": new_objects,
            "updated_objects": updated_objects,
            "new_object_ids": new_object_ids,
            "updated_object_ids": updated_object_ids,
            "object_count": len(self.sg.query.find_nodes_by_type(NodeType.OBJECT)),
        }

        self._log_update_stats(detection_count, new_objects)

        return result_stats

    def _empty_update_stats(
        self,
        *,
        rejected: int = 0,
        rejected_by_reason: Optional[Dict[str, int]] = None,
    ) -> Dict[str, object]:
        """Return the zero-work result shape used when nothing was applied."""
        rejected = max(0, int(rejected))
        return {
            "objects_processed": rejected,
            "accepted_detections": 0,
            "rejected_detections": rejected,
            "rejected_by_reason": dict(rejected_by_reason or {}),
            "new_objects": 0,
            "updated_objects": 0,
            "new_object_ids": [],
            "updated_object_ids": [],
        }

    def _resolve_transform(
        self,
        tf_buffer,
        detection_frame: str,
        fixed_frame_id: str,
        transform_stamped,
    ):
        """Resolve the proposal-frame transform, or the failure sentinel."""
        if (
            transform_stamped is not None
            or not detection_frame
            or detection_frame == fixed_frame_id
        ):
            return transform_stamped

        try:
            return tf_buffer.lookup_transform(
                fixed_frame_id,  # target frame
                detection_frame,  # source frame (camera/sensor frame)
                Time(),  # Get latest available transform
                Duration(seconds=1),  # timeout
            )
        except Exception as exc:
            self.logger.warning(
                "[object_proposals] TF lookup failed "
                f"{detection_frame}->{fixed_frame_id}: {exc}",
                throttle_duration_sec=5.0,
            )
            return _TRANSFORM_FAILED

    @staticmethod
    def _message_timestamp(header) -> float:
        """Return a usable message timestamp, falling back to wall time."""
        stamp = getattr(header, "stamp", None)
        try:
            timestamp = (
                float(getattr(stamp, "sec", 0))
                + float(getattr(stamp, "nanosec", 0)) * 1e-9
            )
        except (TypeError, ValueError, OverflowError):
            timestamp = float("nan")
        if not math.isfinite(timestamp) or timestamp <= 0.0:
            return time.time()
        return timestamp

    def _record_detection_rejection(self, reason: str, count: int = 1) -> None:
        """Track detection rejection reasons for runtime diagnostics."""
        reason = str(reason or "unknown")
        count = max(0, int(count))
        if count <= 0:
            return
        self._rejected_by_reason[reason] += count
        self.stats["rejected_by_reason"] = dict(self._rejected_by_reason)

    def _is_finite_pose(self, pose: Pose) -> bool:
        """Return True if a pose contains finite position and orientation values."""
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        return all(math.isfinite(float(value)) for value in values)

    @staticmethod
    def _is_finite(value) -> bool:
        """Return True when a scalar converts to a finite float."""
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @classmethod
    def _finite_scalar(cls, value) -> Optional[float]:
        """Return a finite float, or ``None`` when the value is unusable."""
        return float(value) if cls._is_finite(value) else None

    @staticmethod
    def _nonnegative_int(value, default: int = 0) -> int:
        """Return a non-negative integer, falling back to ``default``."""
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return max(0, int(default))

    def _process_single_detection(
        self,
        proposal,
        timestamp: float,
        transform_stamped,
        detection_index: int = 0,
        room_id: Optional[int] = None,
    ) -> Tuple[Optional[BaseNode], bool, Optional[str]]:
        """Process one ``semantic_perception_msgs/ObjectProposal3D``.

        Returns:
            Tuple of (object_node, is_new, rejection_reason). ``object_node`` is
            ``None`` when the proposal was rejected.
        """
        detection_confidence = self._finite_scalar(
            getattr(proposal, "detection_confidence", None)
        )
        if detection_confidence is None:
            self.logger.warning(
                "[object_proposals] rejected proposal "
                f"idx={detection_index}: nonfinite_detection_confidence",
                throttle_duration_sec=5.0,
            )
            return None, False, "nonfinite_detection_confidence"

        if not bool(getattr(proposal, "valid_3d", False)):
            self.logger.warning(
                "[object_proposals] rejected proposal "
                f"idx={detection_index}: invalid_3d_geometry",
                throttle_duration_sec=5.0,
            )
            return None, False, "invalid_3d_geometry"

        centroid = getattr(proposal, "centroid_3d", None)
        if centroid is None or not all(
            self._is_finite(getattr(centroid, axis, None)) for axis in ("x", "y", "z")
        ):
            self.logger.warning(
                "[object_proposals] rejected proposal "
                f"idx={detection_index}: nonfinite_centroid",
                throttle_duration_sec=5.0,
            )
            return None, False, "nonfinite_centroid"

        # semantic_perception reports object geometry as a 3D centroid; the graph
        # keeps object orientation identity because proposals are axis-agnostic.
        det_ps = Pose()
        det_ps.position.x = float(centroid.x)
        det_ps.position.y = float(centroid.y)
        det_ps.position.z = float(centroid.z)
        det_ps.orientation.w = 1.0

        if transform_stamped is None:
            world_ps = det_ps
        else:
            try:
                world_ps = do_transform_pose(det_ps, transform_stamped)
            except Exception as exc:
                self.logger.warning(
                    "[object_proposals] rejected proposal "
                    f"idx={detection_index}: transform_failed: {exc}",
                    throttle_duration_sec=5.0,
                )
                return None, False, "transform_failed"

        if world_ps is None or not self._is_finite_pose(world_ps):
            self.logger.warning(
                "[object_proposals] rejected proposal "
                f"idx={detection_index}: invalid_world_pose",
                throttle_duration_sec=5.0,
            )
            return None, False, "invalid_world_pose"

        incoming_embedding = normalize_embedding(
            getattr(proposal, "fused_embedding", None)
        )

        node = ObjectNode()  # ID will be assigned by graph
        node.pose.position.x = world_ps.position.x
        node.pose.position.y = world_ps.position.y
        node.pose.position.z = world_ps.position.z
        node.pose.orientation.x = world_ps.orientation.x
        node.pose.orientation.y = world_ps.orientation.y
        node.pose.orientation.z = world_ps.orientation.z
        node.pose.orientation.w = world_ps.orientation.w
        node.created_at = timestamp
        node.last_seen = timestamp
        node.attributes = self._proposal_attributes(proposal, incoming_embedding)

        try:
            merged_node, is_new, similarity = self._find_or_add(
                node,
                incoming_embedding=incoming_embedding,
                range_m=self.spatial_association_threshold,
                room_id=room_id,
            )
        except Exception as exc:
            self.logger.error(
                "[object_proposals] rejected proposal "
                f"idx={detection_index}: graph_mutation_failed: "
                f"{exc}\n{traceback.format_exc()}"
            )
            return None, False, "graph_mutation_failed"

        attributes = (
            merged_node.attributes if isinstance(merged_node.attributes, dict) else {}
        )
        merged_node.attributes = attributes

        if is_new:
            # ``_proposal_attributes`` already seeded object_embedding and its
            # observation count for the first observation.
            attributes.setdefault("observation_count", 1)
            self.stats["total_objects_created"] += 1
            self.stats["last_object_create_stamp_sec"] = timestamp
            self.logger.debug(
                "[object_proposals] accepted proposal "
                f"idx={detection_index} action=create object_id={merged_node.id} "
                f"world=({merged_node.pose.position.x:.2f}, "
                f"{merged_node.pose.position.y:.2f}, "
                f"{merged_node.pose.position.z:.2f})"
            )
        else:
            # ``observation_count`` counts detections folded into this node;
            # ``embedding_observation_count`` counts only the valid embeddings
            # that contributed to the running mean, so the two can diverge.
            previous_observations = self._nonnegative_int(
                attributes.get("observation_count"), default=0
            )
            updated_embedding, updated_embedding_count = running_mean_embedding(
                attributes.get("object_embedding"),
                self._nonnegative_int(
                    attributes.get("embedding_observation_count"), default=0
                ),
                incoming_embedding,
            )
            attributes["object_embedding"] = (
                updated_embedding.astype(np.float32).tolist()
                if updated_embedding is not None
                else None
            )
            attributes["embedding_observation_count"] = updated_embedding_count
            attributes["observation_count"] = previous_observations + 1
            attributes.update(
                self._proposal_attributes(
                    proposal, incoming_embedding, only_present=True
                )
            )
            if similarity is not None:
                attributes["last_semantic_similarity"] = float(similarity)

            self._update_spatial_representation(
                merged_node, world_ps, previous_observations
            )
            merged_node.last_seen = timestamp
            self.stats["total_objects_updated"] += 1
            self.stats["last_object_update_stamp_sec"] = timestamp
            self.logger.debug(
                "[object_proposals] accepted proposal "
                f"idx={detection_index} action=update object_id={merged_node.id} "
                f"similarity={similarity} "
                f"world=({world_ps.position.x:.2f}, "
                f"{world_ps.position.y:.2f}, "
                f"{world_ps.position.z:.2f})"
            )

        self.sg.update.update_node(merged_node.id, merged_node)

        if room_id is not None and self.room_manager is not None:
            self.room_manager.attach_direct_member_to_room(
                int(room_id),
                int(merged_node.id),
                allow_reassignment=True,
                reason="semantic_perception_proposal",
            )

        return merged_node, is_new, None

    def _proposal_attributes(
        self,
        proposal,
        embedding: Optional[np.ndarray],
        *,
        only_present: bool = False,
    ) -> Dict[str, object]:
        """Convert message metadata to stable, JSON-safe object attributes.

        With ``only_present`` the result omits fields the proposal left blank so
        an update never overwrites richer state already stored on the node.
        """
        attributes: Dict[str, object] = {}

        class_name = str(getattr(proposal, "class_name", "") or "").strip()
        if class_name or not only_present:
            attributes["class_name"] = class_name
        class_id = self._nonnegative_int(getattr(proposal, "class_id", 0), default=0)
        if class_id or not only_present:
            attributes["class_id"] = class_id
        detector_source = str(getattr(proposal, "detector_source", "") or "").strip()
        if detector_source or not only_present:
            attributes["detector_source"] = detector_source

        attributes["detection_confidence"] = self._finite_scalar(
            getattr(proposal, "detection_confidence", None)
        )
        for name in ("similarity_score", "entropy_score"):
            score = self._finite_scalar(getattr(proposal, name, None))
            attributes[name] = 0.0 if score is None else score
        attributes["semantic_perception_id"] = self._nonnegative_int(
            getattr(proposal, "id", 0), default=0
        )
        attributes["valid_3d"] = bool(getattr(proposal, "valid_3d", False))

        if not only_present:
            attributes["object_embedding"] = (
                embedding.astype(np.float32).tolist() if embedding is not None else None
            )
            attributes["embedding_observation_count"] = 1 if embedding is not None else 0

        size = getattr(getattr(proposal, "bbox_3d", None), "size", None)
        if size is not None and all(
            self._is_finite(getattr(size, axis, None)) for axis in ("x", "y", "z")
        ):
            attributes["bbox_3d_size"] = [
                float(size.x),
                float(size.y),
                float(size.z),
            ]

        return attributes

    def _update_spatial_representation(
        self,
        node: BaseNode,
        world_pose: Pose,
        previous_observations: int,
    ) -> None:
        """Fold one observed pose into the node using the configured policy."""
        if self.position_update_policy == "running_mean" and previous_observations > 0:
            weight = 1.0 / float(previous_observations + 1)
            node.pose.position.x += weight * (
                float(world_pose.position.x) - float(node.pose.position.x)
            )
            node.pose.position.y += weight * (
                float(world_pose.position.y) - float(node.pose.position.y)
            )
            node.pose.position.z += weight * (
                float(world_pose.position.z) - float(node.pose.position.z)
            )
        else:
            node.pose.position.x = world_pose.position.x
            node.pose.position.y = world_pose.position.y
            node.pose.position.z = world_pose.position.z

        node.pose.orientation.x = world_pose.orientation.x
        node.pose.orientation.y = world_pose.orientation.y
        node.pose.orientation.z = world_pose.orientation.z
        node.pose.orientation.w = world_pose.orientation.w

    def _find_or_add(
        self,
        node: BaseNode,
        incoming_embedding: Optional[np.ndarray] = None,
        range_m: float = 0.75,
        room_id: Optional[int] = None,
    ) -> Tuple[BaseNode, bool, Optional[float]]:
        """Associate by distance and cosine similarity, or insert a new node.

        A candidate is only accepted when it lies within ``range_m``, belongs to
        the same room as the incoming observation, and its stored
        ``object_embedding`` reaches ``semantic_similarity_threshold``. The best
        candidate is the most similar one, ties broken by distance then node ID.

        Returns:
            Tuple of (node, is_new, similarity). ``similarity`` is ``None`` for
            newly created nodes.
        """
        candidates_with_dist = self.sg.query.find_nodes_by_position_xyz(
            node.pose.position, range_m, node_type=node.node_type
        )

        best_candidate: Optional[BaseNode] = None
        best_similarity: Optional[float] = None
        best_distance = float("inf")
        best_id = float("inf")

        for candidate_node, distance in candidates_with_dist:
            if candidate_node.layer != node.layer:
                continue
            if not self._candidate_is_in_room(candidate_node, room_id):
                continue

            similarity = cosine_similarity(
                incoming_embedding,
                (candidate_node.attributes or {}).get("object_embedding"),
            )
            # Missing or dimensionally inconsistent embeddings never associate.
            if similarity is None or similarity < self.semantic_similarity_threshold:
                continue

            candidate_id = (
                int(candidate_node.id) if candidate_node.id is not None else -1
            )
            candidate_distance = float(distance)

            if (
                best_candidate is None
                or similarity > float(best_similarity)
                or (
                    math.isclose(similarity, float(best_similarity), rel_tol=1e-12)
                    and (candidate_distance, candidate_id) < (best_distance, best_id)
                )
            ):
                best_candidate = candidate_node
                best_similarity = float(similarity)
                best_distance = candidate_distance
                best_id = candidate_id

        if best_candidate is not None:
            self.stats["total_objects_associated"] += 1
            return best_candidate, False, best_similarity

        node_id = self.sg.update.add_node(node)
        node.id = node_id  # Update node with assigned ID
        return node, True, None

    def _candidate_is_in_room(
        self,
        candidate_node: BaseNode,
        incoming_room_id: Optional[int],
    ) -> bool:
        """Keep association candidates within one explicit room transition."""
        candidate_room_id = self._room_id_for_object(candidate_node)
        if incoming_room_id is None:
            return candidate_room_id is None
        return (
            candidate_room_id is None
            or int(candidate_room_id) == int(incoming_room_id)
        )

    def _room_id_for_object(self, object_node: BaseNode) -> Optional[int]:
        """Return the single room that currently owns one object node."""
        if object_node.id is None:
            return None
        if self.room_manager is not None:
            return self.room_manager.get_room_id_for_direct_member(int(object_node.id))

        room_ids: List[int] = []
        for edge in self.sg.query.get_incoming_edges(
            int(object_node.id), EdgeType.ROOM_CONTAINS
        ):
            room_node = self.sg.query.get_node(int(edge.source_id))
            if room_node is not None and room_node.node_type == NodeType.ROOM:
                room_ids.append(int(room_node.id))
        return min(room_ids) if room_ids else None

    def _create_observation_edge(self, pose_node: BaseNode, obj_node: BaseNode) -> bool:
        """
        Create OBSERVATION_ANCHOR edge from pose to object.

        Args:
            pose_node: Pose node (observer)
            obj_node: Object node (observed)

        Returns:
            True if edge was created, False otherwise
        """
        if pose_node is None or obj_node is None:
            return False

        if pose_node.id is None or obj_node.id is None:
            self.logger.warn("Cannot create observation edge: node IDs are None")
            return False

        # Check if edge already exists to avoid duplicates
        if self.sg.query.has_edge(
            pose_node.id, obj_node.id, edge_type=EdgeType.OBSERVATION_ANCHOR
        ):
            self.logger.debug(
                f"OBSERVATION_ANCHOR edge already exists: pose {pose_node.id} -> object {obj_node.id}"
            )
            return False

        # Create edge (relational, not structural)
        edge = Edge(
            source_id=pose_node.id,
            target_id=obj_node.id,
            type=EdgeType.OBSERVATION_ANCHOR,
        )
        self.sg.update.add_edge(edge, is_structural=False)
        self.stats["total_observation_edges_created"] += 1

        self.logger.debug(
            f"Created OBSERVATION_ANCHOR edge: pose {pose_node.id} -> object {obj_node.id}"
        )

        return True

    def _create_observation_edges_batch(
        self,
        pose_node: BaseNode,
        object_nodes: List[BaseNode],
        existing_target_ids: Optional[Set[int]] = None,
    ) -> int:
        """Create missing OBSERVATION_ANCHOR edges from one pose in a single patch."""
        if pose_node is None or pose_node.id is None or not object_nodes:
            return 0

        existing_target_ids = existing_target_ids or set()
        patch = GraphPatch()
        added_count = 0
        pose_id = int(pose_node.id)
        for obj_node in object_nodes:
            if obj_node is None or obj_node.id is None:
                continue
            object_id = int(obj_node.id)
            if object_id in existing_target_ids:
                continue
            patch.add_edge(
                Edge(
                    source_id=pose_id,
                    target_id=object_id,
                    type=EdgeType.OBSERVATION_ANCHOR,
                    is_structural=False,
                ),
                is_structural=False,
            )
            existing_target_ids.add(object_id)
            added_count += 1

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)
            self.stats["total_observation_edges_created"] += added_count

        return added_count

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid cell indices.

        Args:
            x, y: World coordinates (meters)

        Returns:
            Tuple of (grid_x, grid_y) cell indices
        """
        if self.occupancy_grid is None:
            return None, None

        grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
        grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)

        return grid_x, grid_y

    def _is_grid_cell_valid(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid cell is within bounds.

        Args:
            grid_x, grid_y: Grid cell indices

        Returns:
            True if cell is within grid bounds
        """
        if self.occupancy_grid is None:
            return False

        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height

    def _is_grid_cell_occupied(
        self, grid_x: int, grid_y: int, unknown_is_occupied: bool = True
    ) -> bool:
        """
        Check if grid cell is occupied or unknown.

        Args:
            grid_x, grid_y: Grid cell indices
            unknown_is_occupied: Treat unknown cells (-1) as occupied

        Returns:
            True if cell is occupied or (optionally) unknown
        """
        if not self._is_grid_cell_valid(grid_x, grid_y):
            return True  # Out of bounds = occupied

        cell_value = self.occupancy_grid[grid_y, grid_x]

        # OccupancyGrid values: -1=unknown, 0=free, 100=occupied
        if cell_value >= 50:  # Occupied threshold
            return True
        if unknown_is_occupied and cell_value < 0:  # Unknown
            return True

        return False

    def _raycast_grid_bresenham(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        unknown_is_occupied: bool = False,
    ) -> bool:
        """
        Raycast from (x0,y0) to (x1,y1) in world coordinates using Bresenham.

        Early-exit if any occupied cell is encountered along the ray.

        Args:
            x0, y0: Start position in world coordinates (pose)
            x1, y1: End position in world coordinates (object)
            unknown_is_occupied: Treat unknown cells as occupied (stricter)

        Returns:
            True if ray is clear (not occluded), False if occluded
        """
        if self.occupancy_grid is None:
            # No grid available - fall back to permissive (not occluded)
            return True

        # Convert to grid coordinates
        gx0, gy0 = self._world_to_grid(x0, y0)
        gx1, gy1 = self._world_to_grid(x1, y1)

        if gx0 is None or gx1 is None:
            # Conversion failed
            return True

        # Bresenham's line algorithm
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy

        gx, gy = gx0, gy0

        # March along the ray
        while True:
            # Check current cell (skip start and end cells)
            if not (gx == gx0 and gy == gy0) and not (gx == gx1 and gy == gy1):
                if self._is_grid_cell_occupied(gx, gy, unknown_is_occupied):
                    # Hit an occupied cell - ray is occluded
                    return False

            # Reached end
            if gx == gx1 and gy == gy1:
                break

            # Bresenham step
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx += sx
            if e2 < dx:
                err += dx
                gy += sy

        # Ray is clear
        return True

    def get_los_candidates(
        self, pose_node: BaseNode, radius: Optional[float] = None
    ) -> List[BaseNode]:
        """
        Get candidate object IDs for line-of-sight test from the scene graph.

        Uses spatial query to find objects within radius of the pose position.

        Args:
            pose_node: Current pose node
            radius: Search radius in metres (defaults to max_los_range)

        Returns:
            Object nodes within radius
        """
        if radius is None:
            radius = self.max_los_range

        # Spatial query: Find all OBJECT nodes within radius
        # Returns list of (node, distance) tuples
        nearby_nodes_with_dist = self.sg.query.find_nodes_by_position_xyz(
            pose_node.pose.position,
            max_range=radius,
            node_type=NodeType.OBJECT,
        )

        candidate_nodes: List[BaseNode] = [
            node for node, _ in nearby_nodes_with_dist if node.id is not None
        ]

        self.logger.debug(
            f"LoS candidates from spatial query: {len(candidate_nodes)} objects within {radius:.1f}m"
        )

        return candidate_nodes

    def compute_line_of_sight_for_pose(
        self,
        pose_node: BaseNode,
        max_los_range: Optional[float] = None,
        fov_deg: Optional[float] = None,
        unknown_is_occupied: Optional[bool] = None,
    ) -> List[int]:
        """
        Compute which objects are visible from *pose_node* using Bresenham raycasting
        on the occupancy grid.

        Algorithm
        ---------
        1. Spatial query — collect all OBJECT nodes within *max_los_range*.
        2. Distance filter — discard objects beyond *max_los_range*.
        3. FOV filter — discard objects outside the robot's field of view
           (centred on its heading; 360° means omnidirectional).
        4. Bresenham raycast — march from the robot grid cell to the object
           grid cell along a straight line. The ray is **blocked** by any grid
           cell whose value is ≥ 50 (occupied). Unknown cells (value == -1)
           block the ray when *unknown_is_occupied* is True.
        5. If the ray is clear → object is visible.
        6. Persist visible IDs into ``pose_node.attributes["object_in_los"]``
           and update the node in the scene graph.

        Assumptions
        -----------
        * The occupancy grid and the pose/object positions share the same 2-D
          coordinate frame (x-forward, y-left).
        * Only the x-y plane is considered (3-D objects are projected to 2-D).
        * When no occupancy grid is available the method returns an empty list
          without raising an error.

        Args:
            pose_node:            Current robot pose node.
            max_los_range:        Maximum sensing range in metres.
                                  Defaults to ``self.max_los_range``.
            fov_deg:              Field-of-view angle in degrees, centred on the
                                  robot heading.  360 means fully omnidirectional.
                                  Defaults to ``self.los_fov_deg``.
            unknown_is_occupied:  If True, unknown grid cells (-1) are treated as
                                  obstacles during raycasting.
                                  Defaults to ``self.los_unknown_is_occupied``.

        Returns:
            List of object IDs confirmed visible from *pose_node*.
        """
        if pose_node is None or pose_node.id is None:
            return []

        # Resolve defaults
        if max_los_range is None:
            max_los_range = self.max_los_range
        if fov_deg is None:
            fov_deg = self.los_fov_deg
        if unknown_is_occupied is None:
            unknown_is_occupied = self.los_unknown_is_occupied

        # Fetch latest pose state from the graph
        current_pose = self.sg.query.get_node(pose_node.id)
        if current_pose is None:
            self.logger.warn(f"LoS: pose node {pose_node.id} not found in graph")
            return []

        if self.occupancy_grid is None:
            self.logger.debug("LoS skipped: no occupancy grid available")
            self._persist_pose_los(current_pose, set())
            return []

        # Collect candidates within range
        candidate_nodes = self.get_los_candidates(current_pose, radius=max_los_range)
        if not candidate_nodes:
            self.logger.debug(
                f"LoS pose {current_pose.id}: no candidates within {max_los_range:.1f}m"
            )
            self._persist_pose_los(current_pose, set())
            return []

        # Precompute FOV half-angle and robot heading (only needed for <360° FOV)
        half_fov_rad: Optional[float] = None
        robot_yaw: float = 0.0
        if fov_deg < 360.0:
            half_fov_rad = np.deg2rad(fov_deg / 2.0)
            q = current_pose.pose.orientation
            robot_yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2),
            )

        rx = float(current_pose.pose.position.x)
        ry = float(current_pose.pose.position.y)
        raycast = self._raycast_grid_bresenham

        visible_ids: Set[int] = set()
        visible_nodes: List[BaseNode] = []
        occluded_count = 0
        fov_filtered = 0
        existing_observation_targets = {
            int(edge.target_id)
            for edge in self.sg.query.get_outgoing_edges(
                int(current_pose.id),
                edge_type=EdgeType.OBSERVATION_ANCHOR,
            )
        }

        for obj_node in candidate_nodes:
            obj_id = int(obj_node.id)

            ox = float(obj_node.pose.position.x)
            oy = float(obj_node.pose.position.y)
            dx = ox - rx
            dy = oy - ry
            dist = np.hypot(dx, dy)

            # ── Filter 1: distance ────────────────────────────────────────────
            if dist > max_los_range:
                continue  # already filtered by spatial query, but be defensive

            # ── Filter 2: field of view ───────────────────────────────────────
            if half_fov_rad is not None:
                angle = np.arctan2(dy, dx)
                # Normalise relative angle to [-π, π]
                delta = (angle - robot_yaw + np.pi) % (2.0 * np.pi) - np.pi
                if abs(delta) > half_fov_rad:
                    fov_filtered += 1
                    continue

            # ── Filter 3: Bresenham raycast on occupancy grid ─────────────────
            if raycast(rx, ry, ox, oy, unknown_is_occupied):
                visible_ids.add(obj_id)
                visible_nodes.append(obj_node)
            else:
                occluded_count += 1

        if visible_nodes:
            self._create_observation_edges_batch(
                current_pose,
                visible_nodes,
                existing_target_ids=existing_observation_targets,
            )

        # Persist and log
        self._persist_pose_los(current_pose, visible_ids)

        self.logger.debug(
            f"LoS pose {current_pose.id}: {len(candidate_nodes)} candidates → "
            f"{len(visible_ids)} visible, {occluded_count} occluded, "
            f"{fov_filtered} outside FOV"
        )

        return list(visible_ids)

    def _persist_pose_los(self, pose_node: BaseNode, visible_ids: Set[int]) -> None:
        """
        Write *visible_ids* into ``pose_node.attributes["object_in_los"]`` and
        persist the updated node to the scene graph.

        The attribute is **replaced** (not accumulated) so that each call
        reflects the current visibility state of the pose.

        Args:
            pose_node:   Pose node to update.
            visible_ids: Set of currently visible object IDs.
        """
        self._initialize_pose_attributes(pose_node)
        pose_node.attributes["object_in_los"] = sorted(int(v) for v in visible_ids)
        self.sg.update.update_node(pose_node.id, pose_node)

    def get_statistics(self) -> dict:
        """
        Get manager statistics.

        Returns:
            Dictionary with statistics (objects created, updated, etc.)
        """
        return self.stats.copy()
