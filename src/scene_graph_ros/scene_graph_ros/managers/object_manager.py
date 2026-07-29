"""
Object Manager - Manage semantic-perception object proposals and observation links.

Proposals arrive already projected into the graph frame by ``semantic_perception``
(the transform is applied at the image timestamp there), so this manager only
validates the frame, never re-projects.

Association policy, in order:

1. Collect OBJECT nodes within ``spatial_association_distance`` of the proposal.
2. No nearby node -> create a new object node.
3. Nearby nodes -> score each candidate's ``object_embedding`` against the
   proposal's ``fused_embedding`` with cosine similarity.
4. Best candidate at or above ``semantic_similarity_threshold`` -> update it.
5. Nearby nodes but none similar enough -> the detection is *spatially ambiguous*
   and no node is created. Semantic disagreement is never treated as evidence
   that a second physical object occupies the same place.

Usage:
    obj_mgr = ObjectNodeManager(sg_interface, logger, room_manager=room_mgr, ...)
    obj_mgr.process_detections_update(proposals_msg, room_id=room_id)
"""

import math
import time
import traceback
from collections import Counter
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from nav_msgs.msg import OccupancyGrid

from scene_graph_core.algorithms.semantic import (
    accumulate_class_evidence,
    accumulate_embedding,
    canonical_class_from_evidence,
    cosine_similarity,
    normalize_embedding,
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


class _Point:
    """Minimal position holder for spatial queries, without a ROS message."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _as_list(vector) -> Optional[List[float]]:
    """Return a JSON-safe list of floats, or ``None`` for a missing vector."""
    if vector is None:
        return None
    return [float(value) for value in np.asarray(vector, dtype=np.float64).reshape(-1)]


class ObjectNodeManager:
    """
    Manages OBJECT nodes representing detected objects in the OBJECT layer.

    This class turns ``semantic_perception`` proposals into OBJECT nodes, keeps a
    sum-based embedding accumulator and accumulated class evidence on each node,
    and performs occupancy-grid line-of-sight checks that write observation edges
    to visible objects.
    """

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        # Object association parameters
        spatial_association_distance: float = 0.75,  # metres
        semantic_similarity_threshold: float = 0.70,  # cosine similarity
        position_update_policy: str = "running_mean",  # "latest" or "running_mean"
        room_manager=None,
        room_resolver: Optional[Callable[[float, float], Optional[int]]] = None,
        # Line-of-sight parameters
        max_los_range: float = 10.0,  # metres — maximum sensing range for LoS
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
            spatial_association_distance: Radius (m) searched for candidate objects
            semantic_similarity_threshold: Minimum cosine similarity for association
            position_update_policy: "running_mean" averages associated observations,
                "latest" keeps the newest observed position
            room_manager: RoomManager owning ROOM_CONTAINS attachment
            room_resolver: Callable mapping an (x, y) graph position to a room ID,
                or None when the position lies outside every known region
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

        self.spatial_association_distance = max(0.0, float(spatial_association_distance))
        self.semantic_similarity_threshold = float(semantic_similarity_threshold)
        self.position_update_policy = str(position_update_policy).strip().lower()
        self.room_manager = room_manager
        self.room_resolver = room_resolver
        self.max_los_range = max_los_range
        self.los_fov_deg = los_fov_deg
        self.los_unknown_is_occupied = los_unknown_is_occupied

        self.enable_debug_logging = enable_debug_logging
        self.debug_log_interval = debug_log_interval
        self.update_counter = 0

        # Occupancy grid for Bresenham raycasting
        self.occupancy_grid = None
        self.grid_resolution: Optional[float] = None
        self.grid_origin: Optional[Tuple[float, float]] = None
        self.grid_width: Optional[int] = None
        self.grid_height: Optional[int] = None

        self.stats = {
            "total_detection_messages": 0,
            "total_detections_received": 0,
            "total_detections_accepted": 0,
            "total_detections_rejected": 0,
            "rejected_by_reason": {},
            "objects_created": 0,
            "objects_updated": 0,
            "detections_spatially_ambiguous": 0,
            "rejected_invalid_geometry": 0,
            "rejected_invalid_embedding": 0,
            "rejected_missing_tf": 0,
            "total_observation_edges_created": 0,
            "last_detection_stamp_sec": None,
            "last_object_create_stamp_sec": None,
            "last_object_update_stamp_sec": None,
        }
        self._rejected_by_reason = Counter()

        self.logger.debug("ObjectNodeManager initialized:")
        self.logger.debug(
            f"  - spatial_association_distance: {self.spatial_association_distance}m"
        )
        self.logger.debug(
            f"  - semantic_similarity_threshold: {self.semantic_similarity_threshold}"
        )
        self.logger.debug(f"  - position_update_policy: {self.position_update_policy}")
        self.logger.debug(f"  - max_los_range: {self.max_los_range}m")
        self.logger.debug(f"  - los_fov_deg: {self.los_fov_deg}\u00b0")
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
        self.logger.debug(f"  Objects created: {self.stats['objects_created']}")
        self.logger.debug(f"  Objects updated: {self.stats['objects_updated']}")
        self.logger.debug(
            f"  Spatially ambiguous: {self.stats['detections_spatially_ambiguous']}"
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
        fixed_frame_id: str = "world",
        room_id: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Create or update objects from semantic-perception proposals.

        Args:
            proposals_msg: ObjectProposal3DArray already expressed in the graph frame
            fixed_frame_id: Graph frame the proposals must be published in
            room_id: Room owning the current viewpoint, used only as a fallback when
                no region-based resolver is configured

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
        proposal_frame = str(getattr(header, "frame_id", "") or "")
        if proposal_frame != str(fixed_frame_id):
            # semantic_perception publishes in the graph frame; anything else
            # would mean comparing coordinates from two different frames.
            self.stats["rejected_missing_tf"] += detection_count
            self.stats["total_detections_rejected"] += detection_count
            self._record_detection_rejection("frame_mismatch", detection_count)
            self.logger.warning(
                f"[object_proposals] rejected message: frame {proposal_frame!r} "
                f"is not the graph frame {fixed_frame_id!r}",
                throttle_duration_sec=5.0,
            )
            return self._empty_update_stats(
                rejected=detection_count,
                rejected_by_reason={"frame_mismatch": detection_count},
            )

        timestamp = self._message_timestamp(header)
        self.stats["last_detection_stamp_sec"] = timestamp

        new_objects = 0
        updated_objects = 0
        ambiguous_detections = 0
        accepted_detections = 0
        rejected_by_reason = Counter()
        new_object_ids: List[int] = []
        updated_object_ids: List[int] = []

        for detection_index, proposal in enumerate(proposals):
            obj_node, outcome = self._process_single_detection(
                proposal,
                timestamp,
                detection_index=detection_index,
                room_id=room_id,
            )

            if outcome == "created":
                accepted_detections += 1
                new_objects += 1
                if obj_node is not None and obj_node.id is not None:
                    new_object_ids.append(int(obj_node.id))
            elif outcome == "updated":
                accepted_detections += 1
                updated_objects += 1
                if obj_node is not None and obj_node.id is not None:
                    updated_object_ids.append(int(obj_node.id))
            elif outcome == "spatially_ambiguous":
                ambiguous_detections += 1
                rejected_by_reason["spatially_ambiguous"] += 1
            else:
                rejected_by_reason[str(outcome or "unknown")] += 1

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
            "ambiguous_detections": ambiguous_detections,
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
            "ambiguous_detections": 0,
            "new_object_ids": [],
            "updated_object_ids": [],
        }

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
        detection_index: int = 0,
        room_id: Optional[int] = None,
    ) -> Tuple[Optional[BaseNode], str]:
        """Process one ``semantic_perception_msgs/ObjectProposal3D``.

        Returns:
            Tuple of (object_node, outcome) where outcome is one of ``created``,
            ``updated``, ``spatially_ambiguous``, or a rejection reason.
        """
        confidence = self._finite_scalar(getattr(proposal, "detection_confidence", None))
        if confidence is None:
            self.stats["rejected_invalid_geometry"] += 1
            self.logger.warning(
                f"[object_proposals] rejected idx={detection_index}: "
                "nonfinite_detection_confidence",
                throttle_duration_sec=5.0,
            )
            return None, "nonfinite_detection_confidence"

        if not bool(getattr(proposal, "valid_3d", False)):
            self.stats["rejected_invalid_geometry"] += 1
            self.logger.warning(
                f"[object_proposals] rejected idx={detection_index}: invalid_3d_geometry",
                throttle_duration_sec=5.0,
            )
            return None, "invalid_3d_geometry"

        centroid = getattr(proposal, "centroid_3d", None)
        if centroid is None or not all(
            self._is_finite(getattr(centroid, axis, None)) for axis in ("x", "y", "z")
        ):
            self.stats["rejected_invalid_geometry"] += 1
            self.logger.warning(
                f"[object_proposals] rejected idx={detection_index}: nonfinite_centroid",
                throttle_duration_sec=5.0,
            )
            return None, "nonfinite_centroid"

        position = (float(centroid.x), float(centroid.y), float(centroid.z))

        fused_embedding = normalize_embedding(getattr(proposal, "fused_embedding", None))
        if fused_embedding is None:
            self.stats["rejected_invalid_embedding"] += 1
            self.logger.warning(
                f"[object_proposals] rejected idx={detection_index}: invalid_fused_embedding",
                throttle_duration_sec=5.0,
            )
            return None, "invalid_fused_embedding"

        class_name = " ".join(str(getattr(proposal, "class_name", "") or "").split())
        if not class_name:
            self.stats["rejected_invalid_embedding"] += 1
            self.logger.warning(
                f"[object_proposals] rejected idx={detection_index}: missing_class_name",
                throttle_duration_sec=5.0,
            )
            return None, "missing_class_name"

        target_room_id = self._resolve_room_id(position, room_id)

        try:
            candidates = self._find_candidates(position, target_room_id)
            match, similarity = self._select_best_candidate(
                candidates, fused_embedding
            )
        except Exception as exc:
            self.logger.error(
                f"[object_proposals] rejected idx={detection_index}: "
                f"graph_query_failed: {exc}\n{traceback.format_exc()}"
            )
            return None, "graph_query_failed"

        try:
            if match is not None:
                node = self._update_object_node(
                    match, proposal, position, fused_embedding, class_name,
                    confidence, timestamp, similarity,
                )
                outcome = "updated"
            elif candidates:
                # Nearby objects exist but none is semantically compatible. This
                # is ambiguity, not evidence of a second object at the same place.
                self.stats["detections_spatially_ambiguous"] += 1
                self.logger.debug(
                    f"[object_proposals] ambiguous idx={detection_index} "
                    f"class={class_name!r} candidates={len(candidates)} "
                    f"best_similarity={similarity} "
                    f"threshold={self.semantic_similarity_threshold}",
                    throttle_duration_sec=5.0,
                )
                return None, "spatially_ambiguous"
            else:
                node = self._create_object_node(
                    proposal, position, fused_embedding, class_name,
                    confidence, timestamp,
                )
                outcome = "created"
        except Exception as exc:
            self.logger.error(
                f"[object_proposals] rejected idx={detection_index}: "
                f"graph_mutation_failed: {exc}\n{traceback.format_exc()}"
            )
            return None, "graph_mutation_failed"

        self.sg.update.update_node(node.id, node)
        self._attach_to_room(node, target_room_id)

        if outcome == "created":
            self.stats["objects_created"] += 1
            self.stats["last_object_create_stamp_sec"] = timestamp
            self.logger.debug(
                f"[object_proposals] created idx={detection_index} "
                f"object_id={node.id} class={class_name!r} room={target_room_id} "
                f"position=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
        else:
            self.stats["objects_updated"] += 1
            self.stats["last_object_update_stamp_sec"] = timestamp
            self.logger.debug(
                f"[object_proposals] updated idx={detection_index} "
                f"object_id={node.id} class={class_name!r} room={target_room_id} "
                f"similarity={similarity:.3f} "
                f"position=({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
        return node, outcome

    def _resolve_room_id(
        self,
        position: Tuple[float, float, float],
        fallback_room_id: Optional[int],
    ) -> Optional[int]:
        """Resolve the owning room from region polygons at this position."""
        if self.room_resolver is not None:
            return self.room_resolver(position[0], position[1])
        return fallback_room_id

    def _find_candidates(
        self,
        position: Tuple[float, float, float],
        room_id: Optional[int],
    ) -> List[Tuple[BaseNode, float]]:
        """Return nearby OBJECT nodes eligible for association, nearest first."""
        query = _Point(*position)
        nearby = self.sg.query.find_nodes_by_position_xyz(
            query, self.spatial_association_distance, node_type=NodeType.OBJECT
        )
        candidates: List[Tuple[BaseNode, float]] = []
        for node, distance in nearby:
            if node is None or node.id is None or node.layer != ObjectNode().layer:
                continue
            if not self._has_valid_geometry(node):
                continue
            if not self._room_is_compatible(node, room_id):
                continue
            candidates.append((node, float(distance)))
        candidates.sort(key=lambda item: (item[1], int(item[0].id)))
        return candidates

    @staticmethod
    def _has_valid_geometry(node: BaseNode) -> bool:
        position = getattr(getattr(node, "pose", None), "position", None)
        if position is None:
            return False
        return all(
            ObjectNodeManager._is_finite(getattr(position, axis, None))
            for axis in ("x", "y", "z")
        )

    def _room_is_compatible(self, node: BaseNode, room_id: Optional[int]) -> bool:
        """Objects only associate inside one room, or when neither has a room."""
        candidate_room_id = self._room_id_for_object(node)
        if room_id is None:
            return candidate_room_id is None
        return candidate_room_id is None or int(candidate_room_id) == int(room_id)

    def _select_best_candidate(
        self,
        candidates: List[Tuple[BaseNode, float]],
        fused_embedding: np.ndarray,
    ) -> Tuple[Optional[BaseNode], Optional[float]]:
        """Return the best semantically compatible candidate and its similarity.

        Ranking is highest cosine similarity, then smaller spatial distance, then
        lower node ID. Candidates whose stored embedding is missing or of another
        dimension score ``None`` and can never win.
        """
        best_node: Optional[BaseNode] = None
        best_similarity: Optional[float] = None
        best_key: Optional[Tuple[float, float, int]] = None
        observed_best: Optional[float] = None

        for node, distance in candidates:
            stored = (node.attributes or {}).get("object_embedding")
            similarity = cosine_similarity(fused_embedding, stored)
            if similarity is None:
                continue
            if observed_best is None or similarity > observed_best:
                observed_best = similarity
            if similarity < self.semantic_similarity_threshold:
                continue
            key = (-float(similarity), float(distance), int(node.id))
            if best_key is None or key < best_key:
                best_node = node
                best_similarity = float(similarity)
                best_key = key

        if best_node is not None:
            return best_node, best_similarity
        return None, observed_best

    def _create_object_node(
        self,
        proposal,
        position: Tuple[float, float, float],
        fused_embedding: np.ndarray,
        class_name: str,
        confidence: float,
        timestamp: float,
    ) -> BaseNode:
        """Insert a new OBJECT node seeded from one proposal."""
        node = ObjectNode()
        node.pose.position.x = position[0]
        node.pose.position.y = position[1]
        node.pose.position.z = position[2]
        node.pose.orientation.w = 1.0
        node.created_at = timestamp
        node.last_seen = timestamp

        embedding_sum, embedding_count, mean = accumulate_embedding(
            None, 0, fused_embedding
        )
        evidence = accumulate_class_evidence(None, class_name, confidence)
        canonical_name, canonical_confidence = canonical_class_from_evidence(evidence)

        node.attributes = {
            "class_name": canonical_name or class_name,
            "class_confidence": canonical_confidence,
            "class_evidence": evidence,
            "detection_observation_count": 1,
            "embedding_observation_count": int(embedding_count),
            "embedding_sum": _as_list(embedding_sum),
            "object_embedding": _as_list(mean),
            "first_seen": timestamp,
        }
        node.attributes.update(self._canonical_components(proposal))
        node.id = self.sg.update.add_node(node)
        return node

    def _update_object_node(
        self,
        node: BaseNode,
        proposal,
        position: Tuple[float, float, float],
        fused_embedding: np.ndarray,
        class_name: str,
        confidence: float,
        timestamp: float,
        similarity: Optional[float],
    ) -> BaseNode:
        """Fold one proposal into an existing OBJECT node."""
        attributes = node.attributes if isinstance(node.attributes, dict) else {}
        node.attributes = attributes

        detection_count = self._nonnegative_int(
            attributes.get("detection_observation_count"), default=0
        )
        embedding_count = self._nonnegative_int(
            attributes.get("embedding_observation_count"), default=0
        )
        embedding_sum, embedding_count, mean = accumulate_embedding(
            attributes.get("embedding_sum"), embedding_count, fused_embedding
        )

        evidence = accumulate_class_evidence(
            attributes.get("class_evidence"), class_name, confidence
        )
        canonical_name, canonical_confidence = canonical_class_from_evidence(evidence)

        attributes["detection_observation_count"] = detection_count + 1
        attributes["embedding_observation_count"] = int(embedding_count)
        attributes["embedding_sum"] = _as_list(embedding_sum)
        attributes["object_embedding"] = _as_list(mean)
        attributes["class_evidence"] = evidence
        attributes["class_name"] = canonical_name or class_name
        attributes["class_confidence"] = canonical_confidence
        attributes.setdefault("first_seen", node.created_at)
        if similarity is not None:
            attributes["last_semantic_similarity"] = float(similarity)

        # The canonical CLIP components come from an observation of the canonical
        # class, so they never describe a class the node was not resolved to.
        if class_name == attributes["class_name"]:
            attributes.update(self._canonical_components(proposal))

        self._update_spatial_representation(node, position, detection_count)
        node.last_seen = timestamp
        return node

    def _canonical_components(self, proposal) -> Dict[str, object]:
        """Return the per-observation CLIP components kept on the node."""
        components: Dict[str, object] = {}
        for name in ("mask_embedding", "bbox_embedding", "label_embedding", "fused_embedding"):
            vector = normalize_embedding(getattr(proposal, name, None))
            components[name] = _as_list(vector)
        components["detection_confidence"] = self._finite_scalar(
            getattr(proposal, "detection_confidence", None)
        )
        components["detector_source"] = str(
            getattr(proposal, "detector_source", "") or ""
        )
        components["semantic_perception_class_id"] = self._nonnegative_int(
            getattr(proposal, "class_id", 0), default=0
        )
        size = getattr(getattr(proposal, "bbox_3d", None), "size", None)
        if size is not None and all(
            self._is_finite(getattr(size, axis, None)) for axis in ("x", "y", "z")
        ):
            components["bbox_3d_size"] = [float(size.x), float(size.y), float(size.z)]
        return components

    def _update_spatial_representation(
        self,
        node: BaseNode,
        position: Tuple[float, float, float],
        previous_observations: int,
    ) -> None:
        """Fold one observed position into the node using the configured policy."""
        if self.position_update_policy == "running_mean" and previous_observations > 0:
            weight = 1.0 / float(previous_observations + 1)
            node.pose.position.x += weight * (position[0] - float(node.pose.position.x))
            node.pose.position.y += weight * (position[1] - float(node.pose.position.y))
            node.pose.position.z += weight * (position[2] - float(node.pose.position.z))
        else:
            node.pose.position.x = position[0]
            node.pose.position.y = position[1]
            node.pose.position.z = position[2]

    def _attach_to_room(self, node: BaseNode, room_id: Optional[int]) -> None:
        """Attach or detach the object's single ROOM_CONTAINS ownership edge."""
        if self.room_manager is None or node.id is None:
            return
        if room_id is None:
            self.room_manager.detach_direct_member_from_rooms(int(node.id))
            return
        self.room_manager.attach_direct_member_to_room(
            int(room_id),
            int(node.id),
            allow_reassignment=True,
            reason="semantic_perception_proposal",
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
