"""Geometry-only region helpers for the scene-graph runtime."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Tuple

from incremental_dude_msgs.msg import Region2D, Region2DArray
from scene_graph_ros.managers.region_util import (
    PreparedGeometry,
    point_in_polygon,
    prepare_region_geometry,
)
from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import EdgeType, NodeType
from shapely.geometry import Point, box


@dataclass(frozen=True)
class PreparedTrackerRegion:
    """One stable tracker region prepared for point-in-polygon queries."""

    tracker_region_id: int
    region_msg: Region2D
    prepared_geometry: PreparedGeometry
    bounds: Tuple[float, float, float, float]
    signature: Tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]


@dataclass(frozen=True)
class MembershipResolution:
    """Resolved tracker-region ownership for one graph node."""

    tracker_region_id: Optional[int]
    reason: str
    plausible_tracker_region_ids: Tuple[int, ...] = ()
    used_tiebreak: bool = False


@dataclass(frozen=True)
class NavigationRegionCandidate:
    """One plausible REGION match for a NAVIGATION node."""

    tracker_region_id: int
    reason: str
    centroid_inside: bool
    boundary_within_epsilon: bool
    overlap_ratio: float
    overlap_area: float
    polygon_distance: float
    centroid_distance: float
    previous_membership_match: bool
    neighbor_support: int = 0


@dataclass(frozen=True)
class PointRegionCandidate:
    """One plausible REGION match for a point-like node near a boundary."""

    tracker_region_id: int
    reason: str
    point_inside: bool
    boundary_within_epsilon: bool
    polygon_distance: float
    centroid_distance: float
    previous_membership_match: bool


class RegionManager:
    """Prepare tracker geometry and resolve runtime region membership."""

    MEANINGFUL_MEMBER_TYPES = (
        NodeType.OBJECT,
        NodeType.AGENT,
        NodeType.NAVIGATION,
    )

    @classmethod
    def is_meaningful_member_type(cls, node_type: Optional[NodeType]) -> bool:
        """Return whether one node type counts as semantic region support."""
        return node_type in cls.MEANINGFUL_MEMBER_TYPES

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        z_offset: float = 8.0,
        nav_region_boundary_epsilon_m: float = 0.15,
        nav_region_enable_neighbor_tiebreak: bool = True,
    ):
        self.sg = sg_interface
        self.logger = logger
        self.z_offset = float(z_offset)
        self.nav_region_boundary_epsilon_m = max(
            0.0, float(nav_region_boundary_epsilon_m)
        )
        self.nav_region_enable_neighbor_tiebreak = bool(
            nav_region_enable_neighbor_tiebreak
        )

        self.current_tracker_region_id: Optional[int] = None

        self._prepared_region_cache: Dict[int, PreparedTrackerRegion] = {}
        self._node_tracker_region_membership: Dict[int, Optional[int]] = {}

    def prepare_region_snapshot(
        self,
        stable_regions_msg: Optional[Region2DArray],
    ) -> Tuple[bool, Dict[int, PreparedTrackerRegion]]:
        """Prepare the current tracker-region geometry snapshot only."""
        if stable_regions_msg is None:
            return False, {}

        prepared_regions: Dict[int, PreparedTrackerRegion] = {}
        try:
            for region in stable_regions_msg.regions:
                polygon_points = tuple(
                    (float(point.x), float(point.y)) for point in region.polygon.points
                )
                convex_hull_points = tuple(
                    (float(point.x), float(point.y))
                    for point in region.convex_hull.points
                )
                signature = (polygon_points, convex_hull_points)
                tracker_region_id = int(region.id)
                cached = self._prepared_region_cache.get(tracker_region_id)
                if cached is not None and cached.signature == signature:
                    prepared_regions[tracker_region_id] = PreparedTrackerRegion(
                        tracker_region_id=tracker_region_id,
                        region_msg=region,
                        prepared_geometry=cached.prepared_geometry,
                        bounds=cached.bounds,
                        signature=signature,
                    )
                    continue

                prepared_geometry = prepare_region_geometry(
                    polygon_points=polygon_points,
                    convex_hull_points=convex_hull_points,
                    use_convex_hull=False,
                )
                if not prepared_geometry.is_valid:
                    continue

                xs = [point[0] for point in polygon_points] or [float(region.centroid.x)]
                ys = [point[1] for point in polygon_points] or [float(region.centroid.y)]
                prepared_region = PreparedTrackerRegion(
                    tracker_region_id=tracker_region_id,
                    region_msg=region,
                    prepared_geometry=prepared_geometry,
                    bounds=(min(xs), min(ys), max(xs), max(ys)),
                    signature=signature,
                )
                prepared_regions[tracker_region_id] = prepared_region
                self._prepared_region_cache[tracker_region_id] = prepared_region
        except Exception as exc:
            self.logger.warning(
                "[RegionManager] Failed to prepare stable-region snapshot: "
                f"{exc}"
            )
            return False, {}

        return True, prepared_regions

    def find_tracker_region_for_pose(
        self,
        pose_node_id: Optional[int],
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> Optional[int]:
        """Return the tracker region containing the given pose node."""
        if pose_node_id is None:
            return None

        pose_node = self.sg.query.get_node(int(pose_node_id))
        if pose_node is None or pose_node.node_type != NodeType.AGENT:
            return None

        resolution = self._resolve_tracker_region_for_point_node(
            pose_node,
            prepared_regions,
        )
        self._node_tracker_region_membership[int(pose_node_id)] = (
            int(resolution.tracker_region_id)
            if resolution.tracker_region_id is not None
            else None
        )
        return resolution.tracker_region_id

    def get_prepared_region(
        self,
        tracker_region_id: Optional[int],
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> Optional[PreparedTrackerRegion]:
        """Return the prepared region snapshot for one tracker-region id."""
        if tracker_region_id is None:
            return None
        return prepared_regions.get(int(tracker_region_id))

    def set_current_resolved_region(self, tracker_region_id: Optional[int]) -> None:
        """Cache the currently resolved tracker region."""
        self.current_tracker_region_id = (
            int(tracker_region_id) if tracker_region_id is not None else None
        )

    def gather_region_member_ids(
        self,
        prepared_region: PreparedTrackerRegion,
        *,
        prepared_regions: Optional[Dict[int, PreparedTrackerRegion]] = None,
        node_types: Iterable[NodeType] = (
            NodeType.OBJECT,
            NodeType.AGENT,
            NodeType.NAVIGATION,
        ),
    ) -> Dict[NodeType, set[int]]:
        """Return graph entities currently lying inside one tracker region."""
        members: Dict[NodeType, set[int]] = {
            node_type: set() for node_type in node_types
        }
        effective_prepared_regions = prepared_regions or {
            int(prepared_region.tracker_region_id): prepared_region
        }

        for node_type in node_types:
            for node in self.sg.query.find_nodes_by_type(node_type):
                if node.id is None:
                    continue
                resolution = self._resolve_tracker_region_for_node(
                    node,
                    effective_prepared_regions,
                )
                self._node_tracker_region_membership[int(node.id)] = (
                    int(resolution.tracker_region_id)
                    if resolution.tracker_region_id is not None
                    else None
                )
                if resolution.tracker_region_id == prepared_region.tracker_region_id:
                    members[node_type].add(int(node.id))

        return members

    def count_meaningful_member_ids(
        self,
        member_ids: Optional[Dict[NodeType, set[int]]],
    ) -> int:
        """Count meaningful members in one gathered member-id map."""
        if not member_ids:
            return 0
        return int(
            sum(
                len(member_ids.get(node_type, set()))
                for node_type in self.MEANINGFUL_MEMBER_TYPES
            )
        )

    def has_meaningful_member_ids(
        self,
        member_ids: Optional[Dict[NodeType, set[int]]],
    ) -> bool:
        """Return whether one gathered member-id map has semantic support."""
        return self.count_meaningful_member_ids(member_ids) > 0

    def _resolve_tracker_region_for_node(
        self,
        node,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> MembershipResolution:
        """Resolve one node to at most one tracker region."""
        if node is None or node.id is None:
            return MembershipResolution(tracker_region_id=None, reason="missing_node")

        if node.node_type == NodeType.NAVIGATION:
            return self._find_tracker_region_for_navigation_node(node, prepared_regions)

        return self._resolve_tracker_region_for_point_node(node, prepared_regions)

    def _resolve_tracker_region_for_point_node(
        self,
        node,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> MembershipResolution:
        """Resolve one point-like node using polygon inclusion plus boundary epsilon."""
        if node is None or node.id is None:
            return MembershipResolution(tracker_region_id=None, reason="missing_node")

        node_id = int(node.id)
        node_x = float(node.pose.position.x)
        node_y = float(node.pose.position.y)
        node_point = Point(node_x, node_y)
        previous_tracker_region_id = self._get_previous_tracker_region_id_for_node(node_id)

        candidates: list[PointRegionCandidate] = []
        for tracker_region_id, prepared_region in prepared_regions.items():
            candidate = self._build_point_region_candidate(
                node_point=node_point,
                node_x=node_x,
                node_y=node_y,
                tracker_region_id=int(tracker_region_id),
                prepared_region=prepared_region,
                previous_tracker_region_id=previous_tracker_region_id,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return MembershipResolution(
                tracker_region_id=None,
                reason="outside",
                plausible_tracker_region_ids=(),
            )

        plausible_tracker_region_ids = tuple(
            sorted(candidate.tracker_region_id for candidate in candidates)
        )
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                0 if candidate.point_inside else 1,
                not candidate.previous_membership_match,
                float(candidate.polygon_distance),
                float(candidate.centroid_distance),
                int(candidate.tracker_region_id),
            ),
        )
        selected = sorted_candidates[0]

        if selected.reason != "point":
            self.logger.debug(
                "[RegionManager] Point-node boundary fallback "
                f"node_id={node_id} "
                f"node_type={getattr(node.node_type, 'name', str(node.node_type))} "
                f"selected_tracker_region_id={selected.tracker_region_id} "
                f"reason={selected.reason} "
                f"boundary_distance={selected.polygon_distance:.3f}"
            )

        return MembershipResolution(
            tracker_region_id=selected.tracker_region_id,
            reason=selected.reason,
            plausible_tracker_region_ids=plausible_tracker_region_ids,
            used_tiebreak=len(sorted_candidates) > 1,
        )

    def _find_tracker_region_for_navigation_node(
        self,
        node,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> MembershipResolution:
        """Resolve one NAVIGATION node using centroid, boundary, and footprint tests."""
        if node is None or node.id is None or node.node_type != NodeType.NAVIGATION:
            return MembershipResolution(tracker_region_id=None, reason="outside")

        nav_node_id = int(node.id)
        nav_x = float(node.pose.position.x)
        nav_y = float(node.pose.position.y)
        nav_point = Point(nav_x, nav_y)
        nav_bounds = self._extract_navigation_bounds(node)
        nav_footprint = None
        footprint_area = 0.0
        if nav_bounds is not None:
            nav_footprint = box(nav_bounds[0], nav_bounds[1], nav_bounds[2], nav_bounds[3])
            footprint_area = float(nav_footprint.area)

        previous_tracker_region_id = self._get_previous_tracker_region_id_for_node(
            nav_node_id
        )

        candidates: list[NavigationRegionCandidate] = []
        for tracker_region_id, prepared_region in prepared_regions.items():
            candidate = self._build_navigation_region_candidate(
                node=node,
                nav_point=nav_point,
                nav_bounds=nav_bounds,
                nav_footprint=nav_footprint,
                footprint_area=footprint_area,
                tracker_region_id=int(tracker_region_id),
                prepared_region=prepared_region,
                previous_tracker_region_id=previous_tracker_region_id,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return MembershipResolution(
                tracker_region_id=None,
                reason="outside",
                plausible_tracker_region_ids=(),
            )

        if self.nav_region_enable_neighbor_tiebreak and len(candidates) > 1:
            candidates = [
                NavigationRegionCandidate(
                    tracker_region_id=candidate.tracker_region_id,
                    reason=candidate.reason,
                    centroid_inside=candidate.centroid_inside,
                    boundary_within_epsilon=candidate.boundary_within_epsilon,
                    overlap_ratio=candidate.overlap_ratio,
                    overlap_area=candidate.overlap_area,
                    polygon_distance=candidate.polygon_distance,
                    centroid_distance=candidate.centroid_distance,
                    previous_membership_match=candidate.previous_membership_match,
                    neighbor_support=self._count_navigation_neighbor_support(
                        nav_node_id,
                        candidate.tracker_region_id,
                    ),
                )
                for candidate in candidates
            ]

        plausible_tracker_region_ids = tuple(
            sorted(candidate.tracker_region_id for candidate in candidates)
        )
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                not candidate.previous_membership_match,
                -float(candidate.overlap_ratio),
                -float(candidate.overlap_area),
                float(candidate.polygon_distance),
                float(candidate.centroid_distance),
                -int(candidate.neighbor_support),
                int(candidate.tracker_region_id),
            ),
        )
        selected = sorted_candidates[0]
        used_tiebreak = len(sorted_candidates) > 1

        if used_tiebreak:
            self.logger.debug(
                "[RegionManager] NAV tie-break "
                f"nav_node_id={nav_node_id} "
                f"selected_tracker_region_id={selected.tracker_region_id} "
                f"plausible_tracker_regions={list(plausible_tracker_region_ids)} "
                f"reason={selected.reason}"
            )

        if selected.reason != "centroid":
            self.logger.debug(
                "[RegionManager] NAV fallback region assignment "
                f"nav_node_id={nav_node_id} "
                f"selected_tracker_region_id={selected.tracker_region_id} "
                f"reason={selected.reason} "
                f"overlap_ratio={selected.overlap_ratio:.3f} "
                f"boundary_distance={selected.polygon_distance:.3f}"
            )

        return MembershipResolution(
            tracker_region_id=selected.tracker_region_id,
            reason=selected.reason,
            plausible_tracker_region_ids=plausible_tracker_region_ids,
            used_tiebreak=used_tiebreak,
        )

    def _build_navigation_region_candidate(
        self,
        *,
        node,
        nav_point: Point,
        nav_bounds: Optional[Tuple[float, float, float, float]],
        nav_footprint,
        footprint_area: float,
        tracker_region_id: int,
        prepared_region: PreparedTrackerRegion,
        previous_tracker_region_id: Optional[int],
    ) -> Optional[NavigationRegionCandidate]:
        """Return one plausible REGION candidate for a NAVIGATION node."""
        nav_x = float(node.pose.position.x)
        nav_y = float(node.pose.position.y)
        epsilon = float(self.nav_region_boundary_epsilon_m)

        region_min_x, region_min_y, region_max_x, region_max_y = prepared_region.bounds
        expanded_bounds = (
            region_min_x - epsilon,
            region_min_y - epsilon,
            region_max_x + epsilon,
            region_max_y + epsilon,
        )
        if not self._point_in_bounds((nav_x, nav_y), expanded_bounds):
            if nav_bounds is None or not self._bounds_overlap(
                nav_bounds,
                expanded_bounds,
            ):
                return None

        polygon = prepared_region.prepared_geometry.polygon
        if polygon is None or polygon.is_empty:
            return None

        centroid_inside = point_in_polygon((nav_x, nav_y), polygon)
        polygon_distance = float(polygon.distance(nav_point))
        boundary_within_epsilon = (
            not centroid_inside
            and epsilon > 0.0
            and polygon_distance <= epsilon
        )

        overlap_area = 0.0
        overlap_ratio = 0.0
        if nav_footprint is not None and footprint_area > 0.0:
            overlap_area = float(nav_footprint.intersection(polygon).area)
            overlap_ratio = overlap_area / footprint_area

        if centroid_inside:
            reason = "centroid"
        elif boundary_within_epsilon:
            reason = "boundary"
        else:
            return None

        region_centroid = prepared_region.region_msg.centroid
        centroid_distance = float(
            math.hypot(
                nav_x - float(region_centroid.x),
                nav_y - float(region_centroid.y),
            )
        )
        return NavigationRegionCandidate(
            tracker_region_id=int(tracker_region_id),
            reason=reason,
            centroid_inside=centroid_inside,
            boundary_within_epsilon=boundary_within_epsilon,
            overlap_ratio=float(overlap_ratio),
            overlap_area=float(overlap_area),
            polygon_distance=float(polygon_distance),
            centroid_distance=centroid_distance,
            previous_membership_match=(
                previous_tracker_region_id is not None
                and int(previous_tracker_region_id) == int(tracker_region_id)
            ),
        )

    def _build_point_region_candidate(
        self,
        *,
        node_point: Point,
        node_x: float,
        node_y: float,
        tracker_region_id: int,
        prepared_region: PreparedTrackerRegion,
        previous_tracker_region_id: Optional[int],
    ) -> Optional[PointRegionCandidate]:
        """Return one plausible REGION candidate for a point-like node."""
        epsilon = float(self.nav_region_boundary_epsilon_m)
        region_min_x, region_min_y, region_max_x, region_max_y = prepared_region.bounds
        expanded_bounds = (
            region_min_x - epsilon,
            region_min_y - epsilon,
            region_max_x + epsilon,
            region_max_y + epsilon,
        )
        if not self._point_in_bounds((node_x, node_y), expanded_bounds):
            return None

        polygon = prepared_region.prepared_geometry.polygon
        if polygon is None or polygon.is_empty:
            return None

        point_inside = point_in_polygon((node_x, node_y), polygon)
        polygon_distance = float(polygon.distance(node_point))
        boundary_within_epsilon = (
            not point_inside and epsilon > 0.0 and polygon_distance <= epsilon
        )
        if not point_inside and not boundary_within_epsilon:
            return None

        region_centroid = prepared_region.region_msg.centroid
        centroid_distance = float(
            math.hypot(
                node_x - float(region_centroid.x),
                node_y - float(region_centroid.y),
            )
        )
        return PointRegionCandidate(
            tracker_region_id=int(tracker_region_id),
            reason="point" if point_inside else "boundary",
            point_inside=point_inside,
            boundary_within_epsilon=boundary_within_epsilon,
            polygon_distance=polygon_distance,
            centroid_distance=centroid_distance,
            previous_membership_match=(
                previous_tracker_region_id is not None
                and int(previous_tracker_region_id) == int(tracker_region_id)
            ),
        )

    def _count_navigation_neighbor_support(
        self,
        nav_node_id: int,
        tracker_region_id: int,
    ) -> int:
        """Count neighboring NAVIGATION nodes already assigned to the candidate region."""
        neighbor_ids = {
            int(edge.target_id)
            for edge in self.sg.query.get_outgoing_edges(
                int(nav_node_id),
                EdgeType.NAVIGABLE_PATH,
            )
        }
        neighbor_ids.update(
            int(edge.source_id)
            for edge in self.sg.query.get_incoming_edges(
                int(nav_node_id),
                EdgeType.NAVIGABLE_PATH,
            )
        )
        return int(
            sum(
                1
                for neighbor_id in neighbor_ids
                if self._node_tracker_region_membership.get(int(neighbor_id))
                == int(tracker_region_id)
            )
        )

    def _extract_navigation_bounds(
        self,
        node,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Extract one NAVIGATION footprint bounds tuple from node attributes."""
        attrs = dict(getattr(node, "attributes", None) or {})
        raw_bounds = attrs.get("bounds")
        if not isinstance(raw_bounds, dict):
            return None

        try:
            min_x = float(raw_bounds["min_x"])
            max_x = float(raw_bounds["max_x"])
            min_y = float(raw_bounds["min_y"])
            max_y = float(raw_bounds["max_y"])
        except (KeyError, TypeError, ValueError):
            return None

        if min_x > max_x:
            min_x, max_x = max_x, min_x
        if min_y > max_y:
            min_y, max_y = max_y, min_y
        if max_x - min_x <= 0.0 or max_y - min_y <= 0.0:
            return None
        return (min_x, min_y, max_x, max_y)

    def _get_previous_tracker_region_id_for_node(
        self,
        node_id: int,
    ) -> Optional[int]:
        """Return the previously assigned tracker-region id for one node."""
        previous_tracker_region_id = self._node_tracker_region_membership.get(
            int(node_id)
        )
        if previous_tracker_region_id is None:
            return None
        try:
            return int(previous_tracker_region_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _point_in_bounds(
        point_xy: Tuple[float, float],
        bounds: Tuple[float, float, float, float],
    ) -> bool:
        """Return True when one XY point lies inside one axis-aligned bounds tuple."""
        x, y = point_xy
        min_x, min_y, max_x, max_y = bounds
        return min_x <= x <= max_x and min_y <= y <= max_y

    @staticmethod
    def _bounds_overlap(
        bounds_a: Tuple[float, float, float, float],
        bounds_b: Tuple[float, float, float, float],
    ) -> bool:
        """Return True when two axis-aligned bounds overlap."""
        min_ax, min_ay, max_ax, max_ay = bounds_a
        min_bx, min_by, max_bx, max_by = bounds_b
        return not (
            max_ax < min_bx
            or max_bx < min_ax
            or max_ay < min_by
            or max_by < min_ay
        )

    def _find_tracker_region_containing_point(
        self,
        x: float,
        y: float,
        prepared_regions: Dict[int, PreparedTrackerRegion],
    ) -> Optional[int]:
        best_candidate = None
        for tracker_region_id, prepared_region in prepared_regions.items():
            min_x, min_y, max_x, max_y = prepared_region.bounds
            if x < min_x or x > max_x or y < min_y or y > max_y:
                continue
            if not point_in_polygon((float(x), float(y)), prepared_region.prepared_geometry.polygon):
                continue
            tracker_region_id = int(tracker_region_id)
            if best_candidate is None or tracker_region_id < best_candidate:
                best_candidate = tracker_region_id

        return best_candidate
