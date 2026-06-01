"""Room graph helpers used by the scene-graph runtimes."""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, Optional, Set

from scene_graph_core.graph_interface import SceneGraphInterface
from scene_graph_core.representation import Edge, EdgeType, NodeType, RoomNode
from scene_graph_core.services import GraphPatch
from scene_graph_ros.managers.semantic_signature_utils import (
    SemanticObjectTuple,
    build_object_signature_set_from_object_ids,
    serialize_object_signature_set,
)
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


class RoomManager:
    """Create rooms, anchor them to runtime regions, and build room signatures."""

    DIRECT_MEMBER_TYPES = (
        NodeType.OBJECT,
        NodeType.AGENT,
        NodeType.NAVIGATION,
    )

    def __init__(
        self,
        sg_interface: SceneGraphInterface,
        logger,
        z_offset: float = 12.0,
        semantic_tuple_round_decimals: int = 1,
    ):
        self.sg = sg_interface
        self.logger = logger
        self.z_offset = float(z_offset)
        self.semantic_tuple_round_decimals = int(semantic_tuple_round_decimals)

        self.room_to_stable_region_id: Dict[int, int] = {}
        self.stable_region_to_room: Dict[int, int] = {}
        self.room_to_tracker_region_id: Dict[int, int] = {}
        self.room_object_ids: Dict[int, set[int]] = {}
        self.room_region_centroids: Dict[int, list[tuple[float, float]]] = {}
        self.room_signature_sets: Dict[int, Set[SemanticObjectTuple]] = {}
        self.room_to_direct_members: Dict[int, Dict[NodeType, set[int]]] = {}
        self.direct_member_to_room: Dict[int, int] = {}
        self.room_direct_signature_sets: Dict[int, Set[SemanticObjectTuple]] = {}
        self.room_adjacency_pairs: set[tuple[int, int]] = set()
        self.dirty_room_ids: set[int] = set()

    @classmethod
    def is_direct_room_member_type(cls, node_type: Optional[NodeType]) -> bool:
        """Return whether one node type supports direct room ownership."""
        return node_type in cls.DIRECT_MEMBER_TYPES

    def _empty_direct_member_map(self) -> Dict[NodeType, set[int]]:
        """Return an empty direct-membership map for one room."""
        return {node_type: set() for node_type in self.DIRECT_MEMBER_TYPES}

    def next_room_name(self) -> str:
        """Return the next canonical room label."""
        room_count = len(self.sg.query.find_nodes_by_type(NodeType.ROOM))
        return f"room_{int(room_count)}"

    def create_room_from_pose(
        self,
        pose_node_id: int,
        *,
        name: Optional[str] = None,
    ) -> Optional[int]:
        """Create one room centered on the current pose."""
        pose_node = self.sg.query.get_node(int(pose_node_id))
        if pose_node is None or pose_node.node_type != NodeType.AGENT:
            self.logger.warning(
                f"[RoomManager] Cannot create room from missing pose {pose_node_id}"
            )
            return None

        room_node = RoomNode()
        room_node.pose.position.x = float(pose_node.pose.position.x)
        room_node.pose.position.y = float(pose_node.pose.position.y)
        room_node.pose.position.z = float(self.z_offset)
        room_node.pose.orientation.w = 1.0
        room_node.attributes = {
            "name": str(name or self.next_room_name()),
            "created_at_pose": int(pose_node_id),
        }

        room_id = int(self.sg.update.add_node(room_node))
        self.room_object_ids.setdefault(room_id, set())
        self.room_region_centroids.setdefault(room_id, [])
        self.room_signature_sets.setdefault(room_id, set())
        self.room_to_direct_members.setdefault(room_id, self._empty_direct_member_map())
        self.room_direct_signature_sets.setdefault(room_id, set())
        self.dirty_room_ids.add(room_id)
        self.logger.info(
            "[RoomManager] Created semantic room "
            f"room_id={room_id} name={room_node.attributes['name']} "
            f"from_pose={pose_node_id}"
        )
        return room_id

    def create_room_from_region(
        self,
        prepared_region,
        *,
        name: Optional[str] = None,
        reason: str = "region_observed_entities",
    ) -> Optional[int]:
        """Create one room centered on a prepared runtime region."""
        if prepared_region is None:
            self.logger.warning(
                "[RoomManager] Cannot create room from missing prepared region"
            )
            return None

        centroid = prepared_region.region_msg.centroid
        tracker_region_id = int(prepared_region.tracker_region_id)

        room_node = RoomNode()
        room_node.pose.position.x = float(centroid.x)
        room_node.pose.position.y = float(centroid.y)
        room_node.pose.position.z = float(self.z_offset)
        room_node.pose.orientation.w = 1.0
        room_node.attributes = {
            "name": str(name or self.next_room_name()),
            "created_from_tracker_region_id": tracker_region_id,
            "created_reason": str(reason),
        }

        room_id = int(self.sg.update.add_node(room_node))
        self.room_object_ids.setdefault(room_id, set())
        self.room_region_centroids.setdefault(room_id, [])
        self.room_signature_sets.setdefault(room_id, set())
        self.room_to_direct_members.setdefault(room_id, self._empty_direct_member_map())
        self.room_direct_signature_sets.setdefault(room_id, set())
        self.dirty_room_ids.add(room_id)
        self.logger.info(
            "[RoomManager] Created semantic room from observed region "
            f"room_id={room_id} name={room_node.attributes['name']} "
            f"tracker_region_id={tracker_region_id} reason={reason}"
        )
        return room_id

    def create_initial_room(self, pose_node_id: int) -> Optional[int]:
        """Create the one bootstrap room as ``room_0``."""
        if self.sg.query.find_nodes_by_type(NodeType.ROOM):
            self.logger.warning(
                "[RoomManager] Refusing bootstrap room creation after rooms already exist"
            )
            return None
        return self.create_room_from_pose(int(pose_node_id), name="room_0")

    def remove_room(self, room_node_id: int, *, reason: str = "remove_room") -> bool:
        """Remove one ROOM node and clear its room-anchor/member caches."""
        room_id = int(room_node_id)
        room_node = self.sg.query.get_node(room_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            return False

        stable_region_id = self.get_tracker_region_id_for_room(room_id)
        self._remove_room_adjacency_edges(room_id)
        self.sg.update.remove_node(room_id)
        if stable_region_id is not None:
            self.stable_region_to_room.pop(int(stable_region_id), None)
        for cached_region_id, cached_room_id in list(self.stable_region_to_room.items()):
            if int(cached_room_id) == room_id:
                self.stable_region_to_room.pop(int(cached_region_id), None)
        self.room_to_stable_region_id.pop(room_id, None)
        self.room_to_tracker_region_id.pop(room_id, None)
        self.room_object_ids.pop(room_id, None)
        self.room_region_centroids.pop(room_id, None)
        self.room_signature_sets.pop(room_id, None)
        direct_members = self.room_to_direct_members.pop(room_id, None)
        if direct_members is not None:
            for member_ids in direct_members.values():
                for member_id in member_ids:
                    if self.direct_member_to_room.get(int(member_id)) == room_id:
                        self.direct_member_to_room.pop(int(member_id), None)
        for member_id, owner_room_id in list(self.direct_member_to_room.items()):
            if int(owner_room_id) == room_id:
                self.direct_member_to_room.pop(int(member_id), None)
        self.room_direct_signature_sets.pop(room_id, None)
        self.dirty_room_ids.discard(room_id)
        room_name = (
            room_node.attributes.get("name")
            if isinstance(room_node.attributes, dict)
            else None
        )
        self.logger.info(
            "[RoomManager] Removed room "
            f"room_id={room_id} name={room_name} reason={reason}"
        )
        return True

    def prune_direct_rooms_without_object_support(
        self,
        *,
        min_object_support: int = 1,
        bootstrap_room_name: str = "room_0",
    ) -> list[int]:
        """Prune non-bootstrap direct rooms with insufficient OBJECT support."""
        pruned_room_ids: list[int] = []
        min_support = max(0, int(min_object_support))
        for room_node in list(self.sg.query.find_nodes_by_type(NodeType.ROOM)):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            attrs = room_node.attributes if isinstance(room_node.attributes, dict) else {}
            if attrs.get("name") == bootstrap_room_name:
                continue
            object_ids = self.get_attached_direct_member_ids(
                room_id,
                node_types=(NodeType.OBJECT,),
            ).get(NodeType.OBJECT, set())
            if len(object_ids) >= min_support:
                continue
            if self.remove_room(
                room_id,
                reason=(
                    "non_bootstrap_room_without_object_support "
                    f"object_count={len(object_ids)} min_object_support={min_support}"
                ),
            ):
                pruned_room_ids.append(room_id)
        return pruned_room_ids

    def associate_room_with_tracker_region(
        self,
        room_node_id: int,
        tracker_region_id: int,
        prepared_region,
        *,
        is_bootstrap_region: bool = False,
    ) -> bool:
        """Make one room the persistent semantic anchor for a runtime region."""
        room_id = int(room_node_id)
        tracker_region_id = int(tracker_region_id)
        stable_region_id = tracker_region_id
        room_node = self.sg.query.get_node(room_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            self.logger.warning(
                f"[RoomManager] Cannot anchor missing room {room_node_id}"
            )
            return False

        previous_stable_region_id = self.room_to_stable_region_id.get(room_id)
        if (
            previous_stable_region_id is not None
            and int(previous_stable_region_id) != stable_region_id
            and self.stable_region_to_room.get(int(previous_stable_region_id)) == room_id
        ):
            self.stable_region_to_room.pop(int(previous_stable_region_id), None)

        previous_room_id = self.stable_region_to_room.get(stable_region_id)
        if previous_room_id is not None and int(previous_room_id) != room_id:
            self.room_to_stable_region_id.pop(int(previous_room_id), None)
            self.room_to_tracker_region_id.pop(int(previous_room_id), None)
            self.dirty_room_ids.add(int(previous_room_id))

        self.room_to_stable_region_id[room_id] = stable_region_id
        self.room_to_tracker_region_id[room_id] = tracker_region_id
        self.stable_region_to_room[stable_region_id] = room_id

        attrs = dict(room_node.attributes or {})
        attrs.setdefault("name", self.next_room_name())
        attrs["stable_region_id"] = stable_region_id
        attrs["tracker_region_id"] = tracker_region_id
        attrs["is_bootstrap_region"] = bool(
            attrs.get("is_bootstrap_region", False) or is_bootstrap_region
        )
        attrs.update(self._build_region_geometry_attributes(prepared_region))

        centroid = prepared_region.region_msg.centroid
        new_x = float(centroid.x)
        new_y = float(centroid.y)
        new_z = float(self.z_offset)
        pose_changed = (
            abs(float(room_node.pose.position.x) - new_x) > 1e-9
            or abs(float(room_node.pose.position.y) - new_y) > 1e-9
            or abs(float(room_node.pose.position.z) - new_z) > 1e-9
        )
        attrs_changed = attrs != dict(room_node.attributes or {})
        if pose_changed:
            room_node.pose.position.x = new_x
            room_node.pose.position.y = new_y
            room_node.pose.position.z = new_z
            room_node.pose.orientation.w = 1.0
        if attrs_changed:
            room_node.attributes = attrs
        if pose_changed or attrs_changed:
            self.sg.update.update_node(room_id, room_node)

        self.room_region_centroids[room_id] = [(new_x, new_y)]
        self.dirty_room_ids.add(room_id)
        return bool(pose_changed or attrs_changed)

    def _build_region_geometry_attributes(self, prepared_region) -> dict:
        """Serialize runtime region geometry onto a ROOM node."""
        polygon_points = [
            {"x": float(x), "y": float(y)}
            for x, y in prepared_region.signature[0]
        ]
        convex_hull_points = [
            {"x": float(x), "y": float(y)}
            for x, y in prepared_region.signature[1]
        ]
        min_x, min_y, max_x, max_y = prepared_region.bounds
        centroid = prepared_region.region_msg.centroid
        geometry_signature = {
            "polygon": [[float(x), float(y)] for x, y in prepared_region.signature[0]],
            "convex_hull": [
                [float(x), float(y)] for x, y in prepared_region.signature[1]
            ],
        }
        return {
            "polygon": polygon_points,
            "convex_hull": convex_hull_points,
            "centroid": {"x": float(centroid.x), "y": float(centroid.y)},
            "bounds": {
                "min_x": float(min_x),
                "min_y": float(min_y),
                "max_x": float(max_x),
                "max_y": float(max_y),
            },
            "geometry_signature": geometry_signature,
            "geometry_refresh_token": repr(geometry_signature),
        }

    def refresh_room_footprints_from_direct_navigation(
        self,
        room_ids: Optional[Iterable[int]] = None,
    ) -> int:
        """Mirror owned NAVIGATION block footprints onto ROOM nodes."""
        updated_count = 0
        target_room_ids = (
            {int(room_id) for room_id in room_ids if room_id is not None}
            if room_ids is not None
            else {
                int(room_node.id)
                for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM)
                if room_node.id is not None
            }
        )

        for room_id in sorted(target_room_ids):
            room_node = self.sg.query.get_node(int(room_id))
            if room_node is None or room_node.node_type != NodeType.ROOM:
                continue

            nav_node_ids = self.get_attached_direct_member_ids(
                room_id,
                node_types=(NodeType.NAVIGATION,),
            ).get(NodeType.NAVIGATION, set())

            attrs = dict(room_node.attributes or {})
            geometry_attrs = self._build_navigation_footprint_attributes(
                nav_node_ids
            )
            if geometry_attrs is None:
                geometry_changed = self._clear_direct_navigation_geometry(attrs)
            else:
                geometry_changed = False
                for key, value in geometry_attrs.items():
                    if attrs.get(key) != value:
                        attrs[key] = value
                        geometry_changed = True

            if not geometry_changed:
                continue

            room_node.attributes = attrs
            self.sg.update.update_node(room_id, room_node)
            self.dirty_room_ids.add(room_id)
            updated_count += 1

        return updated_count

    def _build_navigation_footprint_attributes(
        self,
        nav_node_ids: Iterable[int],
    ) -> Optional[dict]:
        """Serialize a room footprint from its directly owned NAVIGATION nodes."""
        nav_boxes = []
        source_nav_node_ids: list[int] = []
        for raw_nav_id in sorted({int(nav_id) for nav_id in nav_node_ids}):
            nav_node = self.sg.query.get_node(int(raw_nav_id))
            if nav_node is None or nav_node.node_type != NodeType.NAVIGATION:
                continue
            bounds = self._extract_navigation_bounds(nav_node)
            if bounds is None:
                continue
            min_x, min_y, max_x, max_y = bounds
            nav_boxes.append(box(min_x, min_y, max_x, max_y))
            source_nav_node_ids.append(int(raw_nav_id))

        if not nav_boxes:
            return None

        footprint = unary_union(nav_boxes)
        if footprint.is_empty:
            return None
        if footprint.geom_type != "Polygon":
            footprint = footprint.convex_hull
        if not footprint.is_valid:
            footprint = footprint.buffer(0)
        if footprint.is_empty or footprint.area <= 0.0:
            return None

        polygon_points = self._polygon_exterior_points(footprint)
        convex_hull_points = self._polygon_exterior_points(footprint.convex_hull)
        if len(polygon_points) < 3 or len(convex_hull_points) < 3:
            return None

        min_x, min_y, max_x, max_y = footprint.bounds
        centroid = footprint.centroid
        geometry_signature = {
            "polygon": [[float(x), float(y)] for x, y in polygon_points],
            "convex_hull": [[float(x), float(y)] for x, y in convex_hull_points],
            "source_nav_node_ids": list(source_nav_node_ids),
        }
        return {
            "polygon": [
                {"x": float(x), "y": float(y)}
                for x, y in polygon_points
            ],
            "convex_hull": [
                {"x": float(x), "y": float(y)}
                for x, y in convex_hull_points
            ],
            "centroid": {"x": float(centroid.x), "y": float(centroid.y)},
            "bounds": {
                "min_x": float(min_x),
                "min_y": float(min_y),
                "max_x": float(max_x),
                "max_y": float(max_y),
            },
            "geometry_source": "direct_navigation",
            "footprint_nav_node_ids": list(source_nav_node_ids),
            "geometry_signature": geometry_signature,
            "geometry_refresh_token": repr(geometry_signature),
        }

    def _extract_navigation_bounds(
        self,
        nav_node,
    ) -> Optional[tuple[float, float, float, float]]:
        """Extract NAVIGATION footprint bounds as min-x/min-y/max-x/max-y."""
        attrs = dict(getattr(nav_node, "attributes", None) or {})
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

    @staticmethod
    def _polygon_exterior_points(polygon: Polygon) -> list[tuple[float, float]]:
        """Return an open exterior ring suitable for room geometry attributes."""
        if polygon.is_empty or polygon.geom_type != "Polygon":
            return []
        points = [(float(x), float(y)) for x, y in polygon.exterior.coords]
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        return points

    @staticmethod
    def _clear_direct_navigation_geometry(attrs: dict) -> bool:
        """Remove stale direct-navigation footprint geometry from room attributes."""
        changed = False
        for key in (
            "polygon",
            "convex_hull",
            "centroid",
            "bounds",
            "geometry_source",
            "footprint_nav_node_ids",
            "geometry_signature",
            "geometry_refresh_token",
        ):
            if key in attrs:
                attrs.pop(key, None)
                changed = True
        return changed

    def get_room_id_for_tracker_region(
        self,
        tracker_region_id: Optional[int],
    ) -> Optional[int]:
        """Return the room anchored to one tracker/stable region."""
        if tracker_region_id is None:
            return None
        tracker_region_id = int(tracker_region_id)
        room_id = self.stable_region_to_room.get(tracker_region_id)
        if room_id is not None:
            return int(room_id)

        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            attrs = dict(room_node.attributes or {})
            stable_region_id = attrs.get(
                "stable_region_id",
                attrs.get("tracker_region_id"),
            )
            try:
                stable_region_id = int(stable_region_id)
            except (TypeError, ValueError):
                continue
            if stable_region_id != tracker_region_id:
                continue
            room_id = int(room_node.id)
            self.room_to_stable_region_id[room_id] = stable_region_id
            self.room_to_tracker_region_id[room_id] = tracker_region_id
            self.stable_region_to_room[stable_region_id] = room_id
            return room_id
        return None

    def get_tracker_region_id_for_room(self, room_node_id: int) -> Optional[int]:
        """Return the stable/tracker-region anchor for one room."""
        room_id = int(room_node_id)
        tracker_region_id = self.room_to_tracker_region_id.get(room_id)
        if tracker_region_id is not None:
            return int(tracker_region_id)
        room_node = self.sg.query.get_node(room_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            return None
        attrs = dict(room_node.attributes or {})
        tracker_region_id = attrs.get("tracker_region_id", attrs.get("stable_region_id"))
        try:
            tracker_region_id = int(tracker_region_id)
        except (TypeError, ValueError):
            return None
        self.room_to_tracker_region_id[room_id] = tracker_region_id
        self.room_to_stable_region_id[room_id] = tracker_region_id
        self.stable_region_to_room.setdefault(tracker_region_id, room_id)
        return tracker_region_id

    def sync_room_membership_from_region(
        self,
        room_node_id: int,
        member_ids: Dict[NodeType, set[int]],
    ) -> Dict[str, int]:
        """Refresh direct ROOM_CONTAINS members from geometry-filtered region members."""
        room_id = int(room_node_id)
        keep_ids = set().union(*member_ids.values()) if member_ids else set()
        stats = {"assigned": 0, "removed": 0}

        patch = GraphPatch()
        current_direct_members = self.get_attached_direct_member_ids(room_id)
        current_keep_ids = set().union(*current_direct_members.values())
        for node_id in current_keep_ids - keep_ids:
            patch.remove_edge(room_id, int(node_id), EdgeType.ROOM_CONTAINS)
            for members in self.room_to_direct_members.setdefault(
                room_id,
                self._empty_direct_member_map(),
            ).values():
                members.discard(int(node_id))
            if self.direct_member_to_room.get(int(node_id)) == room_id:
                self.direct_member_to_room.pop(int(node_id), None)
            stats["removed"] += 1

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)

        for node_type in self.DIRECT_MEMBER_TYPES:
            for node_id in sorted(member_ids.get(node_type, set())):
                if self.attach_direct_member_to_room(
                    room_id,
                    int(node_id),
                    allow_reassignment=True,
                    reason="region_membership_sync",
                ):
                    stats["assigned"] += 1

        if stats["assigned"] or stats["removed"]:
            self.room_object_ids.pop(room_id, None)
            self.room_signature_sets.pop(room_id, None)
            self.room_direct_signature_sets.pop(room_id, None)
            self.dirty_room_ids.add(room_id)
        return stats

    def _refresh_direct_members_for_room(
        self,
        room_node_id: int,
        *,
        node_types: Optional[Iterable[NodeType]] = None,
    ) -> Dict[NodeType, set[int]]:
        """Refresh cached direct ROOM_CONTAINS membership for one room."""
        room_id = int(room_node_id)
        tracked_types = (
            tuple(node_types)
            if node_types is not None
            else self.DIRECT_MEMBER_TYPES
        )
        direct_members = self.room_to_direct_members.setdefault(
            room_id,
            self._empty_direct_member_map(),
        )
        for node_type in tracked_types:
            direct_members.setdefault(node_type, set()).clear()

        for edge in self.sg.query.get_outgoing_edges(room_id, EdgeType.ROOM_CONTAINS):
            target_node = self.sg.query.get_node(int(edge.target_id))
            if target_node is None or not self.is_direct_room_member_type(
                target_node.node_type
            ):
                continue
            if target_node.node_type not in tracked_types:
                continue
            direct_members[target_node.node_type].add(int(edge.target_id))
            self.direct_member_to_room[int(edge.target_id)] = room_id

        return {
            node_type: set(direct_members.get(node_type, set()))
            for node_type in tracked_types
        }

    def _detach_direct_member_from_rooms(self, node_id: int) -> set[int]:
        """Remove any direct ROOM_CONTAINS ownership edges into one member node."""
        detached_room_ids: set[int] = set()
        node_id = int(node_id)
        patch = GraphPatch()
        for edge in self.sg.query.get_incoming_edges(node_id, EdgeType.ROOM_CONTAINS):
            room_id = int(edge.source_id)
            room_node = self.sg.query.get_node(room_id)
            if room_node is None or room_node.node_type != NodeType.ROOM:
                continue
            patch.remove_edge(room_id, node_id, EdgeType.ROOM_CONTAINS)
            direct_members = self.room_to_direct_members.setdefault(
                room_id,
                self._empty_direct_member_map(),
            )
            for member_ids in direct_members.values():
                member_ids.discard(node_id)
            self.room_object_ids.pop(room_id, None)
            self.room_direct_signature_sets.pop(room_id, None)
            self.dirty_room_ids.add(room_id)
            detached_room_ids.add(room_id)

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)

        self.direct_member_to_room.pop(node_id, None)
        return detached_room_ids

    def attach_direct_member_to_room(
        self,
        room_node_id: int,
        node_id: int,
        *,
        allow_reassignment: bool = False,
        reason: str = "direct_attach",
    ) -> bool:
        """Attach one OBJECT/AGENT/NAVIGATION node directly to exactly one room."""
        room_node_id = int(room_node_id)
        node_id = int(node_id)

        room_node = self.sg.query.get_node(room_node_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            self.logger.warning(
                f"[RoomManager] Cannot attach direct member to missing room {room_node_id}"
            )
            return False

        member_node = self.sg.query.get_node(node_id)
        if member_node is None or not self.is_direct_room_member_type(
            member_node.node_type
        ):
            self.logger.warning(
                "[RoomManager] Cannot directly attach non-direct member "
                f"node_id={node_id}"
            )
            return False

        current_room_id = self.get_room_id_for_direct_member(node_id)
        if current_room_id == room_node_id and self.sg.query.has_edge(
            room_node_id, node_id, EdgeType.ROOM_CONTAINS
        ):
            return False

        if current_room_id is not None and int(current_room_id) != room_node_id:
            if not allow_reassignment:
                self.logger.info(
                    "[RoomManager] Skipping direct member reassignment "
                    f"node_id={node_id} node_type={member_node.node_type.value} "
                    f"existing_room_id={int(current_room_id)} "
                    f"requested_room_id={room_node_id} reason={reason}"
                )
                return False
            self.logger.info(
                "[RoomManager] Reassigning direct room member "
                f"node_id={node_id} node_type={member_node.node_type.value} "
                f"from_room={int(current_room_id)} "
                f"to_room={room_node_id} reason={reason}"
            )

        detached_room_ids = self._detach_direct_member_from_rooms(node_id)
        if not self.sg.query.has_edge(room_node_id, node_id, EdgeType.ROOM_CONTAINS):
            self.sg.update.add_edge(
                Edge(
                    source_id=room_node_id,
                    target_id=node_id,
                    type=EdgeType.ROOM_CONTAINS,
                )
            )

        direct_members = self.room_to_direct_members.setdefault(
            room_node_id,
            self._empty_direct_member_map(),
        )
        direct_members.setdefault(member_node.node_type, set()).add(node_id)
        self.direct_member_to_room[node_id] = room_node_id
        self.room_object_ids.pop(room_node_id, None)
        self.room_direct_signature_sets.pop(room_node_id, None)
        self.dirty_room_ids.add(room_node_id)
        self.dirty_room_ids.update(detached_room_ids)
        return True

    def attach_direct_members_to_room(
        self,
        room_node_id: int,
        node_ids: Iterable[int],
        *,
        allow_reassignment: bool = False,
        reason: str = "bulk_direct_attach",
    ) -> int:
        """Attach many direct members to one room, returning the change count."""
        changed_count = 0
        for raw_node_id in node_ids:
            try:
                node_id = int(raw_node_id)
            except (TypeError, ValueError):
                continue
            if self.attach_direct_member_to_room(
                int(room_node_id),
                node_id,
                allow_reassignment=allow_reassignment,
                reason=reason,
            ):
                changed_count += 1
        return changed_count

    def get_room_id_for_direct_member(self, node_id: int) -> Optional[int]:
        """Return the unique owning room for one direct member node, if any."""
        node_id = int(node_id)
        cached_room_id = self.direct_member_to_room.get(node_id)
        owner_room_ids: list[int] = []
        for edge in self.sg.query.get_incoming_edges(node_id, EdgeType.ROOM_CONTAINS):
            source_node = self.sg.query.get_node(int(edge.source_id))
            if source_node is None or source_node.node_type != NodeType.ROOM:
                continue
            owner_room_ids.append(int(edge.source_id))

        if not owner_room_ids:
            self.direct_member_to_room.pop(node_id, None)
            return None

        owner_room_ids = sorted(set(owner_room_ids))
        preferred_room_id = (
            int(cached_room_id)
            if cached_room_id is not None and int(cached_room_id) in owner_room_ids
            else int(owner_room_ids[0])
        )

        duplicate_room_ids = [
            int(room_id) for room_id in owner_room_ids if int(room_id) != preferred_room_id
        ]
        if duplicate_room_ids:
            patch = GraphPatch()
            for duplicate_room_id in duplicate_room_ids:
                patch.remove_edge(
                    int(duplicate_room_id),
                    node_id,
                    EdgeType.ROOM_CONTAINS,
                )
                for members in self.room_to_direct_members.setdefault(
                    int(duplicate_room_id),
                    self._empty_direct_member_map(),
                ).values():
                    members.discard(node_id)
                self.dirty_room_ids.add(int(duplicate_room_id))
            self.sg.update.apply_patch(patch, validate=False)
            self.logger.warning(
                "[RoomManager] Removed duplicate direct ownership "
                f"node_id={node_id} kept_room_id={preferred_room_id} "
                f"removed_room_ids={duplicate_room_ids}"
            )

        self.direct_member_to_room[node_id] = preferred_room_id
        self._refresh_direct_members_for_room(preferred_room_id)
        return preferred_room_id

    def get_attached_direct_member_ids(
        self,
        room_node_id: int,
        *,
        node_types: Iterable[NodeType] = DIRECT_MEMBER_TYPES,
    ) -> Dict[NodeType, set[int]]:
        """Return current direct ROOM_CONTAINS membership for one room."""
        return self._refresh_direct_members_for_room(
            int(room_node_id),
            node_types=tuple(node_types),
        )

    def relink_rooms_to_replacement_regions(
        self,
        prepared_regions: Dict[int, object],
        *,
        min_old_overlap_ratio: float = 0.65,
        min_member_support_ratio: float = 0.5,
        min_score_margin: float = 0.15,
    ) -> list[dict[str, object]]:
        """
        Relink live rooms whose stored region id disappeared.

        The relink is intentionally conservative: it only applies to rooms that
        still have direct support, whose previous room polygon substantially
        overlaps one current region, and whose direct members also agree with
        that candidate. In the region pipeline this is the only chance for a
        stale room to survive before invalid anchors are pruned.
        """
        if not prepared_regions:
            return []

        valid_region_ids = {int(region_id) for region_id in prepared_regions}
        claimed_region_ids = self._collect_valid_claimed_region_ids(valid_region_ids)
        candidate_records: list[dict[str, object]] = []

        for room_node in sorted(
            self.sg.query.find_nodes_by_type(NodeType.ROOM),
            key=lambda node: int(node.id) if node.id is not None else -1,
        ):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            old_tracker_region_id = self.get_tracker_region_id_for_room(room_id)
            if old_tracker_region_id is None:
                continue
            old_tracker_region_id = int(old_tracker_region_id)
            if old_tracker_region_id in valid_region_ids:
                continue

            direct_member_ids = self._collect_all_direct_member_ids(room_id)
            if not direct_member_ids:
                continue

            candidate = self._select_replacement_region_for_room(
                room_node,
                direct_member_ids,
                prepared_regions,
                claimed_region_ids=claimed_region_ids,
                min_old_overlap_ratio=min_old_overlap_ratio,
                min_member_support_ratio=min_member_support_ratio,
                min_score_margin=min_score_margin,
            )
            if candidate is None:
                continue

            candidate["room_node_id"] = room_id
            candidate["old_tracker_region_id"] = old_tracker_region_id
            candidate_records.append(candidate)

        relinked_rooms: list[dict[str, object]] = []
        used_room_ids: set[int] = set()
        used_region_ids: set[int] = set()
        for candidate in sorted(
            candidate_records,
            key=lambda item: (
                -float(item["score"]),
                -float(item["old_overlap_ratio"]),
                int(item["room_node_id"]),
            ),
        ):
            room_id = int(candidate["room_node_id"])
            new_region_id = int(candidate["new_tracker_region_id"])
            if room_id in used_room_ids:
                continue
            if new_region_id in claimed_region_ids or new_region_id in used_region_ids:
                continue

            prepared_region = prepared_regions.get(new_region_id)
            if prepared_region is None:
                continue

            self.associate_room_with_tracker_region(
                room_id,
                new_region_id,
                prepared_region,
            )
            claimed_region_ids.add(new_region_id)
            used_region_ids.add(new_region_id)
            used_room_ids.add(room_id)
            relinked_rooms.append(
                {
                    "room_node_id": room_id,
                    "old_tracker_region_id": int(candidate["old_tracker_region_id"]),
                    "new_tracker_region_id": new_region_id,
                    "score": float(candidate["score"]),
                    "old_overlap_ratio": float(candidate["old_overlap_ratio"]),
                    "new_overlap_ratio": float(candidate["new_overlap_ratio"]),
                    "iou": float(candidate["iou"]),
                    "member_support_ratio": float(
                        candidate["member_support_ratio"]
                    ),
                    "centroid_inside": bool(candidate["centroid_inside"]),
                }
            )

            self.logger.info(
                "[RoomManager] Relinked room to replacement region "
                f"room_id={room_id} old_tracker_region_id="
                f"{candidate['old_tracker_region_id']} "
                f"new_tracker_region_id={new_region_id} "
                f"old_overlap={float(candidate['old_overlap_ratio']):.3f} "
                f"member_support={float(candidate['member_support_ratio']):.3f} "
                f"iou={float(candidate['iou']):.3f}"
            )

        return relinked_rooms

    def _collect_valid_claimed_region_ids(self, valid_region_ids: set[int]) -> set[int]:
        """Return current region ids already anchored by non-stale rooms."""
        claimed_region_ids: set[int] = set()
        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            tracker_region_id = self.get_tracker_region_id_for_room(int(room_node.id))
            if tracker_region_id is None:
                continue
            tracker_region_id = int(tracker_region_id)
            if tracker_region_id in valid_region_ids:
                claimed_region_ids.add(tracker_region_id)
        return claimed_region_ids

    def _collect_all_direct_member_ids(self, room_node_id: int) -> set[int]:
        """Return all direct member ids currently attached to one room."""
        direct_members = self.get_attached_direct_member_ids(int(room_node_id))
        member_ids: set[int] = set()
        for ids_for_type in direct_members.values():
            member_ids.update(int(node_id) for node_id in ids_for_type)
        return member_ids

    def _select_replacement_region_for_room(
        self,
        room_node,
        direct_member_ids: set[int],
        prepared_regions: Dict[int, object],
        *,
        claimed_region_ids: set[int],
        min_old_overlap_ratio: float,
        min_member_support_ratio: float,
        min_score_margin: float,
    ) -> Optional[dict[str, object]]:
        """Return the best confident replacement-region candidate for one room."""
        room_polygon = self._extract_room_polygon(room_node)
        if room_polygon is None or room_polygon.area <= 0.0:
            return None

        room_centroid = self._extract_room_centroid(room_node, room_polygon)
        candidates: list[dict[str, object]] = []
        for tracker_region_id, prepared_region in prepared_regions.items():
            tracker_region_id = int(tracker_region_id)
            if tracker_region_id in claimed_region_ids:
                continue
            candidate = self._build_replacement_region_candidate(
                tracker_region_id,
                prepared_region,
                room_polygon,
                room_centroid,
                direct_member_ids,
                min_old_overlap_ratio=min_old_overlap_ratio,
                min_member_support_ratio=min_member_support_ratio,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item["score"]),
                -float(item["old_overlap_ratio"]),
                -float(item["member_support_ratio"]),
                int(item["new_tracker_region_id"]),
            ),
        )
        if (
            len(ranked) > 1
            and float(ranked[0]["score"]) - float(ranked[1]["score"])
            < float(min_score_margin)
        ):
            self.logger.debug(
                "[RoomManager] Refusing ambiguous replacement-region relink "
                f"room_id={room_node.id} "
                f"best_region={ranked[0]['new_tracker_region_id']} "
                f"second_region={ranked[1]['new_tracker_region_id']} "
                f"best_score={float(ranked[0]['score']):.3f} "
                f"second_score={float(ranked[1]['score']):.3f}"
            )
            return None
        return ranked[0]

    def _build_replacement_region_candidate(
        self,
        tracker_region_id: int,
        prepared_region,
        room_polygon: Polygon,
        room_centroid: Point,
        direct_member_ids: set[int],
        *,
        min_old_overlap_ratio: float,
        min_member_support_ratio: float,
    ) -> Optional[dict[str, object]]:
        """Score one current region as a replacement for one stale room anchor."""
        candidate_polygon = getattr(
            getattr(prepared_region, "prepared_geometry", None),
            "polygon",
            None,
        )
        if candidate_polygon is None or candidate_polygon.is_empty:
            return None
        if not self._bounds_overlap_tuple(room_polygon.bounds, candidate_polygon.bounds):
            return None

        intersection_area = float(room_polygon.intersection(candidate_polygon).area)
        if intersection_area <= 0.0:
            return None

        room_area = float(room_polygon.area)
        candidate_area = float(candidate_polygon.area)
        if room_area <= 0.0 or candidate_area <= 0.0:
            return None

        old_overlap_ratio = intersection_area / room_area
        if old_overlap_ratio < float(min_old_overlap_ratio):
            return None

        member_support_ratio = self._compute_member_support_ratio(
            direct_member_ids,
            candidate_polygon,
        )
        if member_support_ratio < float(min_member_support_ratio):
            return None

        centroid_inside = bool(candidate_polygon.covers(room_centroid))
        if not centroid_inside and member_support_ratio < 0.75:
            return None

        union_area = float(room_polygon.union(candidate_polygon).area)
        iou = intersection_area / union_area if union_area > 0.0 else 0.0
        new_overlap_ratio = intersection_area / candidate_area
        score = (
            0.45 * old_overlap_ratio
            + 0.35 * member_support_ratio
            + 0.10 * iou
            + 0.10 * (1.0 if centroid_inside else 0.0)
        )
        return {
            "new_tracker_region_id": int(tracker_region_id),
            "score": float(score),
            "old_overlap_ratio": float(old_overlap_ratio),
            "new_overlap_ratio": float(new_overlap_ratio),
            "iou": float(iou),
            "member_support_ratio": float(member_support_ratio),
            "centroid_inside": centroid_inside,
        }

    def _compute_member_support_ratio(
        self,
        direct_member_ids: set[int],
        candidate_polygon,
    ) -> float:
        """Return fraction of direct room members covered by a candidate region."""
        if not direct_member_ids:
            return 0.0

        supported_count = 0
        valid_count = 0
        for node_id in sorted(direct_member_ids):
            member_node = self.sg.query.get_node(int(node_id))
            if member_node is None or not self.is_direct_room_member_type(
                member_node.node_type
            ):
                continue
            valid_count += 1
            member_point = Point(
                float(member_node.pose.position.x),
                float(member_node.pose.position.y),
            )
            if candidate_polygon.covers(member_point):
                supported_count += 1

        if valid_count == 0:
            return 0.0
        return float(supported_count) / float(valid_count)

    def _extract_room_polygon(self, room_node) -> Optional[Polygon]:
        """Return the mirrored support-region polygon stored on a room."""
        attrs = dict(getattr(room_node, "attributes", None) or {})
        raw_polygon = attrs.get("polygon")
        points: list[tuple[float, float]] = []

        if isinstance(raw_polygon, list):
            for raw_point in raw_polygon:
                try:
                    if isinstance(raw_point, dict):
                        point_xy = (float(raw_point["x"]), float(raw_point["y"]))
                    else:
                        point_xy = (float(raw_point[0]), float(raw_point[1]))
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                points.append(point_xy)

        if len(points) < 3:
            geometry_signature = attrs.get("geometry_signature")
            signature_polygon = (
                geometry_signature.get("polygon")
                if isinstance(geometry_signature, dict)
                else None
            )
            if isinstance(signature_polygon, list):
                points = []
                for raw_point in signature_polygon:
                    try:
                        points.append((float(raw_point[0]), float(raw_point[1])))
                    except (IndexError, TypeError, ValueError):
                        continue

        if len(points) < 3:
            return None
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0.0:
            return None
        return polygon

    def _extract_room_centroid(self, room_node, room_polygon: Polygon) -> Point:
        """Return the stored room support centroid, falling back to polygon centroid."""
        attrs = dict(getattr(room_node, "attributes", None) or {})
        centroid = attrs.get("centroid")
        if isinstance(centroid, dict):
            try:
                return Point(float(centroid["x"]), float(centroid["y"]))
            except (KeyError, TypeError, ValueError):
                pass
        return room_polygon.centroid

    @staticmethod
    def _bounds_overlap_tuple(
        bounds_a: tuple[float, float, float, float],
        bounds_b: tuple[float, float, float, float],
    ) -> bool:
        """Return True when two Shapely bounds tuples overlap."""
        min_ax, min_ay, max_ax, max_ay = bounds_a
        min_bx, min_by, max_bx, max_by = bounds_b
        return not (
            max_ax < min_bx
            or max_bx < min_ax
            or max_ay < min_by
            or max_by < min_ay
        )

    def prune_rooms_without_valid_anchors(
        self,
        valid_stable_region_ids: Iterable[int],
    ) -> list[dict[str, Optional[str]]]:
        """
        Prune every ROOM whose stable/tracker-region anchor is not current.

        In the region pipeline, ROOM validity is defined by the existence of its
        current DuDe stable-region anchor. Direct members are derived
        attachments and never protect a room whose stable region disappeared.
        """
        pruned_rooms: list[dict[str, Optional[str]]] = []
        valid_ids = {int(region_id) for region_id in valid_stable_region_ids}

        for room_node in list(self.sg.query.find_nodes_by_type(NodeType.ROOM)):
            if room_node.id is None:
                continue

            room_id = int(room_node.id)
            stable_region_id = self.get_tracker_region_id_for_room(room_id)
            if stable_region_id is not None and int(stable_region_id) in valid_ids:
                continue

            room_name = None
            if isinstance(room_node.attributes, dict):
                room_name = room_node.attributes.get("name")

            pruned_room = {
                "room_node_id": room_id,
                "name": str(room_name) if room_name is not None else None,
                "stable_region_id": (
                    str(stable_region_id)
                    if stable_region_id is not None
                    else None
                ),
            }
            if self.remove_room(
                room_id,
                reason=(
                    "invalid_stable_region_anchor "
                    f"stable_region_id={stable_region_id}"
                ),
            ):
                self.logger.info(
                    "[RoomManager] Pruned room with invalid anchor "
                    f"room_id={room_id} name={room_name} "
                    f"stable_region_id={stable_region_id}"
                )
                pruned_rooms.append(pruned_room)

        return pruned_rooms

    def _collect_room_object_ids(self, room_node_id: int) -> Set[int]:
        """Collect unique OBJECT ids directly owned by one room."""
        room_id = int(room_node_id)
        object_ids = self._collect_room_direct_object_ids(room_id)
        self.room_object_ids[room_id] = object_ids
        return object_ids

    def _collect_room_direct_object_ids(
        self,
        room_node_id: int,
        *,
        exclude_object_ids: Iterable[int] = (),
    ) -> Set[int]:
        """Collect unique directly owned OBJECT ids for one room."""
        room_id = int(room_node_id)
        excluded_ids = {
            int(object_id)
            for object_id in exclude_object_ids
            if object_id is not None
        }
        direct_member_ids = self.get_attached_direct_member_ids(
            room_id,
            node_types=(NodeType.OBJECT,),
        ).get(NodeType.OBJECT, set())
        object_ids = {
            int(object_id)
            for object_id in direct_member_ids
            if int(object_id) not in excluded_ids
        }

        if not excluded_ids:
            self.room_object_ids[room_id] = set(object_ids)
        return object_ids

    def build_room_region_signature_set(
        self,
        room_node_id: int,
        *,
        persist: bool = False,
    ) -> Set[SemanticObjectTuple]:
        """Build one room signature from geometry-gated direct memberships."""
        room_id = int(room_node_id)
        room_node = self.sg.query.get_node(room_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            self.logger.warning(
                f"[RoomManager] Cannot build signature set for missing room {room_node_id}"
            )
            return set()

        if room_id in self.room_signature_sets and room_id not in self.dirty_room_ids:
            signature_set = set(self.room_signature_sets[room_id])
        else:
            signature_set = build_object_signature_set_from_object_ids(
                self.sg,
                self._collect_room_object_ids(room_id),
                round_decimals=self.semantic_tuple_round_decimals,
            )
            self.room_signature_sets[room_id] = set(signature_set)

        if not persist:
            return signature_set

        serialized_signature_set = serialize_object_signature_set(signature_set)
        room_attrs = dict(room_node.attributes or {})
        if (
            room_attrs.get("signature_set") != serialized_signature_set
            or "class_set" in room_attrs
            or "class_histogram" in room_attrs
        ):
            room_attrs["signature_set"] = serialized_signature_set
            room_attrs.pop("class_set", None)
            room_attrs.pop("class_histogram", None)
            room_node.attributes = room_attrs
            self.sg.update.update_node(room_id, room_node)

        return signature_set

    def build_room_direct_signature_set(
        self,
        room_node_id: int,
        *,
        persist: bool = False,
        exclude_object_ids: Iterable[int] = (),
    ) -> Set[SemanticObjectTuple]:
        """Build one room signature from directly owned OBJECT nodes."""
        room_id = int(room_node_id)
        room_node = self.sg.query.get_node(room_id)
        if room_node is None or room_node.node_type != NodeType.ROOM:
            self.logger.warning(
                f"[RoomManager] Cannot build direct signature set for missing room {room_node_id}"
            )
            return set()

        signature_set = build_object_signature_set_from_object_ids(
            self.sg,
            self._collect_room_direct_object_ids(
                room_id,
                exclude_object_ids=exclude_object_ids,
            ),
            round_decimals=self.semantic_tuple_round_decimals,
        )
        if not exclude_object_ids:
            self.room_direct_signature_sets[room_id] = set(signature_set)

        if not persist:
            return signature_set

        serialized_signature_set = serialize_object_signature_set(signature_set)
        room_attrs = dict(room_node.attributes or {})
        if room_attrs.get("signature_set") != serialized_signature_set:
            room_attrs["signature_set"] = serialized_signature_set
            room_node.attributes = room_attrs
            self.sg.update.update_node(room_id, room_node)

        return signature_set

    def build_all_room_region_signature_sets(
        self,
        *,
        persist: bool = False,
    ) -> Dict[int, Set[SemanticObjectTuple]]:
        """Build room signatures for every semantic room."""
        signature_sets: Dict[int, Set[SemanticObjectTuple]] = {}
        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            signature_sets[room_id] = self.build_room_region_signature_set(
                room_id,
                persist=persist,
            )
        return signature_sets

    def build_all_room_direct_signature_sets(
        self,
        *,
        persist: bool = False,
        exclude_object_ids_by_room: Optional[Dict[int, Iterable[int]]] = None,
    ) -> Dict[int, Set[SemanticObjectTuple]]:
        """Build direct room signatures for every room."""
        signature_sets: Dict[int, Set[SemanticObjectTuple]] = {}
        exclude_object_ids_by_room = exclude_object_ids_by_room or {}
        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            signature_sets[room_id] = self.build_room_direct_signature_set(
                room_id,
                persist=persist,
                exclude_object_ids=exclude_object_ids_by_room.get(room_id, ()),
            )
        return signature_sets

    def recompute_room_centroids_from_anchors(
        self,
        room_ids: Optional[Iterable[int]] = None,
    ) -> int:
        """Refresh room poses from their mirrored support-region centroid."""
        updated_count = 0
        target_room_ids = (
            {int(room_id) for room_id in room_ids}
            if room_ids is not None
            else set(self.dirty_room_ids) or set(self.room_to_stable_region_id)
        )

        for room_id in target_room_ids:
            room_node = self.sg.query.get_node(int(room_id))
            if room_node is None or room_node.node_type != NodeType.ROOM:
                continue

            attrs = dict(room_node.attributes or {})
            centroid = attrs.get("centroid")
            if not isinstance(centroid, dict):
                continue

            try:
                new_x = float(centroid["x"])
                new_y = float(centroid["y"])
            except (KeyError, TypeError, ValueError):
                continue
            self.room_region_centroids[int(room_id)] = [(new_x, new_y)]
            old_x = float(room_node.pose.position.x)
            old_y = float(room_node.pose.position.y)

            if abs(old_x - new_x) <= 1e-9 and abs(old_y - new_y) <= 1e-9:
                continue

            room_node.pose.position.x = float(new_x)
            room_node.pose.position.y = float(new_y)
            self.sg.update.update_node(int(room_id), room_node)
            updated_count += 1

            self.logger.info(
                "[RoomManager] Updated room anchor centroid "
                f"room_id={room_id} stable_region_id={attrs.get('stable_region_id')} "
                f"old=({old_x:.3f}, {old_y:.3f}) "
                f"new=({new_x:.3f}, {new_y:.3f})"
            )

        return updated_count

    def recompute_room_centroids_from_direct_members(
        self,
        room_ids: Optional[Iterable[int]] = None,
    ) -> int:
        """Refresh derived room centroids from directly attached entities."""
        updated_count = 0
        target_room_ids = (
            {int(room_id) for room_id in room_ids}
            if room_ids is not None
            else set(self.dirty_room_ids) or set(self.room_to_direct_members)
        )

        for room_id in target_room_ids:
            room_node = self.sg.query.get_node(int(room_id))
            if room_node is None or room_node.node_type != NodeType.ROOM:
                continue

            direct_members = self.get_attached_direct_member_ids(int(room_id))
            member_points: list[tuple[float, float]] = []
            for node_type in self.DIRECT_MEMBER_TYPES:
                for node_id in sorted(direct_members.get(node_type, set())):
                    member_node = self.sg.query.get_node(int(node_id))
                    if member_node is None or member_node.node_type != node_type:
                        continue
                    member_points.append(
                        (
                            float(member_node.pose.position.x),
                            float(member_node.pose.position.y),
                        )
                    )

            if not member_points:
                continue

            new_x = sum(point[0] for point in member_points) / len(member_points)
            new_y = sum(point[1] for point in member_points) / len(member_points)
            old_x = float(room_node.pose.position.x)
            old_y = float(room_node.pose.position.y)
            if abs(old_x - new_x) <= 1e-9 and abs(old_y - new_y) <= 1e-9:
                continue

            room_node.pose.position.x = float(new_x)
            room_node.pose.position.y = float(new_y)
            self.sg.update.update_node(int(room_id), room_node)
            updated_count += 1

            self.logger.info(
                "[RoomManager] Updated direct room centroid "
                f"room_id={room_id} attached_entities={len(member_points)} "
                f"old=({old_x:.3f}, {old_y:.3f}) "
                f"new=({new_x:.3f}, {new_y:.3f})"
            )

        return updated_count

    def rebuild_room_adjacency(
        self,
        prepared_regions: Optional[dict[int, object]] = None,
        *,
        dirty_room_ids: Optional[Iterable[int]] = None,
        region_nav_ids_by_tracker_region: Optional[Dict[int, set[int]]] = None,
    ) -> Dict[str, object]:
        """Rebuild the undirected ROOM_ADJACENCY relation."""
        if prepared_regions is None:
            return self.rebuild_room_adjacency_from_direct_navigation()

        return self.rebuild_room_adjacency_from_dude_regions_and_navigation(
            prepared_regions,
            dirty_room_ids=dirty_room_ids,
            region_nav_ids_by_tracker_region=region_nav_ids_by_tracker_region,
        )

    def _build_dude_region_adjacency_pairs(
        self,
        prepared_regions: Dict[int, object],
    ) -> set[tuple[int, int]]:
        """Return current undirected DuDe stable-region adjacency pairs."""
        current_region_ids = {int(region_id) for region_id in prepared_regions}
        dude_region_pairs: set[tuple[int, int]] = set()
        for region_id, prepared_region in prepared_regions.items():
            source_region_id = int(region_id)
            adjacent_ids = getattr(
                getattr(prepared_region, "region_msg", None),
                "adjacent_ids",
                [],
            )
            for adjacent_id in adjacent_ids:
                try:
                    target_region_id = int(adjacent_id)
                except (TypeError, ValueError):
                    continue
                if target_region_id == source_region_id:
                    continue
                if target_region_id not in current_region_ids:
                    continue
                dude_region_pairs.add(
                    tuple(sorted((source_region_id, target_region_id)))
                )
        return dude_region_pairs

    def _build_current_region_to_room_map(
        self,
        prepared_regions: Dict[int, object],
    ) -> Dict[int, int]:
        """Return valid current tracker-region ids mapped to ROOM node ids."""
        region_to_room: Dict[int, int] = {}
        for region_id in prepared_regions:
            room_id = self.get_room_id_for_tracker_region(int(region_id))
            if room_id is None:
                continue
            room_node = self.sg.query.get_node(int(room_id))
            if room_node is None or room_node.node_type != NodeType.ROOM:
                continue
            region_to_room[int(region_id)] = int(room_id)
        return region_to_room

    def _sanitize_region_navigation_membership(
        self,
        region_nav_ids_by_tracker_region: Optional[Dict[int, set[int]]],
        prepared_regions: Dict[int, object],
    ) -> tuple[Dict[int, set[int]], list[int], list[int]]:
        """Return valid per-region NAVIGATION members, excluding ambiguous ids."""
        current_region_ids = {int(region_id) for region_id in prepared_regions}
        raw_region_nav_ids: Dict[int, set[int]] = {}
        nav_region_counts: Dict[int, int] = {}
        missing_nav_node_ids: set[int] = set()

        for raw_region_id, raw_nav_ids in (
            region_nav_ids_by_tracker_region or {}
        ).items():
            try:
                region_id = int(raw_region_id)
            except (TypeError, ValueError):
                continue
            if region_id not in current_region_ids:
                continue
            nav_ids = raw_region_nav_ids.setdefault(region_id, set())
            for raw_nav_id in raw_nav_ids or set():
                try:
                    nav_id = int(raw_nav_id)
                except (TypeError, ValueError):
                    continue
                nav_node = self.sg.query.get_node(nav_id)
                if nav_node is None or nav_node.node_type != NodeType.NAVIGATION:
                    missing_nav_node_ids.add(nav_id)
                    continue
                nav_ids.add(nav_id)

        for nav_ids in raw_region_nav_ids.values():
            for nav_id in nav_ids:
                nav_region_counts[nav_id] = int(nav_region_counts.get(nav_id, 0)) + 1

        ambiguous_nav_node_ids = sorted(
            nav_id for nav_id, count in nav_region_counts.items() if count > 1
        )
        ambiguous_nav_node_id_set = set(ambiguous_nav_node_ids)
        region_nav_ids = {
            int(region_id): {
                int(nav_id)
                for nav_id in nav_ids
                if int(nav_id) not in ambiguous_nav_node_id_set
            }
            for region_id, nav_ids in raw_region_nav_ids.items()
        }

        return region_nav_ids, sorted(missing_nav_node_ids), ambiguous_nav_node_ids

    def _apply_room_adjacency_pairs(
        self,
        adjacent_room_pairs: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """Apply a full replacement set of undirected ROOM_ADJACENCY pairs."""
        patch = GraphPatch()
        previous_pairs = set(self.room_adjacency_pairs)
        removed_edges = 0
        added_edges = 0
        for room_a_id, room_b_id in previous_pairs - adjacent_room_pairs:
            for source_id, target_id in (
                (int(room_a_id), int(room_b_id)),
                (int(room_b_id), int(room_a_id)),
            ):
                patch.remove_edge(source_id, target_id, EdgeType.ROOM_ADJACENCY)
                removed_edges += 1

        for room_a_id, room_b_id in adjacent_room_pairs - previous_pairs:
            for source_id, target_id in (
                (int(room_a_id), int(room_b_id)),
                (int(room_b_id), int(room_a_id)),
            ):
                patch.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=target_id,
                        type=EdgeType.ROOM_ADJACENCY,
                        is_structural=False,
                    ),
                    is_structural=False,
                )
                added_edges += 1

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)
        self.room_adjacency_pairs = adjacent_room_pairs

        return int(added_edges), int(removed_edges)

    def rebuild_room_adjacency_from_dude_regions_and_navigation(
        self,
        prepared_regions: Dict[int, object],
        *,
        dirty_room_ids: Optional[Iterable[int]] = None,
        region_nav_ids_by_tracker_region: Optional[Dict[int, set[int]]] = None,
    ) -> Dict[str, object]:
        """Rebuild ROOM_ADJACENCY as DuDe adjacency intersected with NAV bridges."""
        dude_region_pairs = self._build_dude_region_adjacency_pairs(prepared_regions)
        region_to_room = self._build_current_region_to_room_map(prepared_regions)
        (
            region_nav_ids,
            missing_nav_node_ids,
            ambiguous_nav_node_ids,
        ) = self._sanitize_region_navigation_membership(
            region_nav_ids_by_tracker_region,
            prepared_regions,
        )
        nav_to_region: Dict[int, int] = {}
        for region_id, nav_ids in region_nav_ids.items():
            for nav_id in nav_ids:
                nav_to_region[int(nav_id)] = int(region_id)

        adjacent_room_pairs: set[tuple[int, int]] = set()
        cross_region_nav_bridges: list[tuple[int, int, int, int]] = []
        candidate_cross_region_nav_bridges: list[tuple[int, int, int, int]] = []
        rejected_missing_room_pairs: set[tuple[int, int]] = set()
        rejected_missing_dude_adjacency_pairs: set[tuple[int, int]] = set()
        skipped_nav_edges: list[tuple[int, int, Optional[int], Optional[int]]] = []

        for edge in self.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH):
            source_node = self.sg.query.get_node(int(edge.source_id))
            target_node = self.sg.query.get_node(int(edge.target_id))
            if (
                source_node is None
                or target_node is None
                or source_node.node_type != NodeType.NAVIGATION
                or target_node.node_type != NodeType.NAVIGATION
            ):
                continue

            source_region_id = nav_to_region.get(int(edge.source_id))
            target_region_id = nav_to_region.get(int(edge.target_id))
            if source_region_id is None or target_region_id is None:
                skipped_nav_edges.append(
                    (
                        int(edge.source_id),
                        int(edge.target_id),
                        int(source_region_id) if source_region_id is not None else None,
                        int(target_region_id) if target_region_id is not None else None,
                    )
                )
                continue
            if source_region_id == target_region_id:
                continue

            region_pair = tuple(sorted((int(source_region_id), int(target_region_id))))
            bridge = (
                int(edge.source_id),
                int(edge.target_id),
                int(source_region_id),
                int(target_region_id),
            )
            candidate_cross_region_nav_bridges.append(bridge)

            if region_pair not in dude_region_pairs:
                rejected_missing_dude_adjacency_pairs.add(region_pair)
                continue

            source_room_id = region_to_room.get(int(source_region_id))
            target_room_id = region_to_room.get(int(target_region_id))
            if source_room_id is None or target_room_id is None:
                rejected_missing_room_pairs.add(region_pair)
                continue
            if source_room_id == target_room_id:
                continue

            cross_region_nav_bridges.append(bridge)
            adjacent_room_pairs.add(
                tuple(sorted((int(source_room_id), int(target_room_id))))
            )

        rejected_missing_nav_bridge_pairs = set(dude_region_pairs)
        for bridge in candidate_cross_region_nav_bridges:
            rejected_missing_nav_bridge_pairs.discard(
                tuple(sorted((int(bridge[2]), int(bridge[3]))))
            )
        for region_pair in rejected_missing_room_pairs:
            rejected_missing_nav_bridge_pairs.discard(region_pair)

        added_edges, removed_edges = self._apply_room_adjacency_pairs(
            adjacent_room_pairs
        )

        debug_summary = (
            "[RoomManager] region-aware room adjacency "
            f"dude_region_pairs={sorted(dude_region_pairs)} "
            f"region_to_room={dict(sorted(region_to_room.items()))} "
            f"candidate_cross_region_nav_bridges={sorted(candidate_cross_region_nav_bridges)} "
            f"rejected_missing_room_pairs={sorted(rejected_missing_room_pairs)} "
            f"rejected_missing_nav_bridge_pairs={sorted(rejected_missing_nav_bridge_pairs)} "
            f"rejected_missing_dude_adjacency_pairs={sorted(rejected_missing_dude_adjacency_pairs)} "
            f"final_room_pairs={sorted(adjacent_room_pairs)} "
            f"added_edges={added_edges} removed_edges={removed_edges}"
        )
        self.logger.debug(debug_summary)

        return {
            "dude_region_pairs": sorted(dude_region_pairs),
            "region_to_room": dict(sorted(region_to_room.items())),
            "region_nav_ids_by_tracker_region": {
                int(region_id): sorted(nav_ids)
                for region_id, nav_ids in sorted(region_nav_ids.items())
            },
            "candidate_cross_region_nav_bridges": sorted(
                candidate_cross_region_nav_bridges
            ),
            "cross_region_nav_bridges": sorted(cross_region_nav_bridges),
            "navigation_room_pairs": sorted(adjacent_room_pairs),
            "adjacent_room_pairs": sorted(adjacent_room_pairs),
            "missing_nav_node_ids": sorted(missing_nav_node_ids),
            "ambiguous_nav_node_ids": sorted(ambiguous_nav_node_ids),
            "skipped_nav_edges": sorted(skipped_nav_edges),
            "rejected_missing_room_pairs": sorted(rejected_missing_room_pairs),
            "rejected_missing_nav_bridge_pairs": sorted(
                rejected_missing_nav_bridge_pairs
            ),
            "rejected_missing_dude_adjacency_pairs": sorted(
                rejected_missing_dude_adjacency_pairs
            ),
            "dirty_room_ids": sorted(int(room_id) for room_id in dirty_room_ids or []),
            "added_edges": int(added_edges),
            "removed_edges": int(removed_edges),
        }

    def _build_direct_nav_adjacency(self) -> Dict[int, set[int]]:
        """Return the undirected NAVIGATION connectivity graph."""
        adjacency: Dict[int, set[int]] = {}
        for nav_node in self.sg.query.find_nodes_by_type(NodeType.NAVIGATION):
            if nav_node.id is None:
                continue
            adjacency.setdefault(int(nav_node.id), set())

        for edge in self.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH):
            source_node = self.sg.query.get_node(int(edge.source_id))
            target_node = self.sg.query.get_node(int(edge.target_id))
            if (
                source_node is None
                or target_node is None
                or source_node.node_type != NodeType.NAVIGATION
                or target_node.node_type != NodeType.NAVIGATION
            ):
                continue
            adjacency.setdefault(int(edge.source_id), set()).add(int(edge.target_id))
            adjacency.setdefault(int(edge.target_id), set()).add(int(edge.source_id))

        return adjacency

    def _seed_navigation_nodes_from_direct_objects(self) -> Dict[int, set[int]]:
        """Return room-owned seed NAVIGATION nodes derived from direct OBJECT ownership."""
        room_seed_nav_ids: Dict[int, set[int]] = {}
        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            object_ids = self.get_attached_direct_member_ids(
                room_id,
                node_types=(NodeType.OBJECT,),
            ).get(NodeType.OBJECT, set())
            seed_nav_ids: set[int] = set()
            for object_id in object_ids:
                for edge in self.sg.query.get_outgoing_edges(
                    int(object_id), EdgeType.NEAREST_FREE_SPACE
                ):
                    target_node = self.sg.query.get_node(int(edge.target_id))
                    if target_node is None or target_node.node_type != NodeType.NAVIGATION:
                        continue
                    seed_nav_ids.add(int(edge.target_id))
            if seed_nav_ids:
                room_seed_nav_ids[room_id] = seed_nav_ids
        return room_seed_nav_ids

    def _nearest_navigation_node_id_for_member(
        self,
        node_id: int,
        *,
        max_distance_m: Optional[float] = None,
    ) -> Optional[int]:
        """Return the nearest NAVIGATION node to a direct member pose."""
        member_node = self.sg.query.get_node(int(node_id))
        if member_node is None:
            return None

        member_x = float(member_node.pose.position.x)
        member_y = float(member_node.pose.position.y)
        best_key: Optional[tuple[float, int]] = None
        for nav_node in self.sg.query.find_nodes_by_type(NodeType.NAVIGATION):
            if nav_node.id is None:
                continue
            dx = float(nav_node.pose.position.x) - member_x
            dy = float(nav_node.pose.position.y) - member_y
            distance = math.hypot(dx, dy)
            if max_distance_m is not None and distance > float(max_distance_m):
                continue
            candidate_key = (distance, int(nav_node.id))
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key

        return int(best_key[1]) if best_key is not None else None

    def _seed_navigation_nodes_from_direct_poses(
        self,
        *,
        max_distance_m: Optional[float] = None,
    ) -> Dict[int, set[int]]:
        """Return room-owned seed NAVIGATION nodes derived from direct AGENT poses."""
        room_seed_nav_ids: Dict[int, set[int]] = {}
        for room_node in self.sg.query.find_nodes_by_type(NodeType.ROOM):
            if room_node.id is None:
                continue
            room_id = int(room_node.id)
            pose_ids = self.get_attached_direct_member_ids(
                room_id,
                node_types=(NodeType.AGENT,),
            ).get(NodeType.AGENT, set())
            seed_nav_ids: set[int] = set()
            for pose_id in pose_ids:
                nearest_nav_id = self._nearest_navigation_node_id_for_member(
                    int(pose_id),
                    max_distance_m=max_distance_m,
                )
                if nearest_nav_id is not None:
                    seed_nav_ids.add(int(nearest_nav_id))
            if seed_nav_ids:
                room_seed_nav_ids[room_id] = seed_nav_ids
        return room_seed_nav_ids

    def repair_direct_navigation_room_ownership(
        self,
        *,
        max_bfs_depth: Optional[int] = 3,
        pose_seed_max_distance_m: Optional[float] = None,
    ) -> Dict[str, object]:
        """Conservatively assign NAVIGATION nodes from local room-owned seeds."""
        nav_adjacency = self._build_direct_nav_adjacency()
        object_seed_nav_ids = self._seed_navigation_nodes_from_direct_objects()
        pose_seed_nav_ids = self._seed_navigation_nodes_from_direct_poses(
            max_distance_m=pose_seed_max_distance_m,
        )
        room_seed_nav_ids: Dict[int, set[int]] = {}
        for room_id in set(object_seed_nav_ids).union(pose_seed_nav_ids):
            room_seed_nav_ids[int(room_id)] = set(object_seed_nav_ids.get(room_id, set()))
            room_seed_nav_ids[int(room_id)].update(pose_seed_nav_ids.get(room_id, set()))

        distance_by_room: Dict[int, Dict[int, int]] = {}
        for room_id, seed_nav_ids in sorted(room_seed_nav_ids.items()):
            room_distances: Dict[int, int] = {}
            queue = deque(sorted(seed_nav_ids))
            for nav_id in sorted(seed_nav_ids):
                room_distances[int(nav_id)] = 0

            while queue:
                nav_id = int(queue.popleft())
                next_depth = int(room_distances[nav_id]) + 1
                if max_bfs_depth is not None and next_depth > int(max_bfs_depth):
                    continue
                for neighbor_nav_id in sorted(nav_adjacency.get(nav_id, set())):
                    if neighbor_nav_id in room_distances:
                        continue
                    room_distances[int(neighbor_nav_id)] = next_depth
                    queue.append(int(neighbor_nav_id))

            distance_by_room[room_id] = room_distances

        nav_owner_map: Dict[int, int] = {}
        ambiguous_nav_node_ids: list[int] = []
        skipped_nav_node_ids: list[int] = []
        changed_nav_owners: list[dict[str, Optional[int]]] = []
        assigned_nav_count = 0
        removed_nav_count = 0

        for nav_node in self.sg.query.find_nodes_by_type(NodeType.NAVIGATION):
            if nav_node.id is None:
                continue
            nav_id = int(nav_node.id)
            current_owner_id = self.get_room_id_for_direct_member(nav_id)

            candidate_distances = [
                (int(room_id), int(room_distances[nav_id]))
                for room_id, room_distances in distance_by_room.items()
                if nav_id in room_distances
            ]
            if not candidate_distances:
                skipped_nav_node_ids.append(nav_id)
                continue

            min_depth = min(depth for _, depth in candidate_distances)
            tied_room_ids = sorted(
                room_id
                for room_id, depth in candidate_distances
                if depth == min_depth
            )
            if len(tied_room_ids) > 1:
                ambiguous_nav_node_ids.append(nav_id)
                if current_owner_id in tied_room_ids:
                    nav_owner_map[nav_id] = int(current_owner_id)
                else:
                    skipped_nav_node_ids.append(nav_id)
                continue

            selected_room_id = int(tied_room_ids[0])
            nav_owner_map[nav_id] = selected_room_id
            if current_owner_id == selected_room_id:
                continue

            allow_reassignment = False
            change_reason = "nav_repair_unowned_local_seed"
            if current_owner_id is not None:
                current_depth = distance_by_room.get(int(current_owner_id), {}).get(nav_id)
                allow_reassignment = (
                    (current_depth is not None and min_depth < int(current_depth))
                    or (current_depth is None and min_depth == 0)
                )
                change_reason = (
                    "nav_repair_stronger_local_evidence"
                    if allow_reassignment
                    else "nav_repair_sticky_existing_owner"
                )
                if not allow_reassignment:
                    skipped_nav_node_ids.append(nav_id)
                    continue

            if self.attach_direct_member_to_room(
                selected_room_id,
                nav_id,
                allow_reassignment=allow_reassignment,
                reason=change_reason,
            ):
                assigned_nav_count += 1
                changed_nav_owners.append(
                    {
                        "nav_id": int(nav_id),
                        "previous_room_id": (
                            int(current_owner_id)
                            if current_owner_id is not None
                            else None
                        ),
                        "new_room_id": int(selected_room_id),
                    }
                )

        self.logger.debug(
            "[RoomManager] direct navigation ownership repair "
            f"rooms_with_seeds={len(room_seed_nav_ids)} "
            f"nav_nodes={len(nav_adjacency)} assigned={assigned_nav_count} "
            f"ambiguous={len(ambiguous_nav_node_ids)} skipped={len(skipped_nav_node_ids)} "
            f"changed={changed_nav_owners[:10]}"
        )

        return {
            "object_seed_nav_ids_by_room": {
                int(room_id): sorted(seed_nav_ids)
                for room_id, seed_nav_ids in sorted(object_seed_nav_ids.items())
            },
            "pose_seed_nav_ids_by_room": {
                int(room_id): sorted(seed_nav_ids)
                for room_id, seed_nav_ids in sorted(pose_seed_nav_ids.items())
            },
            "seed_nav_ids_by_room": {
                int(room_id): sorted(seed_nav_ids)
                for room_id, seed_nav_ids in sorted(room_seed_nav_ids.items())
            },
            "distance_room_count": int(len(distance_by_room)),
            "nav_owner_map": {
                int(nav_id): int(room_id)
                for nav_id, room_id in sorted(nav_owner_map.items())
            },
            "ambiguous_nav_node_ids": sorted(ambiguous_nav_node_ids),
            "skipped_nav_node_ids": sorted(skipped_nav_node_ids),
            "changed_nav_owners": changed_nav_owners,
            "assigned_nav_count": int(assigned_nav_count),
            "removed_nav_count": int(removed_nav_count),
        }

    def _remove_room_adjacency_edges(self, room_node_id: int) -> int:
        """Remove all ROOM_ADJACENCY edges touching one room."""
        room_node_id = int(room_node_id)
        patch = GraphPatch()
        removed_edges = 0
        pairs_to_remove = {
            pair for pair in self.room_adjacency_pairs if room_node_id in pair
        }
        for room_a_id, room_b_id in pairs_to_remove:
            for source_id, target_id in (
                (int(room_a_id), int(room_b_id)),
                (int(room_b_id), int(room_a_id)),
            ):
                patch.remove_edge(source_id, target_id, EdgeType.ROOM_ADJACENCY)
                removed_edges += 1
        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)
        self.room_adjacency_pairs.difference_update(pairs_to_remove)
        return int(removed_edges)

    def rebuild_room_adjacency_from_direct_navigation(self) -> Dict[str, object]:
        """Rebuild ROOM_ADJACENCY from direct NAVIGATION ownership."""
        nav_to_room: dict[int, int] = {}
        ambiguous_nav_node_ids: list[int] = []
        for nav_node in self.sg.query.find_nodes_by_type(NodeType.NAVIGATION):
            if nav_node.id is None:
                continue
            nav_id = int(nav_node.id)
            owner_room_ids = {
                int(edge.source_id)
                for edge in self.sg.query.get_incoming_edges(nav_id, EdgeType.ROOM_CONTAINS)
                if (
                    (room_node := self.sg.query.get_node(int(edge.source_id))) is not None
                    and room_node.node_type == NodeType.ROOM
                )
            }
            if len(owner_room_ids) > 1:
                ambiguous_nav_node_ids.append(nav_id)
                continue
            if len(owner_room_ids) == 1:
                nav_to_room[nav_id] = next(iter(owner_room_ids))

        adjacent_room_pairs: set[tuple[int, int]] = set()
        unowned_nav_node_ids: set[int] = set()
        skipped_nav_edges: list[tuple[int, int, Optional[int], Optional[int]]] = []
        cross_room_nav_edges: list[tuple[int, int, int, int]] = []
        for edge in self.sg.query.get_all_edges(EdgeType.NAVIGABLE_PATH):
            source_room_id = nav_to_room.get(int(edge.source_id))
            target_room_id = nav_to_room.get(int(edge.target_id))
            if source_room_id is None or target_room_id is None:
                if source_room_id is None:
                    unowned_nav_node_ids.add(int(edge.source_id))
                if target_room_id is None:
                    unowned_nav_node_ids.add(int(edge.target_id))
                skipped_nav_edges.append(
                    (
                        int(edge.source_id),
                        int(edge.target_id),
                        int(source_room_id) if source_room_id is not None else None,
                        int(target_room_id) if target_room_id is not None else None,
                    )
                )
                continue

            if source_room_id == target_room_id:
                continue

            cross_room_nav_edges.append(
                (
                    int(edge.source_id),
                    int(edge.target_id),
                    int(source_room_id),
                    int(target_room_id),
                )
            )
            adjacent_room_pairs.add(
                tuple(sorted((int(source_room_id), int(target_room_id))))
            )

        patch = GraphPatch()
        previous_pairs = set(self.room_adjacency_pairs)
        removed_edges = 0
        added_edges = 0
        for room_a_id, room_b_id in previous_pairs - adjacent_room_pairs:
            for source_id, target_id in (
                (int(room_a_id), int(room_b_id)),
                (int(room_b_id), int(room_a_id)),
            ):
                patch.remove_edge(source_id, target_id, EdgeType.ROOM_ADJACENCY)
                removed_edges += 1

        for room_a_id, room_b_id in adjacent_room_pairs - previous_pairs:
            for source_id, target_id in (
                (int(room_a_id), int(room_b_id)),
                (int(room_b_id), int(room_a_id)),
            ):
                patch.add_edge(
                    Edge(
                        source_id=source_id,
                        target_id=target_id,
                        type=EdgeType.ROOM_ADJACENCY,
                        is_structural=False,
                    ),
                    is_structural=False,
                )
                added_edges += 1

        if not patch.is_empty():
            self.sg.update.apply_patch(patch, validate=False)
        self.room_adjacency_pairs = adjacent_room_pairs

        return {
            "navigation_room_pairs": sorted(adjacent_room_pairs),
            "adjacent_room_pairs": sorted(adjacent_room_pairs),
            "ambiguous_nav_node_ids": sorted(ambiguous_nav_node_ids),
            "unowned_nav_node_ids": sorted(unowned_nav_node_ids),
            "skipped_nav_edges": sorted(skipped_nav_edges),
            "cross_room_nav_edges": sorted(cross_room_nav_edges),
            "added_edges": int(added_edges),
            "removed_edges": int(removed_edges),
        }
