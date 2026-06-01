from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EdgeType(Enum):
    """Enumerates the different types of edges in the Scene Graph."""

    ROOM_CONTAINS = "ROOM_CONTAINS"  # e.g., RoomA contains ObjectX / PoseY / NavZ
    ROOM_ADJACENCY = "ROOM_ADJACENCY"  # e.g., RoomA is adjacent to RoomB (can be bidirectional)
    REGION_CONTAINS = "REGION_CONTAINS"  # legacy compatibility edge for region-layer runtimes
    NAVIGABLE_PATH = "NAVIGABLE_PATH"  # e.g., Navigational waypoint A ↔ B
    NEAREST_FREE_SPACE = "NEAREST_FREE_SPACE"  # e.g., Nearest free space to ObjectX
    TEMPORAL_LINK = "TEMPORAL_LINK"  # e.g., Motion node at t1 → Motion node at t2
    OBSERVATION_ANCHOR = (
        "OBSERVATION_ANCHOR"  # e.g., RobotPose observed ObjectX at time t
    )
    CUSTOM = "CUSTOM"  # for user‐defined relations

    @classmethod
    def from_string(cls, s: str) -> "EdgeType":
        """
        Convert a string to an EdgeType enum.
        Raises ValueError if the string does not match any EdgeType.
        """
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Invalid edge type: '{s}'. Must be one of {list(cls)}.")


@dataclass
class Edge:
    """
    A single directed edge in the Scene Graph.
    - source_id:     ID of the source node
    - target_id:     ID of the target node
    - type:          EdgeType enum
    - weight:        Optional numeric weight (for path planning, etc.)
    - is_structural: Whether this edge participates in the semantic hierarchy
                     (True = hierarchical/structural, False = relational/non-structural)
    - attributes:    Any extra metadata (e.g. timestamp, cost, sensor info)

    Structural edges (is_structural=True):
        - Define hierarchical parent-child relationships (e.g., ROOM_CONTAINS)
        - Must not create cycles
        - Each node should have at most one structural parent

    Relational edges (is_structural=False):
        - Define non-hierarchical relationships (e.g., NAVIGABLE_PATH, OBSERVATION_ANCHOR)
        - Cycles are allowed
        - Do not participate in the semantic hierarchy
    """

    source_id: int
    target_id: int
    id: Optional[int] = None  # Optional ID for the edge, can be set later
    type: EdgeType = EdgeType.CUSTOM  # Default to CUSTOM if not specified
    weight: float = 1.0
    is_structural: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        """
        Reconstruct an Edge from a dictionary (e.g., loaded from JSON).
        The incoming dictionary should have at least:
        - "source_id": ID of the source node
        - "target_id": ID of the target node
        - "type": Type of the edge (as a string or EdgeType enum)

        Note: Supports legacy "is_tree_edge" for backward compatibility.
        """
        # Support both new "is_structural" and legacy "is_tree_edge"
        is_structural = data.get("is_structural", data.get("is_tree_edge", True))

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            id=data.get("id"),
            type=EdgeType.from_string(data["type"])
            if isinstance(data["type"], str)
            else data["type"],
            weight=data.get("weight", 1.0),
            is_structural=is_structural,
            attributes=data.get("attributes", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.name,
            "weight": self.weight,
            "is_structural": self.is_structural,
            "attributes": self.attributes,
        }
