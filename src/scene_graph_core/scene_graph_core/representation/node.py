from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Dict, Optional

from .geometry import Pose


class NodeLayer(Enum):
    """Enumerates the different types (layers) of nodes in the Scene Graph."""

    GLOBAL = "GLOBAL"
    SEMANTIC = "SEMANTIC"
    NAVIGATION = "NAVIGATION"
    MOTION = "MOTION"
    GEOMETRY = "GEOMETRY"
    OBJECT = "OBJECT"

    @classmethod
    def from_string(cls, s: str) -> "NodeLayer":
        """
        Convert a string to a NodeLayer enum.
        Raises ValueError if the string does not match any NodeLayer.
        """
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Invalid node type: '{s}'. Must be one of {list(cls)}.")


class NodeType(Enum):
    AGENT = "AGENT"
    OBJECT = "OBJECT"
    NAVIGATION = "NAVIGATION"
    REGION = "REGION"
    ROOM = "ROOM"

    @classmethod
    def from_string(cls, s: str) -> "NodeType":
        """
        Convert a string to a NodeType enum.
        Raises ValueError if the string does not match any NodeType.
        """
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Invalid node type: '{s}'. Must be one of {list(cls)}.")


# ID range offsets for type-scoped ID system
# Must match _NODE_TYPE_ID_OFFSETS in graph.py
NODE_TYPE_ID_OFFSETS = {
    NodeType.AGENT: 0,
    NodeType.OBJECT: 1_000_000,
    NodeType.NAVIGATION: 2_000_000,
    NodeType.REGION: 3_000_000,
    NodeType.ROOM: 4_000_000,
}


def get_type_scoped_id(global_id: int, node_type: NodeType) -> int:
    """
    Convert a global ID to a type-scoped ID for display purposes.

    Global IDs use offsets to ensure uniqueness across types:
    - Room 4,000,000 -> displays as Room 0
    - Room 4,000,001 -> displays as Room 1
    - Object 2,000,000 -> displays as Object 0

    Args:
        global_id: The globally unique node ID
        node_type: Type of the node

    Returns:
        Type-scoped sequential ID (0, 1, 2, 3...)
    """
    offset = NODE_TYPE_ID_OFFSETS.get(node_type, 0)
    return global_id - offset


def get_global_id(type_scoped_id: int, node_type: NodeType) -> int:
    """
    Convert a type-scoped ID to a global ID.

    Args:
        type_scoped_id: Sequential ID within type (0, 1, 2, 3...)
        node_type: Type of the node

    Returns:
        Globally unique node ID
    """
    offset = NODE_TYPE_ID_OFFSETS.get(node_type, 0)
    return offset + type_scoped_id


def _is_pose_like(pose: Any) -> bool:
    return (
        hasattr(pose, "position")
        and hasattr(pose, "orientation")
        and all(hasattr(pose.position, field_name) for field_name in ("x", "y", "z"))
        and all(
            hasattr(pose.orientation, field_name)
            for field_name in ("x", "y", "z", "w")
        )
    )


def pose_to_dict(pose: Pose) -> Dict[str, Any]:
    """
    Convert a ROS2 Pose message to a dictionary.

    Args:
        pose: Pose-like object with position and orientation fields

    Returns:
        Dictionary representation of the pose
    """
    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "orientation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def pose_from_dict(data: Dict[str, Any]) -> Pose:
    """
    Create a core Pose from a dictionary.

    Args:
        data: Dictionary with 'position' and 'orientation' keys

    Returns:
        Core Pose object
    """
    pose = Pose()

    position_data = data.get("position", {})
    pose.position.x = position_data.get("x", 0.0)
    pose.position.y = position_data.get("y", 0.0)
    pose.position.z = position_data.get("z", 0.0)

    orientation_data = data.get("orientation", {})
    pose.orientation.x = orientation_data.get("x", 0.0)
    pose.orientation.y = orientation_data.get("y", 0.0)
    pose.orientation.z = orientation_data.get("z", 0.0)
    pose.orientation.w = orientation_data.get("w", 1.0)

    return pose


@dataclass
class BaseNode:
    """
    Base node in the scene graph.

    This is the core node representation with no domain-specific cached fields.
    Use specific subclasses (PoseNode, RoomNode, ObjectNode, NavNode)
    for domain-specific nodes.

    Attributes:
        id: Unique identifier (assigned by Graph)
        pose: Position and orientation in world frame
        created_at: Creation timestamp
        last_seen: Last update timestamp
        node_type: Type of node (AGENT, OBJECT, ROOM, etc.)
        layer: Layer in the scene graph hierarchy
        attributes: Optional metadata dictionary
        active: Active flag for sliding window persistence (e.g., free space nodes)
    """

    id: Optional[int] = None  # Optional ID for the node, can be set later
    pose: Pose = field(default_factory=Pose)
    created_at: float = field(default_factory=time)
    last_seen: Optional[float] = None
    node_type: Optional[NodeType] = None
    layer: Optional[NodeLayer] = None
    attributes: Optional[Dict[str, Any]] = None
    # Active flag for sliding window persistence (free space nodes)
    # When False, the node is outside the active window but retained for revisits
    active: bool = True

    def __post_init__(self):
        # Check that the node_type is a valid NodeLayer
        if self.layer is not None and not isinstance(self.layer, NodeLayer):
            raise ValueError(f"Layer must be a NodeLayer enum, got {self.layer}.")
        # Check that the type is a valid NodeType
        if self.node_type is not None and not isinstance(self.node_type, NodeType):
            raise ValueError(
                f"Node type must be a NodeType enum, got {self.node_type}."
            )
        # Check that the pose has the field shape used by core and ROS poses.
        if self.pose is None or not _is_pose_like(self.pose):
            raise ValueError(f"Pose must have a valid Pose instance, got {self.pose}.")
        # Copy created_at into last_seen if not set
        if self.last_seen is None:
            self.last_seen = self.created_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseNode":
        """
        Reconstruct a Node from a dictionary (e.g., loaded from JSON).
        The incoming dictionary should have at least:
        - "id": Unique identifier for the node
        - "pose": Pose of the node (position and orientation)
        - "created_at": Creation timestamp
        - "last_seen": Last seen timestamp
        - "confidence": Optional confidence score
        - "node_type": Type of the node (as a string or NodeLayer enum)
        - "attributes": Optional dictionary of additional attributes
        """
        return cls(
            id=data.get("id", None),
            pose=pose_from_dict(data.get("pose", {})),
            created_at=data.get("created_at", time()),
            last_seen=data.get("last_seen", None),
            node_type=NodeType.from_string(data["node_type"])
            if "node_type" in data
            else None,
            layer=NodeLayer.from_string(data["layer"]) if "layer" in data else None,
            attributes=data.get("attributes", {}),
        )

    def to_dict(self) -> Dict:
        """
        Convert the Node instance to a dictionary representation (JSON serializable).
        """

        def _ser_time(t):
            # ROS2 builtin_interfaces.msg.Time
            if hasattr(t, "sec") and hasattr(t, "nanosec"):
                return t.sec + t.nanosec * 1e-9
            # Fallback to time in seconds
            if hasattr(t, "isoformat"):
                return t.isoformat()
            return t if isinstance(t, (int, float)) else None

        return {
            "id": self.id,
            "pose": pose_to_dict(self.pose),
            "created_at": _ser_time(self.created_at),
            "last_seen": _ser_time(self.last_seen),
            "node_type": self.node_type.name if self.node_type else None,
            "layer": self.layer.name if self.layer else None,
            "attributes": self.attributes or {},
        }


# ========== Domain-Specific Node Subclasses ==========


@dataclass
class PoseNode(BaseNode):
    """
    Node representing a robot pose (AGENT node in MOTION layer).

    This node represents a snapshot of the robot's position at a specific time,
    used to track the robot's trajectory through the environment.

    Attributes:
        Inherits all BaseNode attributes
        node_type: Always NodeType.AGENT
        layer: Always NodeLayer.MOTION
    """

    def __post_init__(self):
        """Initialize as an AGENT node in MOTION layer."""
        if self.node_type is None:
            self.node_type = NodeType.AGENT
        if self.layer is None:
            self.layer = NodeLayer.MOTION

        # Validate invariants
        if self.node_type != NodeType.AGENT:
            raise ValueError(
                f"PoseNode must have node_type=AGENT, got {self.node_type}"
            )
        if self.layer != NodeLayer.MOTION:
            raise ValueError(f"PoseNode must have layer=MOTION, got {self.layer}")

        # Call parent post_init
        super().__post_init__()


@dataclass
class RoomNode(BaseNode):
    """
    Node representing a room (ROOM node in SEMANTIC layer).

    This node represents a semantic region/room in the environment,
    typically detected through clustering or learned models.

    Attributes:
        Inherits all BaseNode attributes
        node_type: Always NodeType.ROOM
        layer: Always NodeLayer.SEMANTIC
    """

    def __post_init__(self):
        """Initialize as a ROOM node in SEMANTIC layer."""
        if self.node_type is None:
            self.node_type = NodeType.ROOM
        if self.layer is None:
            self.layer = NodeLayer.SEMANTIC

        # Validate invariants
        if self.node_type != NodeType.ROOM:
            raise ValueError(f"RoomNode must have node_type=ROOM, got {self.node_type}")
        if self.layer != NodeLayer.SEMANTIC:
            raise ValueError(f"RoomNode must have layer=SEMANTIC, got {self.layer}")

        # Call parent post_init
        super().__post_init__()


@dataclass
class ObjectNode(BaseNode):
    """
    Node representing an object (OBJECT node in OBJECT layer).

    This node represents a detected object in the environment,
    typically from perception/detection systems.

    Attributes:
        Inherits all BaseNode attributes
        node_type: Always NodeType.OBJECT
        layer: Always NodeLayer.OBJECT
    """

    def __post_init__(self):
        """Initialize as an OBJECT node in OBJECT layer."""
        if self.node_type is None:
            self.node_type = NodeType.OBJECT
        if self.layer is None:
            self.layer = NodeLayer.OBJECT

        # Validate invariants
        if self.node_type != NodeType.OBJECT:
            raise ValueError(
                f"ObjectNode must have node_type=OBJECT, got {self.node_type}"
            )
        if self.layer != NodeLayer.OBJECT:
            raise ValueError(f"ObjectNode must have layer=OBJECT, got {self.layer}")

        # Call parent post_init
        super().__post_init__()


@dataclass
class NavNode(BaseNode):
    """
    Node representing a navigational free-space region (NAVIGATION node).

    Used for free-space grid cells and coarse navigation regions
    (FreeSpaceRegion). For DuDe topological regions, use RegionNode instead.

    Attributes:
        Inherits all BaseNode attributes
        node_type: Always NodeType.NAVIGATION
        layer: Always NodeLayer.NAVIGATION
    """

    def __post_init__(self):
        """Initialize as a NAVIGATION node in NAVIGATION layer."""
        if self.node_type is None:
            self.node_type = NodeType.NAVIGATION
        if self.layer is None:
            self.layer = NodeLayer.NAVIGATION

        # Validate invariants
        if self.node_type != NodeType.NAVIGATION:
            raise ValueError(
                f"NavNode must have node_type=NAVIGATION, got {self.node_type}"
            )
        if self.layer != NodeLayer.NAVIGATION:
            raise ValueError(f"NavNode must have layer=NAVIGATION, got {self.layer}")

        # Call parent post_init
        super().__post_init__()


@dataclass
class RegionNode(BaseNode):
    """
    Node representing a topological region (REGION node in NAVIGATION layer).

    Used for DuDe-managed 2D topological regions that partition the
    environment into semantically meaningful zones.  These are distinct
    from navigational free-space cells/regions (NavNode / FreeSpaceRegion):

    * NavNode / FreeSpaceRegion  → NodeType.NAVIGATION  (metric free-space)
    * RegionNode                 → NodeType.REGION       (topological zones)

    RegionNode instances are created and maintained by RegionManager.

    Attributes:
        Inherits all BaseNode attributes
        node_type: Always NodeType.REGION
        layer: Always NodeLayer.NAVIGATION
    """

    def __post_init__(self):
        """Initialize as a REGION node in NAVIGATION layer."""
        if self.node_type is None:
            self.node_type = NodeType.REGION
        if self.layer is None:
            self.layer = NodeLayer.NAVIGATION

        # Validate invariants
        if self.node_type != NodeType.REGION:
            raise ValueError(
                f"RegionNode must have node_type=REGION, got {self.node_type}"
            )
        if self.layer != NodeLayer.NAVIGATION:
            raise ValueError(f"RegionNode must have layer=NAVIGATION, got {self.layer}")

        # Call parent post_init
        super().__post_init__()
