"""
Free Space Region - Coarse-grained abstraction of free-space cells.

This module defines FreeSpaceRegion, a scene-graph node representing a block
of contiguous free-space cells (e.g., 10×10 cells). It replaces tens of thousands
of individual NAVIGATION cell nodes when region abstraction is enabled.

FreeSpaceRegion nodes:
- Are BaseNode instances with NodeType.NAVIGATION
- Store aggregate properties (centroid, bounds, cell count)
- Connect to adjacent regions via NAVIGABLE_PATH edges
- Can be assigned to rooms via CONTAINS edges
- Enable efficient room-based queries and visualization

Design Goals:
1. Memory Efficiency: 100-300 region nodes instead of 10k-100k cell nodes
2. Query Efficiency: O(rooms × regions) instead of O(rooms × cells)
3. Visualization Efficiency: Render regions as large markers (1/100th marker count)
4. Backward Compatibility: System works with or without region abstraction

Architecture:
    FreeSpaceNodeManager creates NAVIGATION nodes when use_region_abstraction=True
    ↓
    FreeSpaceRegion nodes added to SceneGraph (BaseNode instances)
    ↓
    BackgroundNode assigns regions to rooms (ray casting + BFS over regions)
    ↓
    VisualizationNode renders region markers (spheres at centroids)

Usage:
    # Create region
    region = FreeSpaceRegion(
        region_id=(5, 3),           # Grid coordinates of region
        centroid=(12.5, 7.3),       # World coordinates
        bounds=(10.0, 15.0, 5.0, 10.0),  # (xmin, xmax, ymin, ymax)
        cell_count=85               # Number of free cells in region
    )

    # Add to scene graph
    node_id = sg.update.add_node(region)

    # Update properties
    region.cell_count = 90
    region.pose.position.x = 12.8  # Updated centroid
    sg.update.update_node(node_id, region)
"""

from dataclasses import dataclass, field
from typing import Optional, Set, Tuple

from .node import BaseNode, NodeLayer, NodeType


@dataclass
class FreeSpaceRegion(BaseNode):
    """
    A coarse-grained free-space region representing multiple grid cells.

    This is a scene-graph node (BaseNode) with NodeType.NAVIGATION that
    abstracts a block of contiguous free cells. It replaces individual
    NAVIGATION nodes when region abstraction is enabled.

    Attributes:
        region_id: Grid coordinates of the region (ri, rj) in region space
        bounds: World-coordinate bounds (xmin, xmax, ymin, ymax) in meters
        cell_count: Number of free cells in this region
        neighbor_regions: Set of adjacent region node IDs (connected via NAVIGABLE_PATH)
        room_id: Room this region belongs to (via CONTAINS edge from room)

    Inherited from BaseNode:
        id: Global node ID in scene graph
        pose: Centroid position in world coordinates (x, y, z=0.05)
        created_at: Timestamp when region was first created
        last_seen: Timestamp of most recent update
        node_type: Always NodeType.NAVIGATION
        layer: Always NodeLayer.NAVIGATION
        active: True if in sliding window, False if outside (retained for revisits)
        attributes: Optional metadata dictionary
    """

    # Region-specific properties
    region_id: Optional[Tuple[int, int]] = None  # (ri, rj) in region grid space
    bounds: Optional[Tuple[float, float, float, float]] = (
        None  # (xmin, xmax, ymin, ymax)
    )
    cell_count: int = 0  # Number of free cells in this region
    neighbor_regions: Set[int] = field(default_factory=set)  # Adjacent region node IDs
    room_id: Optional[int] = None  # Room assignment (from CONTAINS edge)

    def __post_init__(self):
        """Initialize as a NAVIGATION node in NAVIGATION layer."""
        # Set node type and layer if not already set
        if self.node_type is None:
            self.node_type = NodeType.NAVIGATION
        if self.layer is None:
            self.layer = NodeLayer.NAVIGATION

        # Validate that this is indeed a NAVIGATION node
        if self.node_type != NodeType.NAVIGATION:
            raise ValueError(
                f"FreeSpaceRegion must have node_type=NAVIGATION, got {self.node_type}"
            )
        if self.layer != NodeLayer.NAVIGATION:
            raise ValueError(
                f"FreeSpaceRegion must have layer=NAVIGATION, got {self.layer}"
            )

        # Initialize neighbor set if needed
        if self.neighbor_regions is None:
            self.neighbor_regions = set()

        # Call parent post_init
        super().__post_init__()

    def get_centroid(self) -> Tuple[float, float]:
        """
        Get region centroid in world coordinates.

        Returns:
            (x, y) tuple from pose position
        """
        return (self.pose.position.x, self.pose.position.y)

    def get_area(self) -> float:
        """
        Calculate region area in square meters.

        Returns:
            Area in m² (computed from bounds)
        """
        if self.bounds is None:
            return 0.0
        xmin, xmax, ymin, ymax = self.bounds
        return (xmax - xmin) * (ymax - ymin)

    def add_neighbor(self, region_node_id: int):
        """
        Add an adjacent region to the neighbor set.

        Args:
            region_node_id: Global node ID of adjacent NAVIGATION node
        """
        self.neighbor_regions.add(region_node_id)

    def remove_neighbor(self, region_node_id: int):
        """
        Remove a region from the neighbor set.

        Args:
            region_node_id: Global node ID to remove
        """
        self.neighbor_regions.discard(region_node_id)

    def update_bounds(self, xmin: float, xmax: float, ymin: float, ymax: float):
        """
        Update region bounds in world coordinates.

        Args:
            xmin, xmax, ymin, ymax: Bounding box in meters
        """
        self.bounds = (xmin, xmax, ymin, ymax)

    def update_centroid(self, x: float, y: float, z: float = 0.05):
        """
        Update region centroid position.

        Args:
            x, y: Centroid coordinates in meters
            z: Z offset for visualization (default: 0.05)
        """
        self.pose.position.x = x
        self.pose.position.y = y
        self.pose.position.z = z

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with all FreeSpaceRegion attributes
        """
        # Get base node dictionary
        data = super().to_dict()

        # Add region-specific fields
        data["region_id"] = self.region_id
        data["bounds"] = self.bounds
        data["cell_count"] = self.cell_count
        data["neighbor_regions"] = list(self.neighbor_regions)  # Convert set to list
        data["room_id"] = self.room_id

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FreeSpaceRegion":
        """
        Reconstruct FreeSpaceRegion from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            FreeSpaceRegion instance
        """
        # Extract region-specific fields
        region_id = data.get("region_id")
        bounds = data.get("bounds")
        cell_count = data.get("cell_count", 0)
        neighbor_regions = set(data.get("neighbor_regions", []))
        room_id = data.get("room_id")

        # Create instance using BaseNode.from_dict for common fields
        base_node = BaseNode.from_dict(data)

        # Create FreeSpaceRegion with all fields
        region = cls(
            id=base_node.id,
            pose=base_node.pose,
            created_at=base_node.created_at,
            last_seen=base_node.last_seen,
            node_type=NodeType.NAVIGATION,
            layer=NodeLayer.NAVIGATION,
            attributes=base_node.attributes,
            active=base_node.active,
            region_id=region_id,
            bounds=bounds,
            cell_count=cell_count,
            neighbor_regions=neighbor_regions,
            room_id=room_id,
        )

        return region

    def __repr__(self) -> str:
        """String representation for debugging."""
        centroid = self.get_centroid()
        return (
            f"FreeSpaceRegion(id={self.id}, region_id={self.region_id}, "
            f"centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), "
            f"cell_count={self.cell_count}, room_id={self.room_id}, "
            f"active={self.active})"
        )
