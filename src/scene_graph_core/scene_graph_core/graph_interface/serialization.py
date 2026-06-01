"""Serialization interface backed by the persisted-graph JSON exporter."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..representation import BaseNode, Edge, EdgeType, NodeLayer, NodeType, SceneGraph
from ..representation.node import pose_from_dict
from ..serialization import SceneGraphJsonSerializer


class SerializationInterface:
    """
    Interface for saving and loading scene graphs.

    Provides methods to:
    - Export graph to dictionary (to_dict)
    - Import graph from dictionary (from_dict)
    - Save graph to JSON file (save)
    - Load graph from JSON file (load)

    Usage:
        # Export to dictionary
        data = sg.serialize.to_dict()

        # Import from dictionary
        sg.serialize.from_dict(data)

        # Save to file
        sg.serialize.save("graph.json")

        # Load from file
        sg.serialize.load("graph.json")
    """

    def __init__(self, graph: SceneGraph):
        """
        Initialize the serialization interface.

        Args:
            graph: The SceneGraph to serialize/deserialize
        """
        self._graph = graph
        self._json_serializer = SceneGraphJsonSerializer()

    def to_dict(
        self,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> Dict[str, Any]:
        """
        Export the scene graph to the active persisted-graph JSON schema.

        Returns:
            JSON-safe dictionary with schema_version, metadata, nodes, and edges.
        """
        return self._json_serializer.to_dict(
            self._graph,
            metadata=metadata,
            compact=compact,
        )

    def to_json(
        self,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> str:
        """Export the scene graph to a JSON string."""
        return self._json_serializer.to_json(
            self._graph,
            metadata=metadata,
            compact=compact,
        )

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import a scene graph from a dictionary.

        WARNING: This clears the existing graph before loading!

        Args:
            data: Dictionary with 'nodes' and 'edges' keys
        """
        # Clear existing graph
        for node in list(self._graph.get_all_nodes()):
            self._graph.remove_node(node.id)

        # Load nodes. Supports both the current export schema and the older
        # round-trip shape used before the exporter replacement.
        nodes_data = data.get("nodes", [])
        for node_data in nodes_data:
            node = self._node_from_serialized_dict(node_data)
            self._graph.add_node(node)

        # Load edges.
        edges_data = data.get("edges", [])
        for edge_data in edges_data:
            edge = self._edge_from_serialized_dict(edge_data)
            self._graph.add_edge(edge, is_structural=edge.is_structural)

    def save(
        self,
        filepath: str,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> Path:
        """
        Save the scene graph to a JSON file using the active exporter.

        Args:
            filepath: Path to the JSON file to create

        Returns:
            Path object pointing to the saved file
        """
        return self.export_json(filepath, metadata=metadata, compact=compact)

    def export_json(
        self,
        filepath: str,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> Path:
        """Atomically export the scene graph to JSON."""
        return self._json_serializer.export_json(
            self._graph,
            filepath,
            metadata=metadata,
            compact=compact,
        )

    def load(self, filepath: str) -> None:
        """
        Load a scene graph from a JSON file.

        WARNING: This clears the existing graph before loading!

        Args:
            filepath: Path to the JSON file to load
        """
        path = Path(filepath)

        with open(path, "r") as f:
            data = json.load(f)

        self.from_dict(data)

    def _node_from_serialized_dict(self, data: Dict[str, Any]) -> BaseNode:
        node_type_value = data.get("type", data.get("node_type"))
        layer_value = data.get("layer")
        node = BaseNode(
            id=data.get("id"),
            pose=pose_from_dict(data.get("pose", {})),
            created_at=data.get("created_at"),
            last_seen=data.get("last_seen"),
            node_type=(
                NodeType.from_string(node_type_value)
                if isinstance(node_type_value, str)
                else node_type_value
            ),
            layer=(
                NodeLayer.from_string(layer_value)
                if isinstance(layer_value, str)
                else layer_value
            ),
            attributes=data.get("attributes", {}),
            active=data.get("active", True),
        )
        return node

    def _edge_from_serialized_dict(self, data: Dict[str, Any]) -> Edge:
        edge_type_value = data.get("type", EdgeType.CUSTOM)
        return Edge(
            source_id=data.get("source_id", data.get("source")),
            target_id=data.get("target_id", data.get("target")),
            id=data.get("id"),
            type=(
                EdgeType.from_string(edge_type_value)
                if isinstance(edge_type_value, str)
                else edge_type_value
            ),
            weight=data.get("weight", 1.0),
            is_structural=data.get("is_structural", data.get("is_tree_edge", True)),
            attributes=data.get("attributes", {}),
        )
