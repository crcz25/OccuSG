"""Serialization interface backed by the persisted-graph JSON exporter."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..representation import (
    BaseNode,
    Edge,
    EdgeType,
    NodeLayer,
    NodeType,
    SceneGraph,
    node_factory,
    node_layer_for_type,
)
from ..representation.node import pose_from_dict
from ..representation.object_schema import (
    OBJECT_ATTRIBUTE_GROUPS,
    OBJECT_EMBEDDING_KEYS,
)
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
        for node_data, collection_type in self._iter_serialized_nodes(data.get("nodes", [])):
            node = self._node_from_serialized_dict(node_data, collection_type)
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

    @staticmethod
    def _iter_serialized_nodes(nodes_data):
        """Yield node entries from the grouped schema or the legacy flat list."""
        if isinstance(nodes_data, list):
            for node_data in nodes_data:
                if not isinstance(node_data, dict):
                    raise ValueError("Each legacy node entry must be an object")
                yield node_data, None
            return
        if not isinstance(nodes_data, dict):
            raise ValueError("'nodes' must be a node-layer mapping or legacy list")

        for collection_key, entries in nodes_data.items():
            node_type = NodeType.from_collection_key(collection_key)
            if not isinstance(entries, list):
                raise ValueError(
                    f"Node collection {collection_key!r} must contain a list"
                )
            for node_data in entries:
                if not isinstance(node_data, dict):
                    raise ValueError(
                        f"Node collection {collection_key!r} contains a non-object entry"
                    )
                yield node_data, node_type

    def _node_from_serialized_dict(
        self, data: Dict[str, Any], collection_type: Optional[NodeType] = None
    ) -> BaseNode:
        node_type_value = data.get("type", data.get("node_type"))
        node_type = collection_type
        if node_type is None and isinstance(node_type_value, str):
            node_type = NodeType.from_string(node_type_value)
        if node_type is None and isinstance(data.get("layer"), str):
            layer = NodeLayer.from_string(data["layer"])
            node_type = {
                NodeLayer.SEMANTIC: NodeType.ROOM,
                NodeLayer.OBJECT: NodeType.OBJECT,
                NodeLayer.NAVIGATION: NodeType.NAVIGATION,
                NodeLayer.MOTION: NodeType.AGENT,
            }.get(layer)
            if node_type is None:
                raise ValueError(
                    f"Cannot infer a concrete node type from legacy layer {layer.value!r}"
                )
        if node_type is None:
            raise ValueError("Node entry has no collection, type, node_type, or layer")

        layer_value = node_layer_for_type(node_type)
        if collection_type is None and isinstance(data.get("layer"), str):
            # The legacy flat schema carried the runtime hierarchy layer.
            layer_value = NodeLayer.from_string(data["layer"])
        node = node_factory(
            id=data.get("id"),
            pose=pose_from_dict(data.get("pose", {})),
            created_at=data.get("created_at"),
            last_seen=data.get("last_seen"),
            node_type=node_type,
            layer=layer_value,
            attributes=self._object_attributes(data, node_type),
            active=data.get("active", True),
        )
        return node

    @staticmethod
    def _object_attributes(
        data: Dict[str, Any], node_type: NodeType
    ) -> Dict[str, Any]:
        """Reconstruct runtime attributes, migrating legacy object entries."""
        attributes = dict(data.get("attributes", {}) or {})
        if node_type != NodeType.OBJECT:
            return attributes

        result = {
            key: value
            for key, value in attributes.items()
            if key not in OBJECT_ATTRIBUTE_GROUPS
            and key not in {
                "class_name", "class_confidence", "class_evidence",
                "detection_confidence", "detector_source",
                "semantic_perception_class_id", "object_embedding", "label_embedding",
                "detection_observation_count", "embedding_observation_count",
                "last_semantic_similarity", "bbox_3d_size",
                "detection_score", "object_id", "class_id", "observation_count",
                "valid_3d", "semantic_perception_id", "similarity_score",
                "entropy_score", "first_seen",
            }
        }

        sources = [attributes, data]
        for key in ("geometry", "semantic", "detection", "embeddings", "observations"):
            value = data.get(key)
            if isinstance(value, dict):
                sources.append(value)
        groups = {group: {} for group in OBJECT_ATTRIBUTE_GROUPS}
        for source in sources:
            for key, group in {
                "class_name": "semantic",
                "bbox_3d_size": "geometry",
                "detection_confidence": "detection",
                "detector_source": "detection",
                "semantic_perception_class_id": "detection",
                "object_embedding": "embeddings",
                "label_embedding": "embeddings",
                "detection_observation_count": "observations",
                "embedding_observation_count": "observations",
                "last_semantic_similarity": "observations",
            }.items():
                if key in source:
                    groups[group][key] = source[key]
        for group in OBJECT_ATTRIBUTE_GROUPS:
            value = data.get(group)
            if isinstance(value, dict):
                if group == "embeddings":
                    value = {
                        key: entry for key, entry in value.items()
                        if key in OBJECT_EMBEDDING_KEYS
                    }
                groups[group].update(value)
        semantic = groups["semantic"]
        if "class_name" in semantic:
            groups["semantic"] = {"class_name": semantic["class_name"]}
        return {
            **result,
            **{group: values for group, values in groups.items() if values},
        }

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
