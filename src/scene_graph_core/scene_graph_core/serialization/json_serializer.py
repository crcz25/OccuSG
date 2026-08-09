"""Deterministic JSON export for persisted scene graph state."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from scene_graph_core.algorithms.semantic import normalize_embedding
from scene_graph_core.representation import Edge, NodeType, SceneGraph
from scene_graph_core.representation.node import SUPPORTED_NODE_TYPES
from scene_graph_core.representation.object_schema import (
    OBJECT_ATTRIBUTE_GROUPS,
    OBJECT_EMBEDDING_KEYS,
    OBJECT_KNOWN_KEYS,
    split_object_attributes,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a package dependency.
    np = None


GEOMETRY_ATTRIBUTE_KEYS = (
    "polygon",
    "convex_hull",
    "centroid",
    "bounds",
    "geometry_signature",
    "geometry_refresh_token",
    "geometry_source",
    "footprint_nav_node_ids",
)

SEMANTIC_ATTRIBUTE_KEYS = ("class_name", "signature_set", "object_in_los")

PROTECTED_METADATA_KEYS = frozenset(
    {
        "num_nodes",
        "num_edges",
        "node_type_counts",
        "edge_type_counts",
    }
)


class SceneGraphJsonSerializer:
    """Serialize all persisted nodes and edges in a scene graph to JSON."""

    schema_version = "3.0"
    _LEGACY_OBJECT_KEYS = frozenset(
        {
            "class_name", "class_confidence", "class_evidence",
            "detection_confidence", "detector_source",
            "semantic_perception_class_id", "object_embedding",
            "label_embedding", "mask_embedding", "bbox_embedding",
            "fused_embedding", "embedding_sum", "sum",
            "detection_observation_count", "embedding_observation_count",
            "last_semantic_similarity", "bbox_3d_size",
            "detection_score", "object_id", "class_id", "observation_count",
            "valid_3d", "semantic_perception_id", "similarity_score",
            "entropy_score", "first_seen",
        }
    )

    def to_dict(
        self,
        scene_graph: Any,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> Dict[str, Any]:
        """Return a JSON-safe export dictionary for a graph or graph interface."""
        del compact
        graph = self._resolve_graph(scene_graph)
        nodes = sorted(
            graph.get_all_nodes(),
            key=lambda node: self._stable_sort_key(getattr(node, "id", None)),
        )
        edges = sorted(
            graph.get_all_edges(),
            key=lambda edge: (
                self._stable_sort_key(getattr(edge, "source_id", None)),
                self._stable_sort_key(getattr(edge, "target_id", None)),
                self._stable_sort_key(self._enum_name(getattr(edge, "type", None))),
                self._stable_sort_key(getattr(edge, "id", None)),
            ),
        )

        grouped_nodes = {node_type.collection_key: [] for node_type in SUPPORTED_NODE_TYPES}
        for node in nodes:
            node_type = getattr(node, "node_type", None)
            if not isinstance(node_type, NodeType):
                raise ValueError(
                    f"Cannot serialize node {getattr(node, 'id', None)!r}: "
                    f"unsupported node type {node_type!r}"
                )
            grouped_nodes[node_type.collection_key].append(self._node_to_entry(node))
        for entries in grouped_nodes.values():
            entries.sort(key=lambda entry: self._stable_sort_key(entry.get("id")))

        node_entries = [entry for entries in grouped_nodes.values() for entry in entries]
        edge_entries = [self._edge_to_entry(edge) for edge in edges]
        export_metadata = self._build_metadata(
            metadata, node_entries, edge_entries, grouped_nodes
        )

        return {
            "schema_version": self.schema_version,
            "metadata": export_metadata,
            "nodes": grouped_nodes,
            "edges": edge_entries,
        }

    def to_json(
        self,
        scene_graph: Any,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> str:
        """Return a deterministic JSON string for the scene graph."""
        data = self.to_dict(scene_graph, metadata=metadata, compact=compact)
        return json.dumps(
            data,
            allow_nan=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else (",", ": "),
            sort_keys=True,
        )

    def export_json(
        self,
        scene_graph: Any,
        path: str | Path,
        metadata: Optional[Mapping[str, Any]] = None,
        compact: bool = False,
    ) -> Path:
        """Atomically write a deterministic JSON export to *path*."""
        target_path = Path(path)
        target_dir = target_path.parent if target_path.parent != Path("") else Path(".")
        tmp_path = None
        data = self.to_dict(scene_graph, metadata=metadata, compact=compact)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=target_dir,
                prefix=f".{target_path.name}.",
                suffix=".tmp",
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                json.dump(
                    data,
                    tmp_file,
                    allow_nan=False,
                    indent=None if compact else 2,
                    separators=(",", ":") if compact else (",", ": "),
                    sort_keys=True,
                )
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, target_path)
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except TypeError:  # Python < 3.8 compatibility.
                    if tmp_path.exists():
                        tmp_path.unlink()
            raise

        return target_path

    def _resolve_graph(self, scene_graph: Any) -> SceneGraph:
        if isinstance(scene_graph, SceneGraph):
            return scene_graph
        if hasattr(scene_graph, "query") and hasattr(scene_graph.query, "graph"):
            return scene_graph.query.graph
        if hasattr(scene_graph, "_graph") and isinstance(scene_graph._graph, SceneGraph):
            return scene_graph._graph
        raise TypeError(
            "Expected a SceneGraph or SceneGraphInterface-compatible object, "
            f"got {type(scene_graph)!r}"
        )

    def _node_to_entry(self, node: Any) -> Dict[str, Any]:
        if getattr(node, "node_type", None) == NodeType.OBJECT:
            return self._object_node_to_entry(node)

        attributes = dict(self._json_safe(getattr(node, "attributes", None) or {}))

        entry = {
            "id": self._json_safe(getattr(node, "id", None)),
            "pose": self._pose_to_json(getattr(node, "pose", None)),
            "created_at": self._json_safe(getattr(node, "created_at", None)),
            "last_seen": self._json_safe(getattr(node, "last_seen", None)),
            "active": self._json_safe(getattr(node, "active", True)),
            "attributes": attributes,
            "geometry": self._project_attributes(attributes, GEOMETRY_ATTRIBUTE_KEYS),
            "semantic": self._project_attributes(attributes, SEMANTIC_ATTRIBUTE_KEYS),
        }
        return entry

    def _object_node_to_entry(self, node: Any) -> Dict[str, Any]:
        """Serialize one OBJECT node without projection duplicates."""
        attributes = getattr(node, "attributes", None) or {}
        groups = split_object_attributes(attributes)

        # Custom attributes are retained, while canonical groups are promoted
        # exactly once to their named JSON locations.
        custom_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in OBJECT_ATTRIBUTE_GROUPS
            and key not in OBJECT_KNOWN_KEYS
            and key not in self._LEGACY_OBJECT_KEYS
        }
        entry: Dict[str, Any] = {
            "id": self._json_safe(getattr(node, "id", None)),
            "pose": self._pose_to_json(getattr(node, "pose", None)),
            "created_at": self._json_safe(getattr(node, "created_at", None)),
            "last_seen": self._json_safe(getattr(node, "last_seen", None)),
            "active": self._json_safe(getattr(node, "active", True)),
        }
        if custom_attributes:
            entry["attributes"] = self._json_safe(custom_attributes)

        if "semantic" in groups:
            semantic = groups["semantic"]
            if "class_name" in semantic:
                entry["semantic"] = {"class_name": self._json_safe(semantic["class_name"])}

        for group in ("geometry", "detection", "observations"):
            if group in groups:
                entry[group] = self._json_safe(groups[group])

        if "embeddings" in groups:
            embeddings = {}
            for key, value in groups["embeddings"].items():
                if key == "sum":
                    embeddings[key] = self._json_safe(value)
                    continue
                if key in OBJECT_EMBEDDING_KEYS:
                    vector = normalize_embedding(value)
                    embeddings[key] = (
                        vector.astype(np.float32).tolist()
                        if vector is not None
                        else None
                    )
                else:
                    embeddings[key] = self._json_safe(value)
            entry["embeddings"] = embeddings
        return entry

    def _edge_to_entry(self, edge: Edge) -> Dict[str, Any]:
        return {
            "id": self._json_safe(getattr(edge, "id", None)),
            "source": self._json_safe(edge.source_id),
            "target": self._json_safe(edge.target_id),
            "type": self._enum_name(edge.type),
            "weight": self._json_safe(getattr(edge, "weight", None)),
            "is_structural": self._json_safe(getattr(edge, "is_structural", None)),
            "attributes": self._json_safe(getattr(edge, "attributes", None) or {}),
        }

    def _build_metadata(
        self,
        metadata: Optional[Mapping[str, Any]],
        node_entries: list[Dict[str, Any]],
        edge_entries: list[Dict[str, Any]],
        grouped_nodes: Mapping[str, list[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        user_metadata = dict(metadata or {})
        result = {
            "graph_name": None,
            "frame_id": None,
            "stamp": None,
            "export_time_unix": None,
        }
        result.update(self._json_safe(user_metadata))
        result.update(
            {
                "num_nodes": len(node_entries),
                "num_edges": len(edge_entries),
                "node_type_counts": {
                    node_type.value: len(grouped_nodes[node_type.collection_key])
                    for node_type in SUPPORTED_NODE_TYPES
                    if grouped_nodes[node_type.collection_key]
                },
                "edge_type_counts": self._count_entries(edge_entries, "type"),
            }
        )
        return self._json_safe(result)

    def _count_entries(self, entries: list[Dict[str, Any]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in entries:
            value = entry.get(key)
            name = "null" if value is None else str(value)
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    def _project_attributes(
        self,
        attributes: Any,
        keys: tuple[str, ...],
    ) -> Dict[str, Any]:
        if not isinstance(attributes, Mapping):
            return {}
        return {
            key: attributes[key]
            for key in keys
            if key in attributes
        }

    def _pose_to_json(self, pose: Any) -> Optional[Dict[str, Any]]:
        if pose is None or not (
            hasattr(pose, "position") and hasattr(pose, "orientation")
        ):
            return None
        return {
            "position": self._point_to_json(pose.position),
            "orientation": {
                "x": self._json_safe(getattr(pose.orientation, "x", 0.0)),
                "y": self._json_safe(getattr(pose.orientation, "y", 0.0)),
                "z": self._json_safe(getattr(pose.orientation, "z", 0.0)),
                "w": self._json_safe(getattr(pose.orientation, "w", 1.0)),
            },
        }

    def _point_to_json(self, point: Any) -> Dict[str, Any]:
        return {
            "x": self._json_safe(getattr(point, "x", 0.0)),
            "y": self._json_safe(getattr(point, "y", 0.0)),
            "z": self._json_safe(getattr(point, "z", 0.0)),
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value

        if isinstance(value, float):
            return value if math.isfinite(value) else None

        if isinstance(value, Enum):
            return self._enum_name(value)

        if np is not None:
            if isinstance(value, np.generic):
                return self._json_safe(value.item())
            if isinstance(value, np.ndarray):
                return self._json_safe(value.tolist())

        if isinstance(value, Path):
            return str(value)

        if dataclasses.is_dataclass(value):
            return self._json_safe(dataclasses.asdict(value))

        if self._is_pose_like(value):
            return self._pose_to_json(value)

        if self._is_point_like(value):
            return self._point_to_json(value)

        if isinstance(value, Mapping):
            converted = {}
            for raw_key, raw_value in value.items():
                key = raw_key if isinstance(raw_key, str) else str(self._json_safe(raw_key))
                converted[key] = self._json_safe(raw_value)
            return dict(sorted(converted.items(), key=lambda item: item[0]))

        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]

        if isinstance(value, (set, frozenset)):
            converted = [self._json_safe(item) for item in value]
            return sorted(converted, key=self._stable_json_sort_value)

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return self._json_safe(to_dict())
            except TypeError:
                pass

        if hasattr(value, "__dict__"):
            return self._json_safe(vars(value))

        return str(value)

    def _stable_json_sort_value(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, allow_nan=False)
        except TypeError:
            return str(value)

    def _stable_sort_key(self, value: Any) -> tuple[str, Any]:
        if value is None:
            return ("2", "")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ("0", value if math.isfinite(float(value)) else float("inf"))
        return ("1", str(value))

    def _enum_name(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Enum):
            return str(value.name)
        return str(value)

    def _is_point_like(self, value: Any) -> bool:
        return all(hasattr(value, field_name) for field_name in ("x", "y", "z"))

    def _is_pose_like(self, value: Any) -> bool:
        return hasattr(value, "position") and hasattr(value, "orientation")
