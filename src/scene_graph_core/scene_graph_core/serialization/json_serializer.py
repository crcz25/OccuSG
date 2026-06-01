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

from scene_graph_core.representation import Edge, SceneGraph

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

SEMANTIC_ATTRIBUTE_KEYS = (
    "class_name",
    "class_id",
    "detection_score",
    "object_id",
    "signature_set",
    "object_in_los",
)

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

    schema_version = "1.0"

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

        node_entries = [self._node_to_entry(node) for node in nodes]
        edge_entries = [self._edge_to_entry(edge) for edge in edges]
        export_metadata = self._build_metadata(metadata, node_entries, edge_entries)

        return {
            "schema_version": self.schema_version,
            "metadata": export_metadata,
            "nodes": node_entries,
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
        attributes = self._json_safe(getattr(node, "attributes", None) or {})
        return {
            "id": self._json_safe(getattr(node, "id", None)),
            "type": self._enum_name(getattr(node, "node_type", None)),
            "layer": self._enum_name(getattr(node, "layer", None)),
            "pose": self._pose_to_json(getattr(node, "pose", None)),
            "created_at": self._json_safe(getattr(node, "created_at", None)),
            "last_seen": self._json_safe(getattr(node, "last_seen", None)),
            "active": self._json_safe(getattr(node, "active", True)),
            "attributes": attributes,
            "geometry": self._project_attributes(attributes, GEOMETRY_ATTRIBUTE_KEYS),
            "semantic": self._project_attributes(attributes, SEMANTIC_ATTRIBUTE_KEYS),
        }

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
                "node_type_counts": self._count_entries(node_entries, "type"),
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
