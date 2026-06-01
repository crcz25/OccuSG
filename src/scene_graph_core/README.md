# scene_graph_core

Core Python data structures and services for the 3D Scene Graph.

## Persisted-Graph JSON Export

The standalone JSON exporter lives at:

```python
from scene_graph_core.serialization import SceneGraphJsonSerializer
```

It exports all information persisted on nodes and edges in `scene_graph_core`.
It does not inspect ROS messages or transient `scene_graph_ros` manager caches.

```python
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.serialization import SceneGraphJsonSerializer

sg = create_scene_graph_interface()
serializer = SceneGraphJsonSerializer()

metadata = {
    "frame_id": "odom",
    "graph_name": "small_house_run",
    "stamp": "2024-06-01T12:00:00Z",
}

json_dict = serializer.to_dict(sg, metadata=metadata, compact=False)
json_string = serializer.to_json(sg, metadata=metadata, compact=False)
serializer.export_json(sg, "graph.json", metadata=metadata, compact=False)
```

Existing `sg.serialize.to_dict()`, `sg.serialize.to_json()`, and
`sg.serialize.save()` call the same exporter.

## Schema Overview

The top-level JSON object contains:

```json
{
  "schema_version": "1.0",
  "metadata": {
    "frame_id": "odom",
    "graph_name": "small_house_run",
    "stamp": "2024-06-01T12:00:00Z",
    "num_nodes": 1,
    "num_edges": 0,
    "node_type_counts": {"ROOM": 1},
    "edge_type_counts": {}
  },
  "nodes": [
    {
      "id": 4000000,
      "type": "ROOM",
      "layer": "SEMANTIC",
      "pose": {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
      },
      "created_at": 0.0,
      "last_seen": 0.0,
      "active": true,
      "attributes": {"name": "room_0"},
      "geometry": {},
      "semantic": {}
    }
  ],
  "edges": []
}
```

`attributes` preserves the persisted node or edge attributes after JSON-safe
conversion. `geometry` and `semantic` are convenience projections copied from
known keys already present in `attributes`; the original keys remain in
`attributes`.

User metadata is merged with computed metadata. Computed fields
`num_nodes`, `num_edges`, `node_type_counts`, and `edge_type_counts` are
protected and always reflect the exported graph.

## File Writing

`export_json()` writes atomically: it creates a temporary file in the target
directory, writes strict JSON with `allow_nan=False`, flushes it, and replaces
the target path with `os.replace()`. Failed writes clean up the temporary file.

Use `compact=True` for minimal single-line JSON:

```python
serializer.export_json(sg, "graph.compact.json", compact=True)
```

## Limitations

- Only graph-persisted node and edge state is exported.
- ROS runtime caches are not exported unless their contents were already mirrored
  into node or edge attributes.
- ROS messages are intentionally not serialized by the core exporter.
- The current graph storage is a `networkx.DiGraph`, so only the edge currently
  persisted for a given `(source, target)` pair can be exported.
