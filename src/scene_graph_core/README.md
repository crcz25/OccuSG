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
  "schema_version": "1.1",
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

### OBJECT nodes

Schema `1.1` gives every `OBJECT` node an explicit `semantic` block so consumers
never have to reach into the raw attribute bag:

```json
{
  "id": 1000000,
  "type": "OBJECT",
  "layer": "OBJECT",
  "created_at": 1778481460.35,
  "last_seen": 1778481492.71,
  "semantic": {
    "room_id": 4000000,
    "position": {"x": 1.2, "y": -0.4, "z": 0.6},
    "first_seen": 1778481460.35,
    "observation_count": 12,
    "embedding_observation_count": 12,
    "object_embedding": [0.031, -0.017, "..."],
    "last_semantic_similarity": 0.94,
    "class_name": "",
    "class_id": 0,
    "detection_confidence": 0.61,
    "detector_source": "groundingdino",
    "similarity_score": 0.0,
    "entropy_score": 0.0,
    "semantic_perception_id": 2,
    "valid_3d": true,
    "bbox_3d_size": [0.42, 0.38, 0.71]
  }
}
```

- `object_embedding` is a JSON array of numbers: the unit-norm running mean of the
  `fused_embedding` values of every valid observation associated with the node.
  It is `null` when no valid embedding has ever been observed.
- `observation_count` counts detections associated with the node;
  `embedding_observation_count` counts only the embeddings folded into the running
  mean. The two diverge when observations arrive without a usable embedding.
- `room_id` is resolved from the node's single `ROOM_CONTAINS` parent, or `null`.
- `first_seen` mirrors `created_at`; `last_seen` is the most recent association.

`detection_score` and `object_id`, written by the removed detector-based object
pipeline, are stripped from both `attributes` and `semantic` on export.

`scene_graph_core.algorithms.semantic` provides the ROS-free
`normalize_embedding`, `cosine_similarity`, and `running_mean_embedding` helpers
that define these representation rules.

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
