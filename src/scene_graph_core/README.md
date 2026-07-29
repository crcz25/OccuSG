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
  "schema_version": "2.0",
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

Schema `2.0` gives every `OBJECT` node an explicit `semantic` block so consumers
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
    "room_assigned": true,
    "position": {"x": 1.2, "y": -0.4, "z": 0.6},
    "first_seen": 1778481460.35,
    "last_seen": 1778481492.71,
    "class_name": "chair",
    "class_confidence": 0.82,
    "class_evidence": {"chair": 9.9, "stool": 2.1},
    "detection_observation_count": 14,
    "embedding_observation_count": 14,
    "object_embedding": [0.031, -0.017, "..."],
    "label_embedding": [0.004, 0.052, "..."],
    "mask_embedding": [0.011, -0.008, "..."],
    "bbox_embedding": [0.019, -0.002, "..."],
    "fused_embedding": [0.006, -0.005, "..."],
    "detection_confidence": 0.61,
    "detector_source": "groundingdino",
    "bbox_3d_size": [0.42, 0.38, 0.71],
    "last_semantic_similarity": 0.94
  }
}
```

- Every embedding is a plain numeric JSON array, unit-norm, or `null` when the
  node has no usable vector. `mask_embedding`, `bbox_embedding`, and
  `label_embedding` each have dimension `D`; `object_embedding` also has
  dimension `3D` because it averages `fused_embedding` observations, which are
  `concat(mask, bbox, label)`.
- `object_embedding` is the normalized mean of every valid `fused_embedding`
  associated with the node. The raw sum accumulator lives in `attributes`
  (`embedding_sum`) and is deliberately not exported.
- `detection_observation_count` counts detections folded into the node;
  `embedding_observation_count` counts embeddings folded into the mean. They are
  separate counters and neither is a physical-instance count.
- `class_name` is the canonical class chosen from all accumulated
  `class_evidence`, never the latest detection alone. `class_confidence` is the
  winning class's share of the total evidence mass.
- `room_id` is the object's single `ROOM_CONTAINS` parent; `room_assigned` is
  `false` with `room_id: null` when the object lies outside every DuDe region and
  beyond the configured boundary tolerance.

Attributes written only by removed pipelines — `detection_score`, `object_id`,
`class_id`, `observation_count`, `valid_3d`, `semantic_perception_id`, and the
internal `embedding_sum` — are stripped from both `attributes` and `semantic` on
export.

`scene_graph_core.algorithms.semantic` provides the ROS-free
`normalize_embedding`, `cosine_similarity`, `accumulate_embedding`,
`mean_embedding`, `accumulate_class_evidence`, and
`canonical_class_from_evidence` helpers that define these rules;
`scene_graph_core.algorithms.spatial.assign_region` defines point-in-region room
membership with a boundary tolerance.

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
