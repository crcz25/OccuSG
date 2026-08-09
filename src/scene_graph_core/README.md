# scene_graph_core

Core Python data structures and services for the 3D Scene Graph.

## Persisted-Graph JSON Export

The standalone JSON exporter lives at:

```python
from scene_graph_core.serialization import SceneGraphJsonSerializer
```

It exports graph-persisted nodes and edges. Existing `sg.serialize.to_dict()`,
`sg.serialize.to_json()`, and `sg.serialize.save()` use the same exporter.

The current JSON schema is version `3.0`. The complete OBJECT-node migration,
field mapping, and compatibility implications are documented in
[`OBJECT_NODE_SCHEMA.md`](OBJECT_NODE_SCHEMA.md).

```json
{
  "schema_version": "3.0",
  "metadata": {"num_nodes": 1, "num_edges": 0},
  "nodes": {"room_nodes": [{"id": 4000000, "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}, "created_at": 0.0, "last_seen": 0.0, "active": true, "attributes": {"name": "room_0"}}], "object_nodes": [], "navigation_nodes": [], "region_nodes": [], "agent_nodes": []},
  "edges": []
}
```

Collection keys are derived from the `NodeType` enum (`ROOM` → `room_nodes`,
`OBJECT` → `object_nodes`, `NAVIGATION` → `navigation_nodes`, `REGION` →
`region_nodes`, and `AGENT` → `agent_nodes`). Every collection is present,
including empty collections. The containing collection replaces the redundant
per-node `type` and `layer` fields; loading derives the runtime `NodeLayer`.

For non-object nodes, `attributes` preserves their persisted custom state.
OBJECT nodes use the grouped canonical blocks `geometry`, `semantic`,
`detection`, `embeddings`, and `observations`; those blocks are not copied into
`attributes`.

`object_embedding` is the normalized running aggregate used for association.
The raw accumulator is retained as `embeddings.sum` because a normalized mean
and count cannot reconstruct it exactly. Component embeddings and
`fused_embedding` remain separate for classification inspection and future
observations. `created_at` is the first observation timestamp, while room
membership is represented by the `ROOM_CONTAINS` graph edge.

`scene_graph_core.algorithms.semantic` provides the ROS-free embedding helpers
used by perception updates.

## File Writing

`export_json()` writes atomically, uses strict JSON (`allow_nan=False`), and
cleans up temporary files after failed writes. Use `compact=True` for a
single-line export.

## Limitations

- Only graph-persisted node and edge state is exported.
- ROS runtime caches are not exported unless mirrored into node or edge state.
- The current graph storage is a `networkx.DiGraph`, so only one edge for a
  given `(source, target)` pair can be persisted.
