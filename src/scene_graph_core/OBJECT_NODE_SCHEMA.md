# Scene-graph JSON schema migration

The persisted schema is version `3.0`.

## Previous schema

Version `2.0` stored every node in one flat `nodes` list. Entries repeated the
runtime `type` and `layer` and object perception fields were mixed into a flat
`attributes` map. Object semantic data could contain `class_confidence` and
`class_evidence`.

## Current schema

`nodes` is a mapping of concrete node types to arrays. The keys are derived from
the `NodeType` enum as `node_type.value.lower() + "_nodes"`:

| enum value | JSON key | runtime `NodeLayer` |
| --- | --- | --- |
| `AGENT` | `agent_nodes` | `MOTION` |
| `OBJECT` | `object_nodes` | `OBJECT` |
| `NAVIGATION` | `navigation_nodes` | `NAVIGATION` |
| `REGION` | `region_nodes` | `NAVIGATION` |
| `ROOM` | `room_nodes` | `SEMANTIC` |

Every key is emitted, including empty collections. Node entries no longer carry
the redundant `type` or `layer`; the containing collection identifies the
concrete type and deserialization derives the runtime layer.

```json
{
  "schema_version": "3.0",
  "nodes": {
    "room_nodes": [
      {
        "id": 4000000,
        "pose": {"position": {"x": 0, "y": 0, "z": 0}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
        "created_at": 1778481581.55,
        "last_seen": 1778481590.12,
        "active": true,
        "attributes": {"name": "kitchen"}
      }
    ],
    "object_nodes": [
      {
        "id": 1000000,
        "pose": {"position": {"x": 1.2, "y": -0.4, "z": 0.6}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
        "created_at": 1778481581.55,
        "last_seen": 1778481590.12,
        "active": true,
        "semantic": {"class_name": "carpet"},
        "geometry": {"bbox_3d_size": [0.42, 0.38, 0.71]},
        "detection": {"detection_confidence": 0.59},
        "embeddings": {"object_embedding": [0.1, 0.2]},
        "observations": {"detection_observation_count": 1}
      }
    ],
    "navigation_nodes": [],
    "region_nodes": [],
    "agent_nodes": []
  },
  "edges": []
}
```

Object semantic state retains only `semantic.class_name`. The old
`class_confidence` and `class_evidence` fields are neither serialized nor
loaded into the runtime graph. Object embeddings, geometry, detector metadata,
observation counters, timestamps, and other non-obsolete metadata remain
available in their respective groups.

The loader accepts the old flat node list for migration. It uses the legacy
`type`/`layer` only to determine the collection, converts recognized object
fields into the grouped runtime representation, and drops the two removed
semantic fields. New exports always use the grouped schema, so loading an old
file and exporting it again produces version `3.0`. Unknown collection keys,
non-array collections, and malformed entries raise `ValueError` with the
offending collection identified.
