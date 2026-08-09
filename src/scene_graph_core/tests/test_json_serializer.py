import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scene_graph_core
from scene_graph_core.graph_interface import create_scene_graph_interface
from scene_graph_core.representation import (
    Edge,
    EdgeType,
    NavNode,
    ObjectNode,
    PoseNode,
    RoomNode,
)
from scene_graph_core.serialization import SceneGraphJsonSerializer


def _make_graph():
    sg = create_scene_graph_interface()

    room = RoomNode(
        created_at=10.0,
        last_seen=11.0,
        attributes={
            "name": "kitchen",
            "polygon": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}],
            "convex_hull": [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 2.0}],
            "centroid": {"x": 1.0, "y": 1.0},
            "bounds": {"min_x": 0.0, "min_y": 0.0, "max_x": 2.0, "max_y": 2.0},
            "stable_region_id": 7,
            "tracker_region_id": 8,
            "geometry_signature": {"polygon": [[0.0, 0.0], [2.0, 0.0]]},
            "geometry_source": "direct_navigation",
            "footprint_nav_node_ids": {2000000, 2000001},
            "signature_set": {("chair", 1.0, 2.0), ("table", 3.0, 4.0)},
            "non_finite": float("inf"),
            "np_scalar": np.float64(2.5),
        },
    )
    room.pose.position.x = 1.0
    room.pose.position.y = 2.0
    room.pose.position.z = 3.0

    room_b = RoomNode(attributes={"name": "hall"})

    obj = ObjectNode(
        created_at=20.0,
        last_seen=21.0,
        attributes={
            "geometry": {"bbox_3d_size": [0.4, 0.5, 0.6]},
            "semantic": {"class_name": "chair"},
            "detection": {
                "detection_confidence": np.float32(0.8),
                "detector_source": "groundingdino",
                "semantic_perception_class_id": 3,
            },
            "observations": {
                "detection_observation_count": 5,
                "embedding_observation_count": 4,
                "last_semantic_similarity": 0.91,
            },
            "embeddings": {
                "sum": [6.0, 8.0],
                "object_embedding": [3.0, 4.0],
                "label_embedding": [0.0, 2.0],
                "mask_embedding": [1.0, 0.0],
                "bbox_embedding": [0.0, 1.0],
                "fused_embedding": [1.0, 0.0, 0.0, 1.0, 0.0, 2.0],
            },
        },
    )
    obj.pose.position.x = 1.5

    obj_b = ObjectNode(attributes={"semantic": {"class_name": "lamp"}})

    pose = PoseNode(
        created_at=30.0,
        last_seen=31.0,
        attributes={"object_in_los": {42, 43}},
    )

    pose_b = PoseNode(created_at=32.0, last_seen=33.0)

    nav = NavNode(
        created_at=40.0,
        last_seen=41.0,
        attributes={
            "grid_block": {"x": 1, "y": 2},
            "bounds": {"min_x": 0.0, "max_x": 1.0, "min_y": 0.0, "max_y": 1.0},
            "free_cell_count": 12,
            "meets_minimum_free_cells": True,
        },
    )

    nav_b = NavNode(attributes={"grid_block": {"x": 1, "y": 3}})

    ids = {
        "room": sg.update.add_node(room),
        "room_b": sg.update.add_node(room_b),
        "obj": sg.update.add_node(obj),
        "obj_b": sg.update.add_node(obj_b),
        "pose": sg.update.add_node(pose),
        "pose_b": sg.update.add_node(pose_b),
        "nav": sg.update.add_node(nav),
        "nav_b": sg.update.add_node(nav_b),
    }

    sg.update.add_edge(
        Edge(ids["room"], ids["obj"], type=EdgeType.ROOM_CONTAINS)
    )
    sg.update.add_edge(
        Edge(ids["room_b"], ids["nav_b"], type=EdgeType.REGION_CONTAINS)
    )
    sg.update.add_edge(
        Edge(ids["pose"], ids["pose_b"], type=EdgeType.TEMPORAL_LINK)
    )
    sg.update.add_edge(
        Edge(ids["pose"], ids["obj"], type=EdgeType.OBSERVATION_ANCHOR),
        is_structural=False,
    )
    sg.update.add_edge(
        Edge(ids["nav"], ids["nav_b"], type=EdgeType.NAVIGABLE_PATH),
        is_structural=False,
    )
    sg.update.add_edge(
        Edge(ids["room"], ids["room_b"], type=EdgeType.ROOM_ADJACENCY),
        is_structural=False,
    )
    sg.update.add_edge(
        Edge(
            ids["obj"],
            ids["nav"],
            type=EdgeType.NEAREST_FREE_SPACE,
            attributes={"distance": 0.25},
        ),
        is_structural=False,
    )
    sg.update.add_edge(
        Edge(ids["obj_b"], ids["room_b"], type=EdgeType.CUSTOM),
        is_structural=False,
    )

    return sg, ids


def _entries(data):
    return [entry for collection in data["nodes"].values() for entry in collection]


def test_empty_graph_export():
    serializer = SceneGraphJsonSerializer()
    data = serializer.to_dict(create_scene_graph_interface(), metadata={"frame_id": "odom"})

    assert data["schema_version"] == "3.0"
    assert data["metadata"]["frame_id"] == "odom"
    assert data["metadata"]["num_nodes"] == 0
    assert data["metadata"]["num_edges"] == 0
    assert set(data["nodes"]) == {
        "agent_nodes", "object_nodes", "navigation_nodes", "region_nodes", "room_nodes"
    }
    assert all(collection == [] for collection in data["nodes"].values())
    assert data["edges"] == []


def test_full_persisted_graph_export_and_json_safe_values():
    sg, ids = _make_graph()
    data = SceneGraphJsonSerializer().to_dict(
        sg,
        metadata={
            "frame_id": "odom",
            "graph_name": "small_house_run",
            "stamp": "2024-06-01T12:00:00Z",
            "num_nodes": 999,
        },
    )

    metadata = data["metadata"]
    assert metadata["frame_id"] == "odom"
    assert metadata["graph_name"] == "small_house_run"
    assert metadata["stamp"] == "2024-06-01T12:00:00Z"
    assert metadata["num_nodes"] == 8
    assert metadata["num_edges"] == 8
    assert metadata["node_type_counts"] == {
        "AGENT": 2,
        "NAVIGATION": 2,
        "OBJECT": 2,
        "ROOM": 2,
    }
    assert metadata["edge_type_counts"]["NEAREST_FREE_SPACE"] == 1

    room_entry = next(node for node in _entries(data) if node["id"] == ids["room"])
    assert room_entry["pose"]["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert room_entry["created_at"] == 10.0
    assert room_entry["last_seen"] == 11.0
    assert room_entry["active"] is True
    assert room_entry["attributes"]["stable_region_id"] == 7
    assert room_entry["attributes"]["tracker_region_id"] == 8
    assert room_entry["attributes"]["non_finite"] is None
    assert room_entry["attributes"]["np_scalar"] == 2.5
    assert room_entry["geometry"]["polygon"] == room_entry["attributes"]["polygon"]
    assert "stable_region_id" not in room_entry["geometry"]
    assert room_entry["semantic"]["signature_set"] == [
        ["chair", 1.0, 2.0],
        ["table", 3.0, 4.0],
    ]

    obj_entry = next(node for node in _entries(data) if node["id"] == ids["obj"])
    obj_semantic = obj_entry["semantic"]
    obj_detection = obj_entry["detection"]
    obj_observations = obj_entry["observations"]
    obj_embeddings = obj_entry["embeddings"]
    assert obj_semantic["class_name"] == "chair"
    assert math.isclose(obj_detection["detection_confidence"], 0.8, rel_tol=1e-6)
    assert obj_detection["detector_source"] == "groundingdino"
    assert obj_detection["semantic_perception_class_id"] == 3
    assert obj_observations["detection_observation_count"] == 5
    assert obj_observations["embedding_observation_count"] == 4
    assert math.isclose(obj_observations["last_semantic_similarity"], 0.91, rel_tol=1e-6)
    assert obj_entry["created_at"] == 20.0
    assert obj_entry["last_seen"] == 21.0
    assert obj_entry["geometry"]["bbox_3d_size"] == [0.4, 0.5, 0.6]
    assert "layer" not in obj_entry
    assert "type" not in obj_entry
    assert set(data["nodes"]) == {
        "agent_nodes", "object_nodes", "navigation_nodes", "region_nodes", "room_nodes"
    }
    assert "attributes" not in obj_entry

    # Every embedding is exported as a unit-norm numeric JSON array.
    for key, expected in (
        ("object_embedding", [0.6, 0.8]),
        ("label_embedding", [0.0, 1.0]),
        ("mask_embedding", [1.0, 0.0]),
        ("bbox_embedding", [0.0, 1.0]),
    ):
        vector = obj_embeddings[key]
        assert isinstance(vector, list)
        assert all(isinstance(value, float) for value in vector)
        assert vector == pytest.approx(expected, abs=1e-6)
        assert math.isclose(sum(v * v for v in vector), 1.0, rel_tol=1e-6)

    # The fused vector keeps its 3D dimension through serialization.
    fused = obj_embeddings["fused_embedding"]
    assert len(fused) == 3 * len(obj_embeddings["object_embedding"])
    assert math.isclose(sum(v * v for v in fused), 1.0, rel_tol=1e-6)

    assert obj_embeddings["sum"] == [6.0, 8.0]
    assert "embedding_sum" not in json.dumps(obj_entry)

    obj_b_entry = next(node for node in _entries(data) if node["id"] == ids["obj_b"])
    assert obj_b_entry["semantic"]["class_name"] == "lamp"
    assert "embeddings" not in obj_b_entry
    assert "observations" not in obj_b_entry

    pose_entry = next(node for node in _entries(data) if node["id"] == ids["pose"])
    assert pose_entry["semantic"]["object_in_los"] == [42, 43]

    nav_entry = next(node for node in _entries(data) if node["id"] == ids["nav"])
    assert nav_entry["attributes"]["grid_block"] == {"x": 1, "y": 2}
    assert nav_entry["geometry"]["bounds"] == nav_entry["attributes"]["bounds"]

    nearest_edge = next(
        edge for edge in data["edges"] if edge["type"] == "NEAREST_FREE_SPACE"
    )
    assert nearest_edge["source"] == ids["obj"]
    assert nearest_edge["target"] == ids["nav"]
    assert nearest_edge["is_structural"] is False
    assert nearest_edge["attributes"]["distance"] == 0.25

    assert {edge["type"] for edge in data["edges"]} == {
        "ROOM_CONTAINS",
        "REGION_CONTAINS",
        "TEMPORAL_LINK",
        "OBSERVATION_ANCHOR",
        "NAVIGABLE_PATH",
        "ROOM_ADJACENCY",
        "NEAREST_FREE_SPACE",
        "CUSTOM",
    }


def test_deterministic_ordering_and_compact_output():
    sg, _ = _make_graph()
    serializer = SceneGraphJsonSerializer()

    first = serializer.to_dict(sg)
    second = serializer.to_dict(sg)
    assert first == second
    assert [node["id"] for node in _entries(first)] == sorted(
        node["id"] for node in _entries(first)
    )
    assert first["edges"] == sorted(
        first["edges"],
        key=lambda edge: (edge["source"], edge["target"], edge["type"], edge["id"]),
    )

    compact_json = serializer.to_json(sg, compact=True)
    assert "\n" not in compact_json
    assert "  " not in compact_json
    assert json.loads(compact_json)["metadata"]["num_nodes"] == 8


def test_atomic_export_replaces_existing_file(tmp_path):
    sg, _ = _make_graph()
    target = tmp_path / "graph.json"
    target.write_text('{"old": true}', encoding="utf-8")

    path = SceneGraphJsonSerializer().export_json(
        sg,
        target,
        metadata={"frame_id": "map"},
    )

    assert path == target
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "old" not in data
    assert data["metadata"]["frame_id"] == "map"
    assert not list(tmp_path.glob(".*.tmp"))


def test_scene_graph_interface_serializer_uses_new_export_path(tmp_path):
    sg, _ = _make_graph()

    data = sg.serialize.to_dict(metadata={"graph_name": "wrapper"})
    assert data["schema_version"] == "3.0"
    assert "type" not in _entries(data)[0]
    assert "node_type" not in _entries(data)[0]
    assert data["metadata"]["graph_name"] == "wrapper"

    target = tmp_path / "wrapper.json"
    sg.serialize.save(target, compact=True)
    saved = target.read_text(encoding="utf-8")
    assert "\n" not in saved
    assert json.loads(saved)["schema_version"] == "3.0"


def test_object_embedding_round_trips_through_the_interface(tmp_path):
    sg, ids = _make_graph()
    target = tmp_path / "graph.json"
    sg.serialize.save(target)

    reloaded = create_scene_graph_interface()
    reloaded.serialize.load(str(target))

    obj = reloaded.query.get_node(ids["obj"])
    assert obj.attributes["embeddings"]["object_embedding"] == pytest.approx(
        [0.6, 0.8], abs=1e-6
    )
    assert obj.attributes["embeddings"]["label_embedding"] == pytest.approx(
        [0.0, 1.0], abs=1e-6
    )
    assert len(obj.attributes["embeddings"]["fused_embedding"]) == 6
    reexported = reloaded.serialize.to_dict()
    obj_entry = next(node for node in _entries(reexported) if node["id"] == ids["obj"])
    assert obj_entry["embeddings"]["object_embedding"] == pytest.approx(
        [0.6, 0.8], abs=1e-6
    )
    assert obj_entry["semantic"]["class_name"] == "chair"
    assert obj.attributes["observations"]["detection_observation_count"] == 5
    assert obj.attributes["observations"]["embedding_observation_count"] == 4
    assert obj.attributes["semantic"]["class_name"] == "chair"
    assert obj.attributes["embeddings"]["sum"] == [6.0, 8.0]

    obj_b = reloaded.query.get_node(ids["obj_b"])
    assert "embeddings" not in obj_b.attributes


def test_object_tracking_groups_load_from_canonical_entries():
    sg = create_scene_graph_interface()
    sg.serialize.from_dict(
        {
            "nodes": [
                {
                    "id": 1000000,
                    "type": "OBJECT",
                    "pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}},
                    "attributes": {},
                    "semantic": {"class_name": "chair"},
                    "embeddings": {
                        "object_embedding": [0.0, 1.0],
                        "label_embedding": [1.0, 0.0],
                    },
                    "observations": {
                        "detection_observation_count": 3,
                        "embedding_observation_count": 2,
                    },
                }
            ],
            "edges": [],
        }
    )

    node = sg.query.get_node(1000000)
    assert node.layer.name == "OBJECT"
    assert node.attributes["embeddings"]["object_embedding"] == [0.0, 1.0]
    assert node.attributes["embeddings"]["label_embedding"] == [1.0, 0.0]
    assert node.attributes["semantic"]["class_name"] == "chair"
    assert node.attributes["observations"]["detection_observation_count"] == 3
    assert node.attributes["observations"]["embedding_observation_count"] == 2


def test_object_node_from_dict_reconstructs_grouped_state():
    node = ObjectNode.from_dict(
        {
            "id": 1000001,
            "type": "OBJECT",
            "pose": {"position": {"x": 2.0, "y": 3.0, "z": 4.0}},
            "created_at": 10.0,
            "last_seen": 11.0,
            "semantic": {"class_name": "table"},
            "detection": {"detection_confidence": 0.8},
            "embeddings": {"sum": [1.0, 0.0], "object_embedding": [1.0, 0.0]},
            "observations": {"detection_observation_count": 2},
        }
    )

    assert node.layer.name == "OBJECT"
    assert node.attributes == {
        "semantic": {"class_name": "table"},
        "detection": {"detection_confidence": 0.8},
        "embeddings": {"sum": [1.0, 0.0], "object_embedding": [1.0, 0.0]},
        "observations": {"detection_observation_count": 2},
    }


def test_removed_object_fields_are_not_serialized():
    sg = create_scene_graph_interface()
    node = ObjectNode()
    node.attributes = {
        "semantic": {"class_name": "chair"},
        "detection_score": 0.5,
        "object_id": 42,
        "embedding_sum": [1.0, 0.0],
    }
    node_id = sg.update.add_node(node)

    entry = next(
        item for item in _entries(SceneGraphJsonSerializer().to_dict(sg))
        if item["id"] == node_id
    )
    serialized = json.dumps(entry)
    assert "detection_score" not in serialized
    assert "object_id" not in serialized
    assert "embedding_sum" not in serialized


def test_direct_object_node_serialization_uses_canonical_schema():
    node = ObjectNode(
        attributes={
            "semantic": {"class_name": "chair"},
            "observations": {"detection_observation_count": 1},
        }
    )

    data = node.to_dict()
    assert data["type"] == "OBJECT"
    assert data["layer"] == "OBJECT"
    assert data["semantic"] == {"class_name": "chair"}
    assert data["observations"] == {"detection_observation_count": 1}
    assert "attributes" not in data


def test_node_collections_are_derived_from_node_type_and_reconstruct_layers():
    sg, ids = _make_graph()
    data = SceneGraphJsonSerializer().to_dict(sg)

    assert [entry["id"] for entry in data["nodes"]["room_nodes"]] == sorted(
        (ids["room"], ids["room_b"])
    )
    assert [entry["id"] for entry in data["nodes"]["object_nodes"]] == sorted(
        (ids["obj"], ids["obj_b"])
    )
    assert [entry["id"] for entry in data["nodes"]["navigation_nodes"]] == sorted(
        (ids["nav"], ids["nav_b"])
    )
    assert [entry["id"] for entry in data["nodes"]["agent_nodes"]] == sorted(
        (ids["pose"], ids["pose_b"])
    )
    assert data["nodes"]["region_nodes"] == []

    loaded = create_scene_graph_interface()
    loaded.serialize.from_dict(data)
    assert loaded.query.get_node(ids["room"]).layer.name == "SEMANTIC"
    assert loaded.query.get_node(ids["pose"]).layer.name == "MOTION"
    assert loaded.query.get_node(ids["nav"]).layer.name == "NAVIGATION"
    assert loaded.query.get_node(ids["obj"]).layer.name == "OBJECT"
    assert {
        (edge.source_id, edge.target_id, edge.type)
        for edge in loaded.query.graph.get_all_edges()
    } == {
        (edge.source_id, edge.target_id, edge.type)
        for edge in sg.query.graph.get_all_edges()
    }


def test_unknown_node_collection_key_has_a_clear_error():
    with pytest.raises(ValueError, match="Unknown node-layer collection key"):
        create_scene_graph_interface().serialize.from_dict(
            {"nodes": {"mystery_nodes": []}, "edges": []}
        )


def test_legacy_flat_nodes_are_migrated_without_removed_semantic_state():
    sg = create_scene_graph_interface()
    sg.serialize.from_dict(
        {
            "nodes": [
                {
                    "id": 1000000,
                    "type": "OBJECT",
                    "layer": "OBJECT",
                    "semantic": {
                        "class_name": "carpet",
                        "class_confidence": 0.9,
                        "class_evidence": {"carpet": 2.0},
                    },
                    "object_embedding": [1.0, 0.0],
                    "detection_observation_count": 4,
                }
            ],
            "edges": [],
        }
    )
    node = sg.query.get_node(1000000)
    assert node.attributes["semantic"] == {"class_name": "carpet"}
    assert node.attributes["embeddings"]["object_embedding"] == [1.0, 0.0]
    assert node.attributes["observations"]["detection_observation_count"] == 4
    assert "class_confidence" not in json.dumps(node.attributes)
    assert "class_evidence" not in json.dumps(node.attributes)


def test_object_semantic_schema_contains_only_class_name():
    node = ObjectNode(
        attributes={
            "semantic": {"class_name": "carpet"}
        }
    )
    sg = create_scene_graph_interface()
    node_id = sg.update.add_node(node)
    entry = next(item for item in _entries(SceneGraphJsonSerializer().to_dict(sg)) if item["id"] == node_id)
    assert entry["semantic"] == {"class_name": "carpet"}
    assert "class_confidence" not in json.dumps(entry)
    assert "class_evidence" not in json.dumps(entry)


def test_no_scene_graph_ros_manager_imports_in_core():
    core_root = Path(scene_graph_core.__file__).resolve().parent
    offenders = []
    for path in core_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "scene_graph_ros.managers" in text:
            offenders.append(path.relative_to(core_root))
    assert offenders == []


def test_scene_graph_core_imports_without_geometry_msgs_side_effect():
    core_project_root = str(Path(scene_graph_core.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        core_project_root
        if not env.get("PYTHONPATH")
        else f"{core_project_root}{os.pathsep}{env['PYTHONPATH']}"
    )
    code = (
        "import sys; import scene_graph_core; "
        "print('geometry_msgs' in sys.modules); "
        "print(hasattr(scene_graph_core, 'SceneGraphJsonSerializer'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.stdout.strip().splitlines() == ["False", "True"]
