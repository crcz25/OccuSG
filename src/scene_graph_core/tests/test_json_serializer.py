import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

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
            "class_name": "chair",
            "class_id": "56",
            "detection_score": np.float32(0.8),
            "object_id": 42,
        },
    )
    obj.pose.position.x = 1.5

    obj_b = ObjectNode(attributes={"class_name": "lamp"})

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


def test_empty_graph_export():
    serializer = SceneGraphJsonSerializer()
    data = serializer.to_dict(create_scene_graph_interface(), metadata={"frame_id": "odom"})

    assert data["schema_version"] == "1.0"
    assert data["metadata"]["frame_id"] == "odom"
    assert data["metadata"]["num_nodes"] == 0
    assert data["metadata"]["num_edges"] == 0
    assert data["nodes"] == []
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

    room_entry = next(node for node in data["nodes"] if node["id"] == ids["room"])
    assert room_entry["type"] == "ROOM"
    assert room_entry["layer"] == "SEMANTIC"
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

    obj_entry = next(node for node in data["nodes"] if node["id"] == ids["obj"])
    assert obj_entry["semantic"]["class_name"] == "chair"
    assert math.isclose(obj_entry["semantic"]["detection_score"], 0.8, rel_tol=1e-6)
    assert obj_entry["semantic"]["object_id"] == 42

    pose_entry = next(node for node in data["nodes"] if node["id"] == ids["pose"])
    assert pose_entry["semantic"]["object_in_los"] == [42, 43]

    nav_entry = next(node for node in data["nodes"] if node["id"] == ids["nav"])
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
    assert [node["id"] for node in first["nodes"]] == sorted(
        node["id"] for node in first["nodes"]
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
    assert data["schema_version"] == "1.0"
    assert "type" in data["nodes"][0]
    assert "node_type" not in data["nodes"][0]
    assert data["metadata"]["graph_name"] == "wrapper"

    target = tmp_path / "wrapper.json"
    sg.serialize.save(target, compact=True)
    saved = target.read_text(encoding="utf-8")
    assert "\n" not in saved
    assert json.loads(saved)["schema_version"] == "1.0"


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
