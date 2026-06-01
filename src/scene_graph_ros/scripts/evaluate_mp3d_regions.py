#!/usr/bin/env python3
"""Evaluate scene graph room/region footprints against Matterport3D regions.

Usage:
    python3 /mnt/DATA_5TB/repos/phd/3dsg/src/scene_graph_ros/scripts/evaluate_mp3d_regions.py \
        --scan_id 17DRP5sb8fy \
        --dsg_dir /mnt/DATA_5TB/repos/phd/3dsg/results_ablation/3.0 \
        --mp3d_root /mnt/DATA_5TB/repos/phd/3dsg/mp3d/dataset/v1/scans \
        --output_dir /mnt/DATA_5TB/repos/phd/3dsg/results_ablation/3.0/region_evals

Defaults:
    --dsg_dir .
    --mp3d_root /mnt/DATA/repos/phd/3dsg/mp3d/dataset/v1/scans
    --output_dir .

Outputs:
    region_eval_summary_<scan_id>.csv (one-row canonical metric summary)
    region_eval_summary_<scan_id>.json

The evaluator is intentionally standalone: it does not import ROS packages and
only assumes that the exported scene graph and Matterport annotations share the
same metric global frame. All geometry is projected to the x-y ground plane.
If the graph was exported in a different coordinate frame, IoU and adjacency
metrics will be poor; provide/alignment-transform support should be added before
interpreting those numbers as segmentation errors.

Matterport ground-truth regions are vertical prisms defined by the manual
house_segmentations .house file. This script treats those .house region records
as the room/region units and evaluates their horizontal footprints. The
region_segmentations meshes are not used for room/region segmentation metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on local environment
    np = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - depends on local environment
    linear_sum_assignment = None

try:
    from shapely.geometry import MultiPoint, Polygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - handled in main()
    MultiPoint = None
    Polygon = None
    unary_union = None


REGION_LABELS = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",
    "e": "entryway/foyer/lobby",
    "f": "family room",
    "g": "garage",
    "h": "hallway",
    "i": "library",
    "j": "laundry room",
    "k": "kitchen",
    "l": "living room",
    "m": "meeting room",
    "n": "lounge",
    "o": "office",
    "p": "porch/terrace/deck",
    "r": "recreation/game room",
    "s": "stairs",
    "t": "toilet",
    "u": "utility room",
    "v": "tv room",
    "w": "workout/gym/exercise",
    "x": "outdoor area",
    "y": "balcony",
    "z": "other room",
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa/sauna",
    "Z": "junk",
    "-": "unknown",
}


POLYGON_KEYS = ("polygon", "footprint", "contour", "boundary")
HULL_KEYS = ("convex_hull", "hull")
CENTROID_KEYS = ("centroid", "center", "position", "translation", "pose", "xyz")
AREA_KEYS = ("area", "region_area", "footprint_area")
LABEL_KEYS = ("semantic_label", "label", "class_name", "category", "name")
STABLE_ID_KEYS = (
    "stable_region_id",
    "stable_id",
    "tracker_region_id",
    "region_id",
    "dude_region_id",
)
ADJACENCY_KEYS = ("adjacent_ids", "adjacent_region_ids", "neighbors", "neighbours")
SEVERE_UNDERSEGMENTATION_THRESHOLD = 0.25

CANONICAL_REGION_FIELDS = (
    "scan_id",
    "method",
    "num_gt_regions",
    "num_pred_regions",
    "num_matches",
    "num_unmatched_gt",
    "num_unmatched_pred",
    "region_count_error_abs",
    "region_count_error_rel",
    "region_count_ratio",
    "region_precision",
    "region_recall",
    "region_f1",
    "gt_coverage",
    "mean_iou_matched",
    "mean_iou_gt_penalized",
    "mean_iou_full_penalized",
    "oversegmentation_rate",
    "undersegmentation_rate",
    "no_predicted_regions",
    "no_valid_matches",
    "single_region_failure",
    "severe_undersegmentation",
    "region_count_collapse",
    "failure_mode",
    "source_file",
)


@dataclass
class PredRegion:
    node_id: str
    node_type: str
    label: Optional[str]
    stable_region_id: Optional[str]
    polygon: Any
    polygon_source: Optional[str]
    centroid: Optional[Tuple[float, float]]
    area: Optional[float]
    adjacent_ids: Set[str] = field(default_factory=set)
    raw: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class GtRegion:
    region_index: int
    level_index: int
    label_code: str
    label_name: str
    polygon: Any
    polygon_source: str
    centroid: Tuple[float, float]
    area: float
    height: Optional[float] = None


@dataclass
class Pairwise:
    pred_id: str
    gt_id: int
    iou: float
    intersection_area: float
    union_area: float


def warn(warnings: List[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_int_token(value: str) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def nested_dicts(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield node
    for key in (
        "attributes",
        "geometry",
        "semantic",
        "metadata",
        "data",
        "properties",
        "geometry_signature",
    ):
        value = node.get(key)
        if isinstance(value, dict):
            yield value


def first_nested_value(node: Dict[str, Any], keys: Sequence[str]) -> Any:
    for scope in nested_dicts(node):
        for key in keys:
            if key in scope and scope[key] not in (None, ""):
                return scope[key]
    return None


def parse_literal_dict(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or "{" not in value:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_xy(value: Any, plane: str = "xy") -> Optional[Tuple[float, float]]:
    if value is None:
        return None

    if isinstance(value, dict):
        if "position" in value:
            nested = extract_xy(value["position"], plane=plane)
            if nested is not None:
                return nested
        if {"x", "y"}.issubset(value.keys()):
            coords = {
                "x": to_float(value.get("x")),
                "y": to_float(value.get("y")),
                "z": to_float(value.get("z")),
            }
            return project_coord(coords, plane)
        for key in CENTROID_KEYS:
            if key in value:
                nested = extract_xy(value[key], plane=plane)
                if nested is not None:
                    return nested

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        vals = [to_float(v) for v in value[:3]]
        if vals[0] is None or vals[1] is None:
            return None
        coords = {"x": vals[0], "y": vals[1], "z": vals[2] if len(vals) > 2 else None}
        return project_coord(coords, plane)

    return None


def project_coord(
    coords: Dict[str, Optional[float]], plane: str
) -> Optional[Tuple[float, float]]:
    if plane == "xy":
        a, b = coords.get("x"), coords.get("y")
    elif plane == "xz":
        a, b = coords.get("x"), coords.get("z")
    elif plane == "yz":
        a, b = coords.get("y"), coords.get("z")
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    if a is None or b is None:
        return None
    return (float(a), float(b))


def extract_points(value: Any, plane: str = "xy") -> List[Tuple[float, float]]:
    """Extract a point sequence from common ROS/JSON encodings."""
    if value is None:
        return []

    if isinstance(value, dict):
        for key in ("points", "vertices", "coordinates", "polygon"):
            if key in value:
                points = extract_points(value[key], plane=plane)
                if points:
                    return points
        xy = extract_xy(value, plane=plane)
        return [xy] if xy is not None else []

    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(
            not isinstance(v, (dict, list, tuple)) for v in value[:2]
        ):
            xy = extract_xy(value, plane=plane)
            return [xy] if xy is not None else []
        points: List[Tuple[float, float]] = []
        for item in value:
            xy = extract_xy(item, plane=plane)
            if xy is not None:
                points.append(xy)
        return points

    return []


def clean_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if (
            not cleaned
            or abs(cleaned[-1][0] - x) > 1e-9
            or abs(cleaned[-1][1] - y) > 1e-9
        ):
            cleaned.append((float(x), float(y)))
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def make_polygon(
    points: Sequence[Tuple[float, float]],
    warnings: List[str],
    context: str,
    allow_convex: bool = False,
) -> Optional[Any]:
    if Polygon is None:
        return None
    points = clean_points(points)
    if len(points) < 3:
        return None

    geom = Polygon(points)
    if (not geom.is_valid or geom.is_empty or geom.area <= 0.0) and allow_convex:
        geom = MultiPoint(points).convex_hull
    if not geom.is_valid:
        repaired = geom.buffer(0)
        if repaired.is_valid and not repaired.is_empty:
            warn(warnings, f"Repaired invalid polygon with buffer(0): {context}")
            geom = repaired
    if geom.is_empty or geom.area <= 0.0:
        return None
    return geom


def node_type_value(node: Dict[str, Any]) -> str:
    for scope in nested_dicts(node):
        for key in ("type", "node_type", "layer"):
            value = scope.get(key)
            if isinstance(value, str) and value.strip():
                upper = value.strip().upper()
                if "ROOM" in upper or "REGION" in upper:
                    return value.strip()
    return str(
        node.get("type") or node.get("node_type") or node.get("layer") or "unknown"
    )


def looks_like_region_node(node: Dict[str, Any]) -> bool:
    for scope in nested_dicts(node):
        for key in ("type", "node_type", "layer"):
            value = scope.get(key)
            if isinstance(value, str):
                upper = value.strip().upper()
                if "ROOM" in upper or "REGION" in upper:
                    return True
    return False


def iter_graph_nodes(graph: Any) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    if isinstance(graph, dict):
        nodes = graph.get("nodes")
        if isinstance(nodes, list):
            return [
                (str(n.get("id", i)), n)
                for i, n in enumerate(nodes)
                if isinstance(n, dict)
            ], "top-level nodes list"
        if isinstance(nodes, dict):
            return [(str(k), v) for k, v in nodes.items() if isinstance(v, dict)], (
                "top-level nodes dict"
            )

        node_like = [
            (str(k), v)
            for k, v in graph.items()
            if isinstance(v, dict) and looks_like_region_node(v)
        ]
        if node_like:
            return node_like, "top-level dict of node-like entries"

    if isinstance(graph, list):
        return [(str(i), n) for i, n in enumerate(graph) if isinstance(n, dict)], (
            "top-level list"
        )

    return [], "unrecognized graph layout"


def iter_graph_edges(graph: Any) -> List[Dict[str, Any]]:
    if isinstance(graph, dict):
        edges = graph.get("edges")
        if isinstance(edges, list):
            return [e for e in edges if isinstance(e, dict)]
        if isinstance(edges, dict):
            return [e for e in edges.values() if isinstance(e, dict)]
    return []


def candidate_type_counts(
    node_entries: Sequence[Tuple[str, Dict[str, Any]]],
) -> Counter:
    counts: Counter = Counter()
    for _, node in node_entries:
        ntype = node_type_value(node).upper()
        if "ROOM" in ntype:
            counts["ROOM"] += 1
        if "REGION" in ntype:
            counts["REGION"] += 1
    return counts


def get_polygon_payload(
    node: Dict[str, Any], keys: Sequence[str]
) -> Tuple[Optional[Any], Optional[str]]:
    for scope_name in ("attributes", "geometry", "geometry_signature", "root"):
        scope = node if scope_name == "root" else node.get(scope_name)
        if not isinstance(scope, dict):
            continue
        for key in keys:
            if key in scope and scope[key] not in (None, ""):
                return scope[key], f"{scope_name}.{key}"

    for scope in nested_dicts(node):
        token = parse_literal_dict(scope.get("geometry_refresh_token"))
        if token:
            for key in keys:
                if key in token and token[key] not in (None, ""):
                    return token[key], f"geometry_refresh_token.{key}"

    return None, None


def extract_adjacency_values(node: Dict[str, Any]) -> Set[str]:
    values: Set[str] = set()
    for scope in nested_dicts(node):
        for key in ADJACENCY_KEYS:
            raw = scope.get(key)
            if isinstance(raw, (list, tuple, set)):
                values.update(str(v) for v in raw)
            elif raw not in (None, ""):
                values.add(str(raw))
    return values


def load_predicted_regions(
    graph_json: Path,
    plane: str,
    warnings: List[str],
) -> Tuple[List[PredRegion], Set[Tuple[str, str]], Dict[str, Any]]:
    with graph_json.open("r", encoding="utf-8") as infile:
        graph = json.load(infile)

    node_entries, layout = iter_graph_nodes(graph)
    preds: List[PredRegion] = []
    id_to_pred: Dict[str, PredRegion] = {}
    stable_to_node: Dict[str, str] = {}
    missing_geometry = 0

    for fallback_id, node in node_entries:
        if not looks_like_region_node(node):
            continue

        node_id = str(node.get("id", fallback_id))
        ntype = node_type_value(node)
        label_value = first_nested_value(node, LABEL_KEYS)
        stable = first_nested_value(node, STABLE_ID_KEYS)

        raw_polygon, polygon_source = get_polygon_payload(node, POLYGON_KEYS)
        points = extract_points(raw_polygon, plane=plane)
        source = polygon_source
        if not points:
            raw_polygon, polygon_source = get_polygon_payload(node, HULL_KEYS)
            points = extract_points(raw_polygon, plane=plane)
            source = polygon_source
            if points:
                warn(
                    warnings,
                    f"Predicted region {node_id} uses exported convex hull geometry",
                )

        geom = make_polygon(
            points,
            warnings,
            context=f"predicted node {node_id}",
            allow_convex=True,
        )

        centroid = None
        for scope in nested_dicts(node):
            for key in CENTROID_KEYS:
                if key in scope:
                    centroid = extract_xy(scope[key], plane=plane)
                    if centroid is not None:
                        break
            if centroid is not None:
                break
        if centroid is None and geom is not None:
            centroid = (float(geom.centroid.x), float(geom.centroid.y))

        area = to_float(first_nested_value(node, AREA_KEYS))
        if area is None and geom is not None:
            area = float(geom.area)

        diagnostics: List[str] = []
        if geom is None:
            missing_geometry += 1
            diagnostics.append("missing polygon/convex_hull; IoU not computed")

        pred = PredRegion(
            node_id=node_id,
            node_type=ntype,
            label=str(label_value) if label_value is not None else None,
            stable_region_id=str(stable) if stable is not None else None,
            polygon=geom,
            polygon_source=source,
            centroid=centroid,
            area=area,
            adjacent_ids=extract_adjacency_values(node),
            raw=node,
            diagnostics=diagnostics,
        )
        preds.append(pred)
        id_to_pred[node_id] = pred
        if pred.stable_region_id is not None:
            stable_to_node[pred.stable_region_id] = node_id

    adjacency = build_predicted_adjacency_from_graph(graph, id_to_pred, stable_to_node)
    explicit_adj_count = 0
    for pred in preds:
        for neighbor in pred.adjacent_ids:
            left = pred.node_id
            right = id_to_pred.get(neighbor)
            if right is None and neighbor in stable_to_node:
                right = id_to_pred.get(stable_to_node[neighbor])
            if right is not None:
                adjacency.add(tuple(sorted((left, right.node_id))))
                explicit_adj_count += 1

    if missing_geometry:
        warn(
            warnings,
            f"{missing_geometry} predicted room/region candidates lack polygon geometry",
        )

    # Extract earliest agent/pose node position for frame-alignment seeding.
    first_pose: Optional[Tuple[float, float]] = None
    agent_candidates: List[Tuple[str, Tuple[float, float]]] = []
    for nid, node in node_entries:
        ntype = ""
        for scope in nested_dicts(node):
            val = scope.get("type") or scope.get("node_type", "")
            if isinstance(val, str) and val.strip():
                ntype = val.strip().upper()
                break
        if "AGENT" not in ntype and "POSE" not in ntype:
            continue
        pos: Optional[Tuple[float, float]] = None
        for scope in nested_dicts(node):
            for key in ("position", "translation", "pose"):
                raw = scope.get(key)
                if raw is not None:
                    pos = extract_xy(raw, plane=plane)
                    if pos is not None:
                        break
            if pos is not None:
                break
        if pos is not None:
            agent_candidates.append((nid, pos))
    if agent_candidates:
        try:
            agent_candidates.sort(key=lambda item: int(item[0]))
        except (ValueError, TypeError):
            agent_candidates.sort()
        first_pose = agent_candidates[0][1]

    metadata = {
        "graph_layout": layout,
        "top_level_keys": list(graph.keys()) if isinstance(graph, dict) else [],
        "candidate_type_counts": dict(candidate_type_counts(node_entries)),
        "edge_type_counts": dict(
            Counter(e.get("type") for e in iter_graph_edges(graph))
        ),
        "explicit_predicted_adjacency_edges": len(adjacency),
        "explicit_adjacency_field_hits": explicit_adj_count,
        "first_pose_xy": list(first_pose) if first_pose is not None else None,
    }
    return preds, adjacency, metadata


def build_predicted_adjacency_from_graph(
    graph: Any,
    id_to_pred: Dict[str, PredRegion],
    stable_to_node: Dict[str, str],
) -> Set[Tuple[str, str]]:
    adjacency: Set[Tuple[str, str]] = set()
    for edge in iter_graph_edges(graph):
        etype = str(
            edge.get("type") or edge.get("edge_type") or edge.get("label") or ""
        ).upper()
        if "ADJAC" not in etype and etype not in {"NEIGHBOR", "NEIGHBOUR"}:
            continue
        source = edge.get("source", edge.get("src", edge.get("from")))
        target = edge.get("target", edge.get("dst", edge.get("to")))
        if source is None or target is None:
            continue
        left = resolve_pred_id(str(source), id_to_pred, stable_to_node)
        right = resolve_pred_id(str(target), id_to_pred, stable_to_node)
        if left is not None and right is not None and left != right:
            adjacency.add(tuple(sorted((left, right))))
    return adjacency


def resolve_pred_id(
    raw_id: str,
    id_to_pred: Dict[str, PredRegion],
    stable_to_node: Dict[str, str],
) -> Optional[str]:
    if raw_id in id_to_pred:
        return raw_id
    if raw_id in stable_to_node:
        return stable_to_node[raw_id]
    return None


def read_text_from_dir_or_zip(
    scan_root: Path, subdir: str, suffix: str
) -> Tuple[str, str]:
    directory = scan_root / subdir
    if directory.is_dir():
        matches = sorted(directory.glob(f"*{suffix}"))
        if matches:
            return matches[0].read_text(encoding="utf-8"), str(matches[0])

    zpath = scan_root / f"{subdir}.zip"
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zfile:
            matches = sorted(name for name in zfile.namelist() if name.endswith(suffix))
            if matches:
                return zfile.read(matches[0]).decode("utf-8"), f"{zpath}:{matches[0]}"

    raise FileNotFoundError(f"Could not locate *{suffix} in {directory} or {zpath}")


def locate_graph_file(results_root: Path) -> Path:
    direct = results_root / "scene_graph.json"
    if direct.exists():
        return direct

    matches = sorted(results_root.rglob("scene_graph.json"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate scene_graph.json under {results_root}")


def parse_house(
    scan_root: Path,
    scan_id: str,
    plane: str,
    warnings: List[str],
) -> Tuple[List[GtRegion], Set[Tuple[int, int]], Dict[str, Any]]:
    house_text, house_source = read_text_from_dir_or_zip(
        scan_root, "house_segmentations", ".house"
    )
    regions: Dict[int, Dict[str, Any]] = {}
    surfaces_by_region: Dict[int, List[int]] = defaultdict(list)
    surface_info: Dict[int, Dict[str, Any]] = {}
    vertices_by_surface: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    portals: Set[Tuple[int, int]] = set()

    for line in house_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        record = parts[0]
        if record == "R" and len(parts) >= 16:
            region_index = int(parts[1])
            bbox = tuple(float(v) for v in parts[9:15])
            centroid = project_coord(
                {"x": float(parts[6]), "y": float(parts[7]), "z": float(parts[8])},
                plane,
            )
            regions[region_index] = {
                "region_index": region_index,
                "level_index": int(parts[2]),
                "label_code": parts[5],
                "centroid": centroid,
                "bbox": bbox,
                "height": to_float(parts[15]),
            }
        elif record == "S" and len(parts) >= 18:
            surface_index = int(parts[1])
            region_index = int(parts[2])
            normal = (float(parts[8]), float(parts[9]), float(parts[10]))
            surface_info[surface_index] = {
                "region_index": region_index,
                "label": parts[4],
                "normal": normal,
            }
            surfaces_by_region[region_index].append(surface_index)
        elif record == "V" and len(parts) >= 7:
            surface_index = int(parts[2])
            xy = project_coord(
                {"x": float(parts[4]), "y": float(parts[5]), "z": float(parts[6])},
                plane,
            )
            if xy is not None:
                vertices_by_surface[surface_index].append(xy)
        elif record == "P" and len(parts) >= 4 and is_int_token(parts[1]):
            # Portal P records use integer portal/region ids. Panorama P records
            # use a UUID-like name in parts[1] and are ignored here.
            if is_int_token(parts[2]) and is_int_token(parts[3]):
                left, right = int(parts[2]), int(parts[3])
                if left != right:
                    portals.add(tuple(sorted((left, right))))

    gt_regions: List[GtRegion] = []
    source_counter: Counter = Counter()

    for region_index in sorted(regions):
        region = regions[region_index]
        polygons = []

        for surface_index in surfaces_by_region.get(region_index, []):
            info = surface_info[surface_index]
            normal = info["normal"]
            is_floor_like = info["label"].upper() == "F" or abs(normal[2]) > 0.85
            if not is_floor_like:
                continue
            geom = make_polygon(
                vertices_by_surface.get(surface_index, []),
                warnings,
                context=f"GT region {region_index} surface {surface_index}",
                allow_convex=False,
            )
            if geom is not None:
                polygons.append(geom)

        polygon_source = "house_surfaces"
        if polygons:
            geom = unary_union(polygons) if len(polygons) > 1 else polygons[0]
            if not geom.is_valid:
                geom = geom.buffer(0)
                warn(
                    warnings,
                    f"Repaired GT region {region_index} polygon with buffer(0)",
                )
        else:
            # Fallback: no floor-labelled surfaces found for this region.
            # data_organization.md states that region extents are prisms whose
            # horizontal cross-section is defined by the vertices of ALL associated
            # surfaces, not just floors.  When no floor-like surface is tagged,
            # approximate the footprint from the projected vertex hull of all
            # surfaces in the region.
            all_verts: List[Tuple[float, float]] = []
            for s_idx in surfaces_by_region.get(region_index, []):
                all_verts.extend(vertices_by_surface.get(s_idx, []))
            if all_verts:
                warn(
                    warnings,
                    f"GT region {region_index}: no floor-like surfaces found; "
                    "footprint approximated from all-surface vertex hull",
                )
                geom = make_polygon(
                    all_verts,
                    warnings,
                    context=f"GT region {region_index} all-surface hull",
                    allow_convex=True,
                )
                polygon_source = "house_all_surfaces_hull"
            else:
                geom = None
                polygon_source = "missing_house_footprint"

        if geom is None or geom.is_empty or geom.area <= 0.0:
            warn(warnings, f"Skipping GT region {region_index}; no valid footprint")
            continue

        source_counter[polygon_source] += 1
        centroid = (float(geom.centroid.x), float(geom.centroid.y))
        label_code = region["label_code"]
        gt_regions.append(
            GtRegion(
                region_index=region_index,
                level_index=region["level_index"],
                label_code=label_code,
                label_name=REGION_LABELS.get(label_code, f"label_{label_code}"),
                polygon=geom,
                polygon_source=polygon_source,
                centroid=centroid,
                area=float(geom.area),
                height=region["height"],
            )
        )

    metadata = {
        "house_source": house_source,
        "gt_region_records": len(regions),
        "gt_polygon_sources": dict(source_counter),
        "gt_polygon_source": "mixed"
        if len(source_counter) > 1
        else (next(iter(source_counter)) if source_counter else "none"),
        "portal_edges": len(portals),
        "scan_id": scan_id,
    }
    return gt_regions, portals, metadata


def compute_pairwise(
    preds: Sequence[PredRegion], gts: Sequence[GtRegion]
) -> List[Pairwise]:
    pairs: List[Pairwise] = []
    for pred in preds:
        if pred.polygon is None:
            continue
        for gt in gts:
            inter = pred.polygon.intersection(gt.polygon).area
            union = pred.polygon.union(gt.polygon).area
            iou = inter / union if union > 0.0 else 0.0
            pairs.append(
                Pairwise(
                    pred_id=pred.node_id,
                    gt_id=gt.region_index,
                    iou=float(iou),
                    intersection_area=float(inter),
                    union_area=float(union),
                )
            )
    return pairs


def match_regions(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    iou_threshold: float,
) -> Tuple[Dict[str, int], Dict[Tuple[str, int], Pairwise], str]:
    pair_lookup = {(p.pred_id, p.gt_id): p for p in pairwise}
    pred_ids = [p.node_id for p in preds if p.polygon is not None]
    gt_ids = [g.region_index for g in gts]
    matches: Dict[str, int] = {}

    if pred_ids and gt_ids and linear_sum_assignment is not None and np is not None:
        matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
        for i, pred_id in enumerate(pred_ids):
            for j, gt_id in enumerate(gt_ids):
                matrix[i, j] = pair_lookup.get(
                    (pred_id, gt_id), Pairwise(pred_id, gt_id, 0, 0, 0)
                ).iou
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            iou = matrix[row, col]
            if iou >= iou_threshold:
                matches[pred_ids[row]] = gt_ids[col]
        return matches, pair_lookup, "hungarian"

    sorted_pairs = sorted(pairwise, key=lambda p: p.iou, reverse=True)
    used_preds: Set[str] = set()
    used_gts: Set[int] = set()
    for pair in sorted_pairs:
        if pair.iou < iou_threshold:
            break
        if pair.pred_id in used_preds or pair.gt_id in used_gts:
            continue
        matches[pair.pred_id] = pair.gt_id
        used_preds.add(pair.pred_id)
        used_gts.add(pair.gt_id)
    return matches, pair_lookup, "greedy"


def sample_boundary_points(geom: Any, step: float) -> List[Any]:
    boundary = geom.boundary
    length = float(boundary.length)
    if length <= 0.0:
        return []
    count = max(1, int(math.ceil(length / max(step, 1e-6))))
    return [boundary.interpolate((i / count) * length) for i in range(count)]


def boundary_scores(
    pred_geom: Any, gt_geom: Any, tolerance: float, step: float
) -> Tuple[float, float, float]:
    pred_points = sample_boundary_points(pred_geom, step)
    gt_points = sample_boundary_points(gt_geom, step)
    if not pred_points or not gt_points:
        return 0.0, 0.0, 0.0
    gt_boundary = gt_geom.boundary
    pred_boundary = pred_geom.boundary
    pred_hits = sum(
        1 for point in pred_points if point.distance(gt_boundary) <= tolerance
    )
    gt_hits = sum(
        1 for point in gt_points if point.distance(pred_boundary) <= tolerance
    )
    precision = pred_hits / len(pred_points)
    recall = gt_hits / len(gt_points)
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if precision + recall > 0.0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def polygon_adjacency_gt(
    gts: Sequence[GtRegion], tolerance: float
) -> Set[Tuple[int, int]]:
    adjacency: Set[Tuple[int, int]] = set()
    for i, left in enumerate(gts):
        for right in gts[i + 1 :]:
            if left.level_index != right.level_index:
                continue
            if left.polygon.boundary.distance(right.polygon.boundary) <= tolerance:
                adjacency.add(tuple(sorted((left.region_index, right.region_index))))
    return adjacency


def polygon_adjacency_pred(
    preds: Sequence[PredRegion], tolerance: float
) -> Set[Tuple[str, str]]:
    adjacency: Set[Tuple[str, str]] = set()
    usable = [p for p in preds if p.polygon is not None]
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            if left.polygon.boundary.distance(right.polygon.boundary) <= tolerance:
                adjacency.add(tuple(sorted((left.node_id, right.node_id))))
    return adjacency


def overlap_counts(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pair_lookup: Dict[Tuple[str, int], Pairwise],
    min_overlap_area: float,
) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
    gt_to_preds: Dict[int, List[str]] = defaultdict(list)
    pred_to_gts: Dict[str, List[int]] = defaultdict(list)
    for pred in preds:
        if pred.polygon is None:
            continue
        for gt in gts:
            pair = pair_lookup.get((pred.node_id, gt.region_index))
            if pair and pair.iou > 0.0 and pair.intersection_area >= min_overlap_area:
                gt_to_preds[gt.region_index].append(pred.node_id)
                pred_to_gts[pred.node_id].append(gt.region_index)
    return gt_to_preds, pred_to_gts


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def zero_if_none(value: Optional[float]) -> float:
    """Return 0.0 for undefined match-dependent metrics.

    A scan with no matched regions should remain in aggregate comparisons as a
    zero-quality segmentation result, not as an omitted/NaN datum.
    """
    return float(value) if value is not None else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, with 0.0 when the denominator is absent."""
    return float(numerator / denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def classify_failure_mode(
    gt_count: int,
    pred_count: int,
    matched_count: int,
    gt_region_coverage: float,
    region_count_ratio: float,
) -> Tuple[str, bool]:
    """Categorize scene-level region failures independently of IoU quality.

    Count and coverage metrics make under-generation explicit: one good matched
    room cannot mask a method that failed to produce most GT regions.
    """
    severe = gt_count > 0 and (
        region_count_ratio < SEVERE_UNDERSEGMENTATION_THRESHOLD
        or gt_region_coverage < SEVERE_UNDERSEGMENTATION_THRESHOLD
    )
    if gt_count == 0:
        return "NO_GT_REGIONS", False
    if pred_count == 0:
        return "NO_PREDICTIONS", severe
    if matched_count == 0:
        return "NO_MATCHES", severe
    if severe:
        return "SEVERE_UNDER_SEGMENTATION", True
    if gt_region_coverage < 1.0:
        return "PARTIAL_MATCH_ONLY", False
    return "VALID_MATCHES", False


def compute_metrics(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
    pair_lookup: Dict[Tuple[str, int], Pairwise],
    explicit_pred_adjacency: Set[Tuple[str, str]],
    gt_portals: Set[Tuple[int, int]],
    args: argparse.Namespace,
    warnings: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Tuple[float, float, float]]]:
    matched_pairs = [
        pair_lookup[(pred_id, gt_id)] for pred_id, gt_id in matches.items()
    ]
    ious = [p.iou for p in matched_pairs]

    gt_ids = {g.region_index for g in gts}
    pred_ids = {p.node_id for p in preds}
    matched_gt_ids = set(matches.values())
    matched_pred_ids = set(matches.keys())
    gt_count = len(gts)
    pred_count = len(preds)
    matched_gt_count = len(matched_gt_ids)
    matched_pred_count = len(matched_pred_ids)
    gt_region_coverage = safe_ratio(matched_gt_count, gt_count)
    predicted_region_usefulness = safe_ratio(matched_pred_count, pred_count)
    region_count_ratio = safe_ratio(pred_count, gt_count)
    region_count_error = abs(pred_count - gt_count)
    region_count_error_normalized = safe_ratio(region_count_error, gt_count)
    missed_gt_region_count = gt_count - matched_gt_count
    unmatched_predicted_region_count = pred_count - matched_pred_count
    mean_iou_matched = safe_mean(ious)
    region_f1 = f1_score(predicted_region_usefulness, gt_region_coverage)
    # Penalized IoU over GT regions: sum IoU(match_i) / |GT|. Unmatched GT
    # regions contribute zero, so under-generation remains visible.
    mean_iou_gt_penalized = safe_ratio(sum(ious), gt_count)
    # Penalized IoU over matched pairs plus all unmatched GT and predicted
    # regions: sum IoU(match_i) / (|matches| + |unmatched GT| + |unmatched pred|).
    full_penalty_count = (
        len(matches) + missed_gt_region_count + unmatched_predicted_region_count
    )
    mean_iou_full_penalized = safe_ratio(sum(ious), full_penalty_count)
    failure_mode, severe_undersegmentation = classify_failure_mode(
        gt_count,
        pred_count,
        len(matches),
        gt_region_coverage,
        region_count_ratio,
    )

    boundary_by_pred: Dict[str, Tuple[float, float, float]] = {}
    for pred_id, gt_id in matches.items():
        pred = next(p for p in preds if p.node_id == pred_id)
        gt = next(g for g in gts if g.region_index == gt_id)
        boundary_by_pred[pred_id] = boundary_scores(
            pred.polygon,
            gt.polygon,
            args.boundary_tolerance,
            args.boundary_sample_step,
        )

    boundary_precisions = [v[0] for v in boundary_by_pred.values()]
    boundary_recalls = [v[1] for v in boundary_by_pred.values()]
    boundary_f1s = [v[2] for v in boundary_by_pred.values()]

    gt_to_preds, pred_to_gts = overlap_counts(
        preds, gts, pair_lookup, args.min_overlap_area
    )
    over_segmented = {
        str(gt_id): fragments
        for gt_id, fragments in sorted(gt_to_preds.items())
        if len(fragments) > 1
    }
    under_segmented = {
        pred_id: gt_regions
        for pred_id, gt_regions in sorted(pred_to_gts.items())
        if len(gt_regions) > 1
    }
    oversegmentation_rate = safe_ratio(len(over_segmented), gt_count)
    undersegmentation_rate = safe_ratio(len(under_segmented), pred_count)
    no_predicted_regions = gt_count > 0 and pred_count == 0
    no_valid_matches = gt_count > 0 and len(matches) == 0
    single_region_failure = gt_count > 1 and pred_count == 1
    region_count_collapse = (
        gt_count > 0 and region_count_ratio <= SEVERE_UNDERSEGMENTATION_THRESHOLD
    )

    gt_adj_source = "house_portals"
    gt_adjacency = set(gt_portals)
    if not gt_adjacency:
        gt_adj_source = "polygon_boundary_proximity"
        gt_adjacency = polygon_adjacency_gt(gts, args.adjacency_tolerance)

    pred_adj_source = "exported_graph_or_fields"
    pred_adjacency = set(explicit_pred_adjacency)
    if not pred_adjacency:
        pred_adj_source = "polygon_boundary_proximity"
        pred_adjacency = polygon_adjacency_pred(preds, args.adjacency_tolerance)
        warn(
            warnings,
            "No predicted adjacency fields/edges found; using polygon proximity",
        )

    pred_to_gt = matches
    pred_gt_space_adjacency: Set[Tuple[int, int]] = set()
    for left, right in pred_adjacency:
        if left in pred_to_gt and right in pred_to_gt:
            gt_left, gt_right = pred_to_gt[left], pred_to_gt[right]
            if gt_left != gt_right:
                pred_gt_space_adjacency.add(tuple(sorted((gt_left, gt_right))))

    adjacency_tp = len(pred_gt_space_adjacency & gt_adjacency)
    adjacency_fp = len(pred_gt_space_adjacency - gt_adjacency)
    adjacency_fn = len(gt_adjacency - pred_gt_space_adjacency)
    adjacency_precision = (
        adjacency_tp / (adjacency_tp + adjacency_fp)
        if adjacency_tp + adjacency_fp
        else 0.0
    )
    adjacency_recall = (
        adjacency_tp / (adjacency_tp + adjacency_fn)
        if adjacency_tp + adjacency_fn
        else 0.0
    )
    adjacency_f1 = (
        2.0
        * adjacency_precision
        * adjacency_recall
        / (adjacency_precision + adjacency_recall)
        if adjacency_precision + adjacency_recall
        else 0.0
    )

    metrics = {
        "region_iou": {
            "mean_iou_matched": mean_iou_matched,
            "mean_iou_gt_penalized": float(mean_iou_gt_penalized),
            "mean_iou_full_penalized": float(mean_iou_full_penalized),
            "num_gt_regions": gt_count,
            "num_pred_regions": pred_count,
            "num_pred_regions_with_polygons": sum(
                1 for p in preds if p.polygon is not None
            ),
            "num_matches": len(matches),
            "region_recall": float(gt_region_coverage),
            "region_precision": float(predicted_region_usefulness),
            "region_f1": float(region_f1),
            "gt_coverage": float(gt_region_coverage),
            "region_count_ratio": float(region_count_ratio),
            "region_count_error_abs": region_count_error,
            "region_count_error_rel": float(region_count_error_normalized),
            "num_unmatched_gt": missed_gt_region_count,
            "num_unmatched_pred": unmatched_predicted_region_count,
            "unmatched_gt_regions": sorted(gt_ids - matched_gt_ids),
            "unmatched_predicted_regions": sorted(pred_ids - matched_pred_ids),
            "oversegmentation_rate": float(oversegmentation_rate),
            "undersegmentation_rate": float(undersegmentation_rate),
            "no_predicted_regions": no_predicted_regions,
            "no_valid_matches": no_valid_matches,
            "single_region_failure": single_region_failure,
            "region_count_collapse": region_count_collapse,
            "failure_mode": failure_mode,
            "severe_undersegmentation": severe_undersegmentation,
        },
        "boundary_f1": {
            # Boundary metrics are matched-only geometry scores. They are zero
            # for no-match scans so aggregation preserves the failure case.
            "mean_boundary_precision": zero_if_none(safe_mean(boundary_precisions)),
            "mean_boundary_recall": zero_if_none(safe_mean(boundary_recalls)),
            "mean_boundary_f1": zero_if_none(safe_mean(boundary_f1s)),
            "boundary_tolerance": args.boundary_tolerance,
            "boundary_sample_step": args.boundary_sample_step,
        },
        "over_segmentation": {
            "num_over_segmented_gt_regions": len(over_segmented),
            "mean_predicted_fragments_per_gt_region": safe_mean(
                [float(len(gt_to_preds.get(g.region_index, []))) for g in gts]
            ),
            "affected_gt_region_ids": sorted(int(k) for k in over_segmented.keys()),
            "details": over_segmented,
        },
        "under_segmentation": {
            "num_merged_predicted_regions": len(under_segmented),
            "mean_gt_regions_per_predicted_region": safe_mean(
                [
                    float(len(pred_to_gts.get(p.node_id, [])))
                    for p in preds
                    if p.polygon is not None
                ]
            ),
            "affected_predicted_region_ids": sorted(under_segmented.keys()),
            "details": under_segmented,
        },
        "region_adjacency_f1": {
            "adjacency_tp": adjacency_tp,
            "adjacency_fp": adjacency_fp,
            "adjacency_fn": adjacency_fn,
            "region_adjacency_precision": float(adjacency_precision),
            "region_adjacency_recall": float(adjacency_recall),
            "region_adjacency_f1": float(adjacency_f1),
            "gt_adjacency_source": gt_adj_source,
            "predicted_adjacency_source": pred_adj_source,
            "num_gt_adjacency_edges": len(gt_adjacency),
            "num_predicted_gt_space_adjacency_edges": len(pred_gt_space_adjacency),
        },
    }

    if len(matches) == 0 and preds and gts:
        warn(
            warnings,
            "No regions met the IoU threshold. Check coordinate frame alignment and plane selection.",
        )

    return metrics, boundary_by_pred


PAIRWISE_FIELDS = (
    "pred_node_id",
    "pred_type",
    "pred_stable_region_id",
    "pred_area",
    "pred_centroid_x",
    "pred_centroid_y",
    "gt_region_index",
    "gt_label_code",
    "gt_label_name",
    "gt_area",
    "gt_centroid_x",
    "gt_centroid_y",
    "iou",
    "intersection_area",
    "union_area",
    "centroid_distance",
    "matched",
)


def _build_pairwise_rows(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Build one row per (pred, GT) pair for the full N×M pairwise output."""
    pred_by_id = {p.node_id: p for p in preds}
    gt_by_id = {g.region_index: g for g in gts}
    match_set = set(matches.items())
    rows: List[Dict[str, Any]] = []
    for pair in pairwise:
        pred = pred_by_id.get(pair.pred_id)
        gt = gt_by_id.get(pair.gt_id)
        cdist: Optional[float] = None
        if pred is not None and pred.centroid is not None and gt is not None:
            cdist = math.hypot(
                pred.centroid[0] - gt.centroid[0],
                pred.centroid[1] - gt.centroid[1],
            )
        rows.append({
            "pred_node_id": pair.pred_id,
            "pred_type": pred.node_type if pred else "",
            "pred_stable_region_id": pred.stable_region_id if pred else "",
            "pred_area": pred.area if pred and pred.area is not None else "",
            "pred_centroid_x": pred.centroid[0] if pred and pred.centroid else "",
            "pred_centroid_y": pred.centroid[1] if pred and pred.centroid else "",
            "gt_region_index": pair.gt_id,
            "gt_label_code": gt.label_code if gt else "",
            "gt_label_name": gt.label_name if gt else "",
            "gt_area": gt.area if gt else "",
            "gt_centroid_x": gt.centroid[0] if gt else "",
            "gt_centroid_y": gt.centroid[1] if gt else "",
            "iou": pair.iou,
            "intersection_area": pair.intersection_area,
            "union_area": pair.union_area,
            "centroid_distance": cdist if cdist is not None else "",
            "matched": (pair.pred_id, pair.gt_id) in match_set,
        })
    return rows


def write_pairwise_csv(
    path: Path,
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
) -> None:
    """Write the full N×M pairwise overlap table to a CSV file."""
    rows = _build_pairwise_rows(preds, gts, pairwise, matches)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=PAIRWISE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_json(
    path: Path,
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
    scan_id: str,
) -> None:
    """Write the full N×M pairwise overlap table to a JSON file."""
    rows = _build_pairwise_rows(preds, gts, pairwise, matches)
    payload = {
        "scan_id": scan_id,
        "num_pred_regions": len(preds),
        "num_gt_regions": len(gts),
        "num_pairs": len(rows),
        "num_matched_pairs": len(matches),
        "fields": list(PAIRWISE_FIELDS),
        "pairs": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_matches_csv(
    path: Path,
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    matches: Dict[str, int],
    pair_lookup: Dict[Tuple[str, int], Pairwise],
    boundary_by_pred: Dict[str, Tuple[float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_by_id = {p.node_id: p for p in preds}
    gt_by_id = {g.region_index: g for g in gts}
    fieldnames = (
        "pred_node_id",
        "pred_type",
        "pred_stable_region_id",
        "gt_region_index",
        "gt_label_code",
        "gt_label_name",
        "iou",
        "intersection_area",
        "union_area",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "matched",
        "reason",
    )
    rows: List[Dict[str, Any]] = []

    for pred_id, gt_id in sorted(matches.items(), key=lambda item: (item[1], item[0])):
        pred = pred_by_id[pred_id]
        gt = gt_by_id[gt_id]
        pair = pair_lookup[(pred_id, gt_id)]
        bp, br, bf = boundary_by_pred.get(pred_id, (None, None, None))
        rows.append(
            {
                "pred_node_id": pred.node_id,
                "pred_type": pred.node_type,
                "pred_stable_region_id": pred.stable_region_id,
                "gt_region_index": gt.region_index,
                "gt_label_code": gt.label_code,
                "gt_label_name": gt.label_name,
                "iou": pair.iou,
                "intersection_area": pair.intersection_area,
                "union_area": pair.union_area,
                "boundary_precision": bp,
                "boundary_recall": br,
                "boundary_f1": bf,
                "matched": True,
                "reason": "matched",
            }
        )

    matched_preds = set(matches.keys())
    matched_gts = set(matches.values())
    for pred in preds:
        if pred.node_id in matched_preds:
            continue
        reason = (
            "; ".join(pred.diagnostics) if pred.diagnostics else "unmatched_predicted"
        )
        rows.append(
            {
                "pred_node_id": pred.node_id,
                "pred_type": pred.node_type,
                "pred_stable_region_id": pred.stable_region_id,
                "gt_region_index": "",
                "gt_label_code": "",
                "gt_label_name": "",
                "iou": "",
                "intersection_area": "",
                "union_area": "",
                "boundary_precision": "",
                "boundary_recall": "",
                "boundary_f1": "",
                "matched": False,
                "reason": reason,
            }
        )

    for gt in gts:
        if gt.region_index in matched_gts:
            continue
        rows.append(
            {
                "pred_node_id": "",
                "pred_type": "",
                "pred_stable_region_id": "",
                "gt_region_index": gt.region_index,
                "gt_label_code": gt.label_code,
                "gt_label_name": gt.label_name,
                "iou": "",
                "intersection_area": "",
                "union_area": "",
                "boundary_precision": "",
                "boundary_recall": "",
                "boundary_f1": "",
                "matched": False,
                "reason": "unmatched_gt",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_region_summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the flat per-scan metric row used by aggregate CSV tables."""
    region_iou = summary["metrics"]["region_iou"]
    return {
        "scan_id": summary["scan_id"],
        "method": summary.get("method", "mine"),
        "num_gt_regions": region_iou["num_gt_regions"],
        "num_pred_regions": region_iou["num_pred_regions"],
        "num_matches": region_iou["num_matches"],
        "num_unmatched_gt": region_iou["num_unmatched_gt"],
        "num_unmatched_pred": region_iou["num_unmatched_pred"],
        "region_count_error_abs": region_iou["region_count_error_abs"],
        "region_count_error_rel": region_iou["region_count_error_rel"],
        "region_count_ratio": region_iou["region_count_ratio"],
        "region_precision": region_iou["region_precision"],
        "region_recall": region_iou["region_recall"],
        "region_f1": region_iou["region_f1"],
        "gt_coverage": region_iou["gt_coverage"],
        "mean_iou_matched": region_iou["mean_iou_matched"],
        "mean_iou_gt_penalized": region_iou["mean_iou_gt_penalized"],
        "mean_iou_full_penalized": region_iou["mean_iou_full_penalized"],
        "oversegmentation_rate": region_iou["oversegmentation_rate"],
        "undersegmentation_rate": region_iou["undersegmentation_rate"],
        "no_predicted_regions": region_iou["no_predicted_regions"],
        "no_valid_matches": region_iou["no_valid_matches"],
        "single_region_failure": region_iou["single_region_failure"],
        "severe_undersegmentation": region_iou["severe_undersegmentation"],
        "region_count_collapse": region_iou["region_count_collapse"],
        "failure_mode": region_iou["failure_mode"],
        "source_file": summary["graph_json"],
    }


def write_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=CANONICAL_REGION_FIELDS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerow(row)


def print_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    region_iou = metrics["region_iou"]
    boundary = metrics["boundary_f1"]
    over = metrics["over_segmentation"]
    under = metrics["under_segmentation"]
    adj = metrics["region_adjacency_f1"]

    print("\nMatterport3D Region Evaluation")
    print("==============================")
    print(f"scan_id: {summary['scan_id']}")
    print(f"gt regions: {region_iou['num_gt_regions']}")
    print(f"predicted regions: {region_iou['num_pred_regions']}")
    print(f"matched regions: {region_iou['num_matches']}")
    print(
        "unmatched GT / unmatched predicted: "
        f"{region_iou['num_unmatched_gt']} / {region_iou['num_unmatched_pred']}"
    )
    print(
        "GT coverage / precision / count ratio: "
        f"{fmt(region_iou['gt_coverage'])} / "
        f"{fmt(region_iou['region_precision'])} / "
        f"{fmt(region_iou['region_count_ratio'])}"
    )
    print(
        "region precision/recall/f1: "
        f"{fmt(region_iou['region_precision'])} / "
        f"{fmt(region_iou['region_recall'])} / "
        f"{fmt(region_iou['region_f1'])}"
    )
    print(
        "GT/full penalized IoU: "
        f"{fmt(region_iou['mean_iou_gt_penalized'])} / "
        f"{fmt(region_iou['mean_iou_full_penalized'])}"
    )
    print(f"matched-only mean IoU: {fmt(region_iou['mean_iou_matched'])}")
    print(
        f"boundary precision/recall/f1: {fmt(boundary['mean_boundary_precision'])} / {fmt(boundary['mean_boundary_recall'])} / {fmt(boundary['mean_boundary_f1'])}"
    )
    print(f"over_segmentation: {over['num_over_segmented_gt_regions']} GT regions")
    print(
        f"under_segmentation: {under['num_merged_predicted_regions']} predicted regions"
    )
    print(
        "failure mode / severe undersegmentation: "
        f"{region_iou['failure_mode']} / {region_iou['severe_undersegmentation']}"
    )
    print(
        f"region_adjacency precision/recall/f1: {fmt(adj['region_adjacency_precision'])} / {fmt(adj['region_adjacency_recall'])} / {fmt(adj['region_adjacency_f1'])}"
    )
    if summary.get("warnings"):
        print("\nWarnings:")
        for message in summary["warnings"]:
            print(f"  - {message}")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# ─────────────────────────── Overlap diagnostics ────────────────────────────


def _poly_bounds(polygon: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (minx, miny, maxx, maxy) from a Shapely polygon or any object with .bounds."""
    if polygon is None:
        return None
    try:
        b = polygon.bounds
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except Exception:
        return None


def _global_bounds(
    polygons: Iterable[Any],
) -> Optional[Tuple[float, float, float, float]]:
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    found = False
    for p in polygons:
        b = _poly_bounds(p)
        if b is None:
            continue
        found = True
        lo_x = min(lo_x, b[0])
        lo_y = min(lo_y, b[1])
        hi_x = max(hi_x, b[2])
        hi_y = max(hi_y, b[3])
    return (lo_x, lo_y, hi_x, hi_y) if found else None


def _bounds_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def compute_overlap_diagnostics(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    top_n: int = 5,
) -> Dict[str, Any]:
    """Return unthresholded overlap diagnostics for all pred/GT pairs.

    These values are stored in the JSON output and printed separately.  They
    are computed independently of ``--iou_threshold`` and must not be used as
    valid matches — they exist solely to diagnose why the thresholded matching
    produced zero (or few) results.
    """
    pred_by_id = {p.node_id: p for p in preds}
    gt_by_id = {g.region_index: g for g in gts}

    best_for_pred: Dict[str, Pairwise] = {}
    best_for_gt: Dict[int, Pairwise] = {}
    for pair in pairwise:
        prev = best_for_pred.get(pair.pred_id)
        if prev is None or pair.iou > prev.iou:
            best_for_pred[pair.pred_id] = pair
        prev_g = best_for_gt.get(pair.gt_id)
        if prev_g is None or pair.iou > prev_g.iou:
            best_for_gt[pair.gt_id] = pair

    max_iou = max((p.iou for p in pairwise), default=0.0)
    max_intersection = max((p.intersection_area for p in pairwise), default=0.0)
    intersection_sum = sum(p.intersection_area for p in pairwise)

    gt_bounds = _global_bounds(g.polygon for g in gts)
    pred_bounds = _global_bounds(p.polygon for p in preds if p.polygon is not None)
    gt_total_area = float(sum(g.area for g in gts))
    pred_total_area = float(sum(p.area for p in preds if p.area is not None))
    bounds_overlap = (
        _bounds_overlap(gt_bounds, pred_bounds)
        if gt_bounds is not None and pred_bounds is not None
        else False
    )

    min_centroid_dist: Optional[float] = None
    for pred in preds:
        if pred.centroid is None:
            continue
        for gt in gts:
            d = math.hypot(
                pred.centroid[0] - gt.centroid[0],
                pred.centroid[1] - gt.centroid[1],
            )
            if min_centroid_dist is None or d < min_centroid_dist:
                min_centroid_dist = d

    def _cdist(pred: PredRegion, gt: GtRegion) -> Optional[float]:
        if pred.centroid is None:
            return None
        return float(math.hypot(
            pred.centroid[0] - gt.centroid[0],
            pred.centroid[1] - gt.centroid[1],
        ))

    best_pred_to_gt = []
    for pair in sorted(best_for_pred.values(), key=lambda p: p.iou, reverse=True)[:top_n]:
        pred = pred_by_id.get(pair.pred_id)
        gt = gt_by_id.get(pair.gt_id)
        pb = _poly_bounds(pred.polygon) if pred else None
        gb = _poly_bounds(gt.polygon) if gt else None
        best_pred_to_gt.append({
            "pred_id": pair.pred_id,
            "gt_id": pair.gt_id,
            "iou": float(pair.iou),
            "intersection_area": float(pair.intersection_area),
            "union_area": float(pair.union_area),
            "pred_area": float(pred.area) if pred and pred.area is not None else None,
            "gt_area": float(gt.area) if gt else None,
            "centroid_distance": _cdist(pred, gt) if pred and gt else None,
            "pred_bounds": list(pb) if pb is not None else None,
            "gt_bounds": list(gb) if gb is not None else None,
        })

    best_gt_to_pred = []
    for pair in sorted(best_for_gt.values(), key=lambda p: p.iou, reverse=True)[:top_n]:
        pred = pred_by_id.get(pair.pred_id)
        gt = gt_by_id.get(pair.gt_id)
        best_gt_to_pred.append({
            "gt_id": pair.gt_id,
            "pred_id": pair.pred_id,
            "iou": float(pair.iou),
            "intersection_area": float(pair.intersection_area),
            "union_area": float(pair.union_area),
            "centroid_distance": _cdist(pred, gt) if pred and gt else None,
        })

    return {
        "max_iou": float(max_iou),
        "max_intersection_area": float(max_intersection),
        "pairwise_intersection_sum": float(intersection_sum),
        "min_centroid_distance": float(min_centroid_dist) if min_centroid_dist is not None else None,
        "gt_bounds": list(gt_bounds) if gt_bounds is not None else None,
        "pred_bounds": list(pred_bounds) if pred_bounds is not None else None,
        "gt_total_area": gt_total_area,
        "pred_total_area": pred_total_area,
        "bounds_overlap": bounds_overlap,
        "best_pred_to_gt": best_pred_to_gt,
        "best_gt_to_pred": best_gt_to_pred,
    }


def print_overlap_diagnostics(diag: Dict[str, Any], iou_threshold: float) -> None:
    """Print unthresholded pairwise and geometry diagnostics."""
    print("\nPairwise overlap diagnostics")
    print("----------------------------")
    print(f"max IoU (all pairs):       {fmt(diag['max_iou'])}")
    print(f"max intersection area:     {fmt(diag['max_intersection_area'])}")
    print(f"intersection sum:          {fmt(diag['pairwise_intersection_sum'])}")
    print(f"min centroid distance:     {fmt(diag['min_centroid_distance'])}")
    print(f"IoU threshold:             {iou_threshold}")

    print("\nbest pred->GT pairs (unthresholded, diagnostic only):")
    if diag["best_pred_to_gt"]:
        for entry in diag["best_pred_to_gt"]:
            print(
                f"  pred {str(entry['pred_id']):>6} -> GT {str(entry['gt_id']):>3}: "
                f"IoU={fmt(entry['iou'])}, inter={fmt(entry['intersection_area'])}, "
                f"union={fmt(entry['union_area'])}, "
                f"pred_area={fmt(entry['pred_area'])}, gt_area={fmt(entry['gt_area'])}, "
                f"cdist={fmt(entry['centroid_distance'])}"
            )
    else:
        print("  (no predicted regions with valid polygons)")

    print("\nbest GT->pred pairs (unthresholded, diagnostic only):")
    if diag["best_gt_to_pred"]:
        for entry in diag["best_gt_to_pred"]:
            print(
                f"  GT {str(entry['gt_id']):>3} -> pred {str(entry['pred_id']):>6}: "
                f"IoU={fmt(entry['iou'])}, inter={fmt(entry['intersection_area'])}, "
                f"cdist={fmt(entry['centroid_distance'])}"
            )
    else:
        print("  (none)")

    gt_b = diag["gt_bounds"]
    pred_b = diag["pred_bounds"]
    print("\nGeometry diagnostics")
    print("--------------------")
    if gt_b:
        print(f"GT bounds:       x=[{gt_b[0]:.3f}, {gt_b[2]:.3f}]  y=[{gt_b[1]:.3f}, {gt_b[3]:.3f}]")
    else:
        print("GT bounds:       n/a (no valid GT polygons)")
    if pred_b:
        print(f"Pred bounds:     x=[{pred_b[0]:.3f}, {pred_b[2]:.3f}]  y=[{pred_b[1]:.3f}, {pred_b[3]:.3f}]")
    else:
        print("Pred bounds:     n/a (no valid predicted polygons)")
    print(f"GT total area:   {fmt(diag['gt_total_area'])}")
    print(f"Pred total area: {fmt(diag['pred_total_area'])}")
    print(f"Bounds overlap:  {'yes' if diag['bounds_overlap'] else 'NO'}")

    if pred_b is None or diag["pred_total_area"] == 0.0:
        issue = "missing/invalid predicted polygons"
    elif gt_b is None or diag["gt_total_area"] == 0.0:
        issue = "missing/invalid GT polygons"
    elif not diag["bounds_overlap"]:
        issue = "ALIGNMENT / FRAME MISMATCH — GT and predicted bounding boxes do not overlap at all"
    elif diag["pairwise_intersection_sum"] == 0.0:
        issue = "zero intersection everywhere — coordinate scale or translation offset within overlapping bounds"
    elif diag["max_iou"] < iou_threshold:
        issue = (
            f"all IoUs below threshold ({iou_threshold}) — "
            f"real undersegmentation or systematic shape mismatch (max IoU={fmt(diag['max_iou'])})"
        )
    else:
        issue = "threshold/assignment — at least one pair has IoU >= threshold; check matching"
    print(f"Likely issue:    {issue}")


def validate_run(
    gt_record_count: int,
    gts: Sequence[GtRegion],
    preds: Sequence[PredRegion],
    pairwise: Sequence[Pairwise],
    warnings: List[str],
) -> None:
    """Lightweight sanity checks recorded as warnings rather than exceptions."""
    usable_gt = len(gts)
    if usable_gt < gt_record_count:
        warn(
            warnings,
            f"Validation: {gt_record_count - usable_gt}/{gt_record_count} GT region records skipped (no valid footprint)",
        )

    pred_with_poly = sum(1 for p in preds if p.polygon is not None)
    if pred_with_poly < len(preds):
        warn(
            warnings,
            f"Validation: {len(preds) - pred_with_poly}/{len(preds)} predicted regions lack valid polygons",
        )

    expected_pairs = pred_with_poly * usable_gt
    if len(pairwise) != expected_pairs:
        warn(
            warnings,
            f"Validation: pairwise count {len(pairwise)} != expected {expected_pairs} "
            f"(pred_with_poly={pred_with_poly}, gt={usable_gt})",
        )

    bad = [
        (p.pred_id, p.gt_id, p.iou)
        for p in pairwise
        if not math.isfinite(p.iou) or p.iou < -1e-9 or p.iou > 1.0 + 1e-9
    ]
    if bad:
        warn(
            warnings,
            f"Validation: {len(bad)} pairwise entries have out-of-range IoU (first: {bad[0]})",
        )


def append_scan_id_to_output_path(path: Optional[Path], scan_id: str) -> Optional[Path]:
    """Return an output path with the scan id inserted before the suffix.

    Example:
      region_eval_summary.json -> region_eval_summary_zsNo4HB9uLZ.json

    If the caller already included the scan id in the stem, leave it unchanged
    so repeated runs do not create duplicated names.
    """
    if path is None:
        return None
    if scan_id in path.stem:
        return path
    return path.with_name(f"{path.stem}_{scan_id}{path.suffix}")


def dry_run_report(
    graph_meta: Dict[str, Any],
    preds: Sequence[PredRegion],
    gt_meta: Dict[str, Any],
    gts: Sequence[GtRegion],
) -> None:
    room_count = graph_meta.get("candidate_type_counts", {}).get("ROOM", 0)
    region_count = graph_meta.get("candidate_type_counts", {}).get("REGION", 0)
    sample_pred = next(
        (p for p in preds if p.polygon is not None), preds[0] if preds else None
    )
    sample_gt = gts[0] if gts else None

    print("Dry-run schema report")
    print("=====================")
    print(f"graph top-level keys: {graph_meta.get('top_level_keys', [])}")
    print(f"graph layout: {graph_meta.get('graph_layout')}")
    print(f"ROOM candidates: {room_count}")
    print(f"REGION candidates: {region_count}")
    print(f"predicted candidates total: {len(preds)}")
    if sample_pred:
        print("sample predicted polygon node:")
        print(
            json.dumps(
                {
                    "node_id": sample_pred.node_id,
                    "node_type": sample_pred.node_type,
                    "label": sample_pred.label,
                    "stable_region_id": sample_pred.stable_region_id,
                    "polygon_source": sample_pred.polygon_source,
                    "area": sample_pred.area,
                    "centroid": sample_pred.centroid,
                    "adjacent_ids": sorted(sample_pred.adjacent_ids),
                },
                indent=2,
            )
        )
    print(f"GT region count: {len(gts)}")
    if sample_gt:
        print("sample GT region:")
        print(
            json.dumps(
                {
                    "gt_region_index": sample_gt.region_index,
                    "level_index": sample_gt.level_index,
                    "label_code": sample_gt.label_code,
                    "label_name": sample_gt.label_name,
                    "polygon_source": sample_gt.polygon_source,
                    "area": sample_gt.area,
                    "centroid": sample_gt.centroid,
                },
                indent=2,
            )
        )
    print(f"GT polygon source: {gt_meta.get('gt_polygon_source')}")
    print(f"GT polygon sources: {gt_meta.get('gt_polygon_sources')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate scene graph room/region geometry against Matterport3D regions."
    )
    parser.add_argument("--scan_id", required=True)
    parser.add_argument(
        "--dsg_dir",
        type=Path,
        default=Path("."),
        help="Root directory containing 3DSG scene_graph.json outputs.",
    )
    parser.add_argument(
        "--mp3d_root",
        type=Path,
        default=Path("/mnt/DATA/repos/phd/3dsg/mp3d/dataset/v1/scans"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--method_name", default="mine")
    parser.add_argument("--dry_run_schema", action="store_true")
    parser.add_argument("--plane", default="xy", choices=("xy", "xz", "yz"))
    parser.add_argument("--iou_threshold", type=float, default=0.15)
    parser.add_argument("--boundary_tolerance", type=float, default=0.25)
    parser.add_argument("--boundary_sample_step", type=float, default=0.10)
    parser.add_argument("--min_overlap_area", type=float, default=0.1)
    parser.add_argument("--adjacency_tolerance", type=float, default=0.30)
    parser.add_argument(
        "--gt_tx",
        type=float,
        default=0.0,
        help=(
            "Translate GT polygons by this amount in the projected X axis before "
            "computing IoU. Use to compensate for a known offset between the MP3D "
            "house-file frame and the predicted coordinate frame."
        ),
    )
    parser.add_argument(
        "--gt_ty",
        type=float,
        default=0.0,
        help="Translate GT polygons by this amount in the projected Y axis (see --gt_tx).",
    )
    parser.add_argument(
        "--auto_align",
        action="store_true",
        help=(
            "Search for the GT-to-pred translation (and rotation when "
            "--align_try_rotations is set) that maximizes region overlap before "
            "computing IoU. Uses the first agent/pose node in the scene graph as "
            "the initial search center; falls back to the bounding-box centroid "
            "difference. Mutually exclusive with --gt_tx/--gt_ty."
        ),
    )
    parser.add_argument(
        "--align_radius",
        type=float,
        default=10.0,
        help="Translation search radius in metres for --auto_align (default: %(default)s).",
    )
    parser.add_argument(
        "--align_step",
        type=float,
        default=0.5,
        help="Grid step in metres for --auto_align (default: %(default)s).",
    )
    parser.add_argument(
        "--align_try_rotations",
        action="store_true",
        help="Also try 90/180/270-degree rotations of GT polygons during --auto_align.",
    )
    return parser


def translate_gt_regions(gts: List[GtRegion], tx: float, ty: float) -> None:
    """Shift all GT polygons by (tx, ty) in the projected plane.

    Used to compensate for a known translation offset between the MP3D house-file
    coordinate frame and the predicted/SLAM coordinate frame. Area is invariant
    under translation; centroid and polygon vertices are updated in-place.
    """
    if tx == 0.0 and ty == 0.0:
        return
    for gt in gts:
        if gt.polygon is not None:
            pts = [(x + tx, y + ty) for x, y in gt.polygon.exterior.coords[:-1]]
            geom = Polygon(pts)
            if not geom.is_valid:
                geom = geom.buffer(0)
            gt.polygon = geom
        gt.centroid = (gt.centroid[0] + tx, gt.centroid[1] + ty)


# ─────────────────────── Frame auto-alignment ────────────────────────────────


def _rotate_points(
    points: List[Tuple[float, float]], angle_deg: float, cx: float, cy: float
) -> List[Tuple[float, float]]:
    """Rotate a sequence of 2-D points around (cx, cy) by angle_deg degrees."""
    if angle_deg == 0.0:
        return list(points)
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + cos_a * dx - sin_a * dy, cy + sin_a * dx + cos_a * dy))
    return out


def _transform_shapely(
    geom: Any, tx: float, ty: float, angle_deg: float, cx: float, cy: float
) -> Any:
    """Rotate a shapely polygon around (cx, cy) then translate by (tx, ty)."""
    if geom is None or Polygon is None:
        return geom
    pts = _rotate_points(list(geom.exterior.coords[:-1]), angle_deg, cx, cy)
    pts = [(x + tx, y + ty) for x, y in pts]
    new_geom = Polygon(pts)
    if not new_geom.is_valid:
        new_geom = new_geom.buffer(0)
    return new_geom if not new_geom.is_empty else geom


def _alignment_score(
    pred_polys: List[Any],
    gt_polys: List[Any],
    tx: float,
    ty: float,
    angle_deg: float,
    cx: float,
    cy: float,
) -> float:
    """Sum of all pairwise intersection areas after rotating GT around (cx, cy) then translating."""
    total = 0.0
    for gt_base in gt_polys:
        try:
            gt_t = _transform_shapely(gt_base, tx, ty, angle_deg, cx, cy)
            if gt_t is None:
                continue
            for pred in pred_polys:
                if pred is not None:
                    total += float(pred.intersection(gt_t).area)
        except Exception:
            pass
    return total


def rotate_gt_regions(gts: List[GtRegion], angle_deg: float, cx: float, cy: float) -> None:
    """Rotate all GT polygon vertices around (cx, cy) in-place."""
    if angle_deg == 0.0:
        return
    for gt in gts:
        if gt.polygon is not None:
            pts = _rotate_points(list(gt.polygon.exterior.coords[:-1]), angle_deg, cx, cy)
            new_geom = Polygon(pts)
            if not new_geom.is_valid:
                new_geom = new_geom.buffer(0)
            gt.polygon = new_geom
        (gt.centroid,) = _rotate_points([gt.centroid], angle_deg, cx, cy)


def auto_align(
    preds: Sequence[PredRegion],
    gts: Sequence[GtRegion],
    initial_tx: float,
    initial_ty: float,
    radius: float,
    step: float,
    try_rotations: bool,
    warnings: List[str],
) -> Tuple[float, float, float, float]:
    """Grid-search the GT→pred transform that maximises total pairwise intersection area.

    Rotation (when enabled) is applied around the GT union centroid before translation.
    Returns (best_tx, best_ty, best_rotation_deg, best_overlap_score).
    """
    pred_polys = [p.polygon for p in preds if p.polygon is not None]
    gt_polys = [g.polygon for g in gts if g.polygon is not None]
    if not pred_polys or not gt_polys:
        warn(warnings, "auto_align: no valid polygons to search over.")
        return initial_tx, initial_ty, 0.0, 0.0

    gt_union = unary_union(gt_polys)
    cx = float(gt_union.centroid.x)
    cy = float(gt_union.centroid.y)

    rotations = [0.0, 90.0, 180.0, 270.0] if try_rotations else [0.0]
    n_steps = max(1, int(math.ceil(radius / max(step, 1e-9))))

    best_tx, best_ty, best_rot = initial_tx, initial_ty, 0.0
    best_score = _alignment_score(pred_polys, gt_polys, initial_tx, initial_ty, 0.0, cx, cy)

    for rot in rotations:
        for di in range(-n_steps, n_steps + 1):
            for dj in range(-n_steps, n_steps + 1):
                tx = initial_tx + di * step
                ty = initial_ty + dj * step
                sc = _alignment_score(pred_polys, gt_polys, tx, ty, rot, cx, cy)
                if sc > best_score:
                    best_score = sc
                    best_tx, best_ty, best_rot = tx, ty, rot

    return best_tx, best_ty, best_rot, best_score


def main() -> int:
    args = build_arg_parser().parse_args()
    scan_root = args.mp3d_root / args.scan_id
    results_root = args.dsg_dir / args.scan_id
    graph_json = locate_graph_file(results_root)

    output_csv = args.output_dir / f"region_eval_summary_{args.scan_id}.csv"
    output_json = args.output_dir / f"region_eval_summary_{args.scan_id}.json"

    if Polygon is None:
        print(
            "ERROR: shapely is required for polygon IoU, boundary, and adjacency metrics. "
            "Install shapely and rerun this script.",
            file=sys.stderr,
        )
        return 2

    warnings: List[str] = []
    if args.plane != "xy":
        warn(
            warnings, f"Plane {args.plane} requested; xy is the validated/default path"
        )

    preds, pred_adjacency, graph_meta = load_predicted_regions(
        graph_json, args.plane, warnings
    )
    gts, gt_portals, gt_meta = parse_house(
        scan_root,
        args.scan_id,
        args.plane,
        warnings,
    )
    # ── coordinate-frame alignment ────────────────────────────────────────────
    if args.auto_align:
        # Initial estimate A: first agent/pose node (negated — SLAM origin → MP3D)
        first_pose = graph_meta.get("first_pose_xy")
        pose_tx = -first_pose[0] if first_pose else 0.0
        pose_ty = -first_pose[1] if first_pose else 0.0

        # Initial estimate B: centroid difference between GT and pred bounding boxes
        pred_with_c = [p for p in preds if p.centroid is not None]
        if gts and pred_with_c:
            gt_cx_mean = statistics.mean(g.centroid[0] for g in gts)
            gt_cy_mean = statistics.mean(g.centroid[1] for g in gts)
            cent_tx = statistics.mean(p.centroid[0] for p in pred_with_c) - gt_cx_mean
            cent_ty = statistics.mean(p.centroid[1] for p in pred_with_c) - gt_cy_mean
        else:
            cent_tx = cent_ty = 0.0

        # Pick the candidate with the higher initial overlap score
        pred_polys = [p.polygon for p in preds if p.polygon is not None]
        gt_polys = [g.polygon for g in gts if g.polygon is not None]
        if pred_polys and gt_polys:
            gt_union = unary_union(gt_polys)
            pivot_cx = float(gt_union.centroid.x)
            pivot_cy = float(gt_union.centroid.y)
            score_pose = _alignment_score(pred_polys, gt_polys, pose_tx, pose_ty, 0.0, pivot_cx, pivot_cy)
            score_cent = _alignment_score(pred_polys, gt_polys, cent_tx, cent_ty, 0.0, pivot_cx, pivot_cy)
            if first_pose and score_pose >= score_cent:
                initial_tx, initial_ty = pose_tx, pose_ty
                init_source = f"first_pose={first_pose} score={score_pose:.3f}"
            else:
                initial_tx, initial_ty = cent_tx, cent_ty
                init_source = f"centroid_diff score={score_cent:.3f}"
        else:
            initial_tx, initial_ty = cent_tx, cent_ty
            init_source = "centroid_diff (no polygons to score)"

        tx, ty, rot, score = auto_align(
            preds, gts,
            initial_tx, initial_ty,
            args.align_radius, args.align_step, args.align_try_rotations,
            warnings,
        )
        warn(
            warnings,
            f"auto_align result: tx={tx:.3f} ty={ty:.3f} rot={rot:.1f}deg "
            f"overlap={score:.3f} (initial: {init_source})",
        )
        if rot != 0.0:
            gt_union = unary_union([g.polygon for g in gts if g.polygon is not None])
            rotate_gt_regions(gts, rot, float(gt_union.centroid.x), float(gt_union.centroid.y))
        translate_gt_regions(gts, tx, ty)

    elif args.gt_tx != 0.0 or args.gt_ty != 0.0:
        warn(
            warnings,
            f"GT polygons translated by ({args.gt_tx:.3f}, {args.gt_ty:.3f}) "
            "before IoU computation (--gt_tx / --gt_ty). Ensure this offset "
            "correctly maps the MP3D frame to the predicted frame.",
        )
        translate_gt_regions(gts, args.gt_tx, args.gt_ty)

    if args.dry_run_schema:
        dry_run_report(graph_meta, preds, gt_meta, gts)
        if warnings:
            print("warnings:")
            for message in warnings:
                print(f"  - {message}")
        return 0

    pairwise = compute_pairwise(preds, gts)

    validate_run(gt_meta["gt_region_records"], gts, preds, pairwise, warnings)

    matches, pair_lookup, assignment_method = match_regions(
        preds,
        gts,
        pairwise,
        args.iou_threshold,
    )

    metrics, boundary_by_pred = compute_metrics(
        preds,
        gts,
        pairwise,
        matches,
        pair_lookup,
        pred_adjacency,
        gt_portals,
        args,
        warnings,
    )

    overlap_diag = compute_overlap_diagnostics(preds, gts, pairwise)

    summary = {
        "scan_id": args.scan_id,
        "method": args.method_name,
        "graph_json": str(graph_json),
        "scan_root": str(scan_root),
        "plane": args.plane,
        "iou_threshold": args.iou_threshold,
        "assignment_method": assignment_method,
        "gt_polygon_source": gt_meta.get("gt_polygon_source"),
        "gt_metadata": gt_meta,
        "graph_metadata": graph_meta,
        "metrics": metrics,
        "overlap_diagnostics": overlap_diag,
        "warnings": warnings,
        "assumptions": [
            "Evaluation is performed in the projected ground plane.",
            "Matterport and scene graph coordinates are assumed to share the same metric global frame.",
            "mean_iou_matched is undefined (null) when there are no valid matches; penalized IoU and count metrics remain finite.",
            "overlap_diagnostics.best_pred_to_gt and best_gt_to_pred list the closest pairs by IoU regardless of threshold; they are NOT valid matches.",
        ],
    }
    summary["region_eval_summary"] = build_region_summary_row(summary)

    write_summary_csv(output_csv, summary["region_eval_summary"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    pairwise_csv = args.output_dir / f"region_eval_pairwise_{args.scan_id}.csv"
    pairwise_json = args.output_dir / f"region_eval_pairwise_{args.scan_id}.json"
    write_pairwise_csv(pairwise_csv, preds, gts, pairwise, matches)
    write_pairwise_json(pairwise_json, preds, gts, pairwise, matches, args.scan_id)

    print_summary(summary)
    print_overlap_diagnostics(overlap_diag, args.iou_threshold)
    print(f"\nWrote summary CSV: {output_csv}")
    print(f"Wrote summary JSON: {output_json}")
    print(f"Wrote pairwise CSV: {pairwise_csv}")
    print(f"Wrote pairwise JSON: {pairwise_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
