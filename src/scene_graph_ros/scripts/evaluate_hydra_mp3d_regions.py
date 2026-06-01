#!/usr/bin/env python3
"""Evaluate Hydra room/region outputs against Matterport3D regions.

Usage:
  python3 evaluate_hydra_mp3d_regions.py --scan_id 2t7WUuJeko7

Defaults:
  --hydra_dir /home/crcz/.hydra
  --mp3d_root /mnt/DATA/repos/phd/3dsg/mp3d/dataset/v1/scans
  --output_dir .

Outputs:
  region_eval_summary_<scan_id>.csv (one-row canonical metric summary)
  region_eval_summary_<scan_id>.json

The metric names, matching policy, and summary layout mirror
evaluate_mp3d_regions.py. Ground-truth room/region units come from
house_segmentations .house records. region_segmentations meshes are not used
for room/region segmentation metrics.
Hydra does not always export explicit room footprint polygons. In that case
this script deterministically derives room footprints from child place nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - depends on local environment
    np = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - depends on local environment
    linear_sum_assignment = None

try:
    from shapely.geometry import MultiPolygon, Polygon as ShapelyPolygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover - falls back to raster geometry
    MultiPolygon = None
    ShapelyPolygon = None
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
CONFIDENCE_KEYS = ("confidence", "score", "probability", "semantic_score")

Point = Tuple[float, float]

REGION_EXPORT_FIELDS = (
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

SEVERE_UNDERSEGMENTATION_THRESHOLD = 0.25


@dataclass
class Poly:
    points: List[Point]
    source: str

    @property
    def area(self) -> float:
        return polygon_area(self.points)

    @property
    def centroid(self) -> Point:
        return polygon_centroid(self.points)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class HydraRegion:
    node_id: str
    node_type: str
    label: Optional[str]
    confidence: Optional[float]
    polygon: Optional[Poly]
    centroid: Optional[Point]
    area: Optional[float]
    adjacent_ids: Set[str] = field(default_factory=set)
    source_node_ids: List[str] = field(default_factory=list)
    diagnostics: List[str] = field(default_factory=list)


@dataclass
class GtRegion:
    region_index: int
    level_index: int
    label_code: str
    label_name: str
    polygon: Poly
    centroid: Point
    area: float
    polygon_source: str
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
        logging.warning(message)


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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


def project_coord(coords: Dict[str, Optional[float]], plane: str) -> Optional[Point]:
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


def extract_xy(value: Any, plane: str = "xy") -> Optional[Point]:
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


def extract_points(value: Any, plane: str = "xy") -> List[Point]:
    if value is None:
        return []

    if isinstance(value, dict):
        for key in ("points", "vertices", "coordinates", "polygon", "boundary"):
            if key in value:
                points = extract_points(value[key], plane=plane)
                if points:
                    return points
        xy = extract_xy(value, plane=plane)
        return [xy] if xy is not None else []

    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(not isinstance(v, (dict, list, tuple)) for v in value[:2]):
            xy = extract_xy(value, plane=plane)
            return [xy] if xy is not None else []
        points: List[Point] = []
        for item in value:
            xy = extract_xy(item, plane=plane)
            if xy is not None:
                points.append(xy)
        return points

    return []


def clean_points(points: Sequence[Point]) -> List[Point]:
    cleaned: List[Point] = []
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        point = (float(x), float(y))
        if not cleaned or distance(cleaned[-1], point) > 1e-9:
            cleaned.append(point)
    if len(cleaned) > 1 and distance(cleaned[0], cleaned[-1]) <= 1e-9:
        cleaned.pop()
    return cleaned


def polygon_area(points: Sequence[Point]) -> float:
    pts = clean_points(points)
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def polygon_centroid(points: Sequence[Point]) -> Point:
    pts = clean_points(points)
    if not pts:
        return (0.0, 0.0)
    signed_twice_area = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        cross = x1 * y2 - x2 * y1
        signed_twice_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(signed_twice_area) < 1e-12:
        return (
            float(sum(x for x, _ in pts) / len(pts)),
            float(sum(y for _, y in pts) / len(pts)),
        )
    return (float(cx / (3.0 * signed_twice_area)), float(cy / (3.0 * signed_twice_area)))


def convex_hull(points: Sequence[Point]) -> List[Point]:
    pts = sorted(set(clean_points(points)))
    if len(pts) <= 1:
        return pts

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Point] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def make_polygon(points: Sequence[Point], source: str, allow_convex: bool = False) -> Optional[Poly]:
    pts = clean_points(points)
    if len(pts) < 3:
        return None
    if polygon_area(pts) <= 1e-9 and allow_convex:
        pts = convex_hull(pts)
    if polygon_area(pts) <= 1e-9:
        return None
    return Poly(points=pts, source=source)


def shapely_to_poly(geom: Any, source: str, warnings: List[str], context: str) -> Optional[Poly]:
    """Convert a Shapely footprint to the local Poly representation.

    The evaluator keeps Hydra geometry lightweight, but .house GT footprints may
    be a union of several floor surfaces. When that union is multipart, the
    largest component is used and the approximation is recorded as a warning.
    """
    if geom is None or geom.is_empty:
        return None
    if MultiPolygon is not None and isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda part: part.area)
        warn(warnings, f"{context}: multipart floor footprint; using largest component")
    if hasattr(geom, "geom_type") and geom.geom_type != "Polygon":
        geom = geom.convex_hull
        warn(warnings, f"{context}: non-polygon floor footprint; using convex hull")
    points = [(float(x), float(y)) for x, y in list(geom.exterior.coords)[:-1]]
    return make_polygon(points, source, allow_convex=False)


def poly_to_shapely(poly: Poly) -> Optional[Any]:
    if ShapelyPolygon is None:
        return None
    geom = ShapelyPolygon(poly.points)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom if not geom.is_empty and geom.area > 0.0 else None


def buffered_points(points: Sequence[Point], radius: float) -> List[Point]:
    if radius <= 0.0:
        return list(points)
    expanded: List[Point] = []
    for x, y in points:
        expanded.extend(
            [
                (x - radius, y - radius),
                (x - radius, y + radius),
                (x + radius, y - radius),
                (x + radius, y + radius),
            ]
        )
    return expanded


def distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def point_segment_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return distance(point, start)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    proj = (x1 + t * dx, y1 + t * dy)
    return distance(point, proj)


def boundary_distance(left: Poly, right: Poly) -> float:
    min_dist = math.inf
    left_edges = polygon_edges(left.points)
    right_edges = polygon_edges(right.points)
    for point in left.points:
        min_dist = min(min_dist, min(point_segment_distance(point, a, b) for a, b in right_edges))
    for point in right.points:
        min_dist = min(min_dist, min(point_segment_distance(point, a, b) for a, b in left_edges))
    return float(min_dist)


def polygon_edges(points: Sequence[Point]) -> List[Tuple[Point, Point]]:
    pts = clean_points(points)
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def sample_boundary_points(poly: Poly, step: float) -> List[Point]:
    pts = clean_points(poly.points)
    samples: List[Point] = []
    for start, end in polygon_edges(pts):
        length = distance(start, end)
        count = max(1, int(math.ceil(length / max(step, 1e-6))))
        for idx in range(count):
            t = idx / count
            samples.append((start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1])))
    return samples


def boundary_scores(pred: Poly, gt: Poly, tolerance: float, step: float) -> Tuple[float, float, float]:
    pred_points = sample_boundary_points(pred, step)
    gt_points = sample_boundary_points(gt, step)
    if not pred_points or not gt_points:
        return 0.0, 0.0, 0.0
    gt_edges = polygon_edges(gt.points)
    pred_edges = polygon_edges(pred.points)
    pred_hits = sum(
        1 for point in pred_points if min(point_segment_distance(point, a, b) for a, b in gt_edges) <= tolerance
    )
    gt_hits = sum(
        1 for point in gt_points if min(point_segment_distance(point, a, b) for a, b in pred_edges) <= tolerance
    )
    precision = pred_hits / len(pred_points)
    recall = gt_hits / len(gt_points)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def rasterize_pair(left: Poly, right: Poly, pixel_size: float) -> Tuple[float, float, float]:
    if cv2 is None or np is None:
        return bbox_iou(left, right)

    min_x = min(left.bounds[0], right.bounds[0])
    min_y = min(left.bounds[1], right.bounds[1])
    max_x = max(left.bounds[2], right.bounds[2])
    max_y = max(left.bounds[3], right.bounds[3])
    pad = max(pixel_size * 2.0, 1e-6)
    width = max(2, int(math.ceil((max_x - min_x + 2.0 * pad) / pixel_size)) + 1)
    height = max(2, int(math.ceil((max_y - min_y + 2.0 * pad) / pixel_size)) + 1)

    def to_pixels(poly: Poly) -> Any:
        coords = []
        for x, y in poly.points:
            px = int(round((x - min_x + pad) / pixel_size))
            py = int(round((max_y - y + pad) / pixel_size))
            coords.append([px, py])
        return np.asarray([coords], dtype=np.int32)

    left_mask = np.zeros((height, width), dtype=np.uint8)
    right_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(left_mask, to_pixels(left), 1)
    cv2.fillPoly(right_mask, to_pixels(right), 1)
    intersection = int(np.logical_and(left_mask, right_mask).sum()) * pixel_size * pixel_size
    union = int(np.logical_or(left_mask, right_mask).sum()) * pixel_size * pixel_size
    iou = intersection / union if union > 0.0 else 0.0
    return float(iou), float(intersection), float(union)


def bbox_iou(left: Poly, right: Poly) -> Tuple[float, float, float]:
    lx1, ly1, lx2, ly2 = left.bounds
    rx1, ry1, rx2, ry2 = right.bounds
    ix = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    iy = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = ix * iy
    union = left.area + right.area - intersection
    return (intersection / union if union else 0.0, intersection, union)


def graph_node_type(node: Dict[str, Any]) -> str:
    return str(node.get("attributes", {}).get("type") or node.get("type") or node.get("layer") or "unknown")


def graph_node_label(node: Dict[str, Any]) -> Optional[str]:
    value = first_nested_value(node, LABEL_KEYS)
    return str(value) if value not in (None, "") else None


def graph_node_confidence(node: Dict[str, Any]) -> Optional[float]:
    return to_float(first_nested_value(node, CONFIDENCE_KEYS))


def node_layer_name(graph: Dict[str, Any], node: Dict[str, Any]) -> str:
    type_to_layer = {
        "ObjectNodeAttributes": "OBJECTS",
        "AgentNodeAttributes": "AGENTS",
        "PlaceNodeAttributes": "PLACES",
        "Place2dNodeAttributes": "MESH_PLACES",
        "RoomNodeAttributes": "ROOMS",
        "SemanticNodeAttributes": "BUILDINGS",
    }
    ntype = graph_node_type(node)
    if ntype in type_to_layer:
        return type_to_layer[ntype]

    layer = node.get("layer")
    partition = node.get("partition", 0)
    for name, spec in graph.get("layer_names", {}).items():
        if spec.get("layer") == layer and spec.get("partition", 0) == partition:
            return str(name)
    return f"layer_{layer}_partition_{partition}"


def load_hydra_graph_path(args: argparse.Namespace) -> Path:
    if args.hydra_graph:
        return args.hydra_graph

    scan_dir = args.hydra_dir / args.scan_id
    stages = ["backend", "frontend"] if args.hydra_stage == "auto" else [args.hydra_stage]
    names = ["dsg_with_mesh.json", "dsg.json"]
    for stage in stages:
        for name in names:
            path = scan_dir / stage / name
            if path.exists():
                return path
    raise FileNotFoundError(f"No Hydra DSG JSON found under {scan_dir}")


def load_json(path: Path) -> Dict[str, Any]:
    logging.info("Loading Hydra graph: %s", path)
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, dict):
        raise ValueError(f"Hydra graph must be a JSON object: {path}")
    return data


def graph_edges(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges = graph.get("edges", [])
    if isinstance(edges, list):
        return [e for e in edges if isinstance(e, dict)]
    if isinstance(edges, dict):
        return [e for e in edges.values() if isinstance(e, dict)]
    return []


def get_polygon_payload(node: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    for scope_name in ("attributes", "geometry", "geometry_signature", "root"):
        scope = node if scope_name == "root" else node.get(scope_name)
        if not isinstance(scope, dict):
            continue
        for key in keys:
            if key in scope and scope[key] not in (None, ""):
                return scope[key], f"{scope_name}.{key}"
    return None, None


def direct_node_polygon(node: Dict[str, Any], plane: str) -> Tuple[Optional[Poly], Optional[str]]:
    payload, source = get_polygon_payload(node, POLYGON_KEYS)
    points = extract_points(payload, plane=plane)
    if points:
        poly = make_polygon(points, source or "node.polygon", allow_convex=True)
        if poly is not None:
            return poly, poly.source

    payload, source = get_polygon_payload(node, HULL_KEYS)
    points = extract_points(payload, plane=plane)
    if points:
        hull = convex_hull(points)
        poly = make_polygon(hull, source or "node.convex_hull", allow_convex=True)
        if poly is not None:
            return poly, poly.source

    bbox = node.get("attributes", {}).get("bounding_box")
    if isinstance(bbox, dict):
        dims = bbox.get("dimensions")
        center = extract_xy(bbox.get("world_P_center"), plane=plane)
        if isinstance(dims, list) and len(dims) >= 2 and center is not None:
            dx = to_float(dims[0]) or 0.0
            dy = to_float(dims[1]) or 0.0
            if dx > 0.0 and dy > 0.0:
                cx, cy = center
                points = [
                    (cx - dx / 2.0, cy - dy / 2.0),
                    (cx + dx / 2.0, cy - dy / 2.0),
                    (cx + dx / 2.0, cy + dy / 2.0),
                    (cx - dx / 2.0, cy + dy / 2.0),
                ]
                return Poly(points, "attributes.bounding_box"), "attributes.bounding_box"

    return None, None


def build_adjacency(edges: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        source = edge.get("source", edge.get("src", edge.get("from")))
        target = edge.get("target", edge.get("dst", edge.get("to")))
        if source is None or target is None:
            continue
        left, right = str(source), str(target)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def normalized_edge(left: Any, right: Any) -> Optional[Tuple[str, str]]:
    if left is None or right is None:
        return None
    edge = tuple(sorted((str(left), str(right))))
    return edge if edge[0] != edge[1] else None


def derive_room_polygon(
    room_id: str,
    adjacency: Dict[str, Set[str]],
    node_by_id: Dict[str, Dict[str, Any]],
    plane: str,
    place_position_buffer: float,
) -> Tuple[Optional[Poly], List[str], str]:
    child_ids = [
        node_id
        for node_id in sorted(adjacency.get(room_id, set()))
        if graph_node_type(node_by_id.get(node_id, {})) == "PlaceNodeAttributes"
    ]
    points: List[Point] = []
    for child_id in child_ids:
        child = node_by_id[child_id]
        point = extract_xy(child.get("attributes", {}).get("position"), plane=plane)
        if point is not None:
            points.append(point)

    if len(points) < 3:
        return None, child_ids, "room_child_place_position_hull"

    hull = convex_hull(buffered_points(points, place_position_buffer))
    return make_polygon(hull, "room_child_place_position_hull", allow_convex=True), child_ids, "room_child_place_position_hull"


def hydra_regions_from_rooms(
    graph: Dict[str, Any],
    adjacency: Dict[str, Set[str]],
    plane: str,
    place_position_buffer: float,
    warnings: List[str],
) -> Tuple[List[HydraRegion], Set[Tuple[str, str]], str]:
    node_by_id = {str(node["id"]): node for node in graph.get("nodes", [])}
    rooms = [
        node
        for node in graph.get("nodes", [])
        if graph_node_type(node) == "RoomNodeAttributes" or node_layer_name(graph, node).upper() == "ROOMS"
    ]
    regions: List[HydraRegion] = []

    for room in rooms:
        room_id = str(room["id"])
        poly, source = direct_node_polygon(room, plane)
        child_ids: List[str] = []
        if poly is None:
            poly, child_ids, source = derive_room_polygon(
                room_id,
                adjacency,
                node_by_id,
                plane,
                place_position_buffer,
            )
        diagnostics: List[str] = []
        if poly is None:
            diagnostics.append("missing room footprint; no direct polygon, valid bbox, or child-place hull")
            warn(warnings, f"Hydra room {room_id} has no usable footprint")
        regions.append(
            HydraRegion(
                node_id=room_id,
                node_type=graph_node_type(room),
                label=graph_node_label(room),
                confidence=graph_node_confidence(room),
                polygon=poly,
                centroid=poly.centroid if poly is not None else extract_xy(room.get("attributes", {}).get("position"), plane),
                area=poly.area if poly is not None else None,
                source_node_ids=child_ids or [room_id],
                diagnostics=diagnostics,
            )
        )

    room_ids = {region.node_id for region in regions}
    pred_adj: Set[Tuple[str, str]] = set()
    for left in room_ids:
        for right in adjacency.get(left, set()):
            if right in room_ids:
                edge = normalized_edge(left, right)
                if edge is not None:
                    pred_adj.add(edge)

    return regions, pred_adj, "rooms"


def load_hydra_regions(
    graph_json: Path,
    source: str,
    plane: str,
    place_position_buffer: float,
    warnings: List[str],
) -> Tuple[List[HydraRegion], Set[Tuple[str, str]], Dict[str, Any]]:
    graph = load_json(graph_json)
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Hydra DSG JSON must contain a top-level nodes list")
    adjacency = build_adjacency(graph_edges(graph))

    node_type_counts = Counter(graph_node_type(node) for node in nodes)
    layer_counts = Counter(node_layer_name(graph, node) for node in nodes)
    logging.info("Hydra node type counts: %s", dict(node_type_counts))
    logging.info("Hydra layer counts: %s", dict(layer_counts))

    if source == "rooms":
        regions, pred_adj, selected_source = hydra_regions_from_rooms(
            graph,
            adjacency,
            plane,
            place_position_buffer,
            warnings,
        )
    else:
        raise ValueError("Hydra room evaluation only supports source='rooms'")

    logging.info(
        "Loaded %d Hydra region candidates (%d with polygons) using source=%s",
        len(regions),
        sum(1 for region in regions if region.polygon is not None),
        selected_source,
    )

    # Extract earliest agent/pose node for frame-alignment seeding.
    first_pose: Optional[Point] = None
    agent_candidates: List[Tuple[str, Point]] = []
    for node in nodes:
        ntype = graph_node_type(node).upper()
        if "AGENT" not in ntype and "POSE" not in ntype:
            continue
        pos = extract_xy(node.get("attributes", {}).get("position"), plane=plane)
        if pos is not None:
            agent_candidates.append((str(node.get("id", "")), pos))
    if agent_candidates:
        try:
            agent_candidates.sort(key=lambda item: int(item[0]))
        except (ValueError, TypeError):
            agent_candidates.sort()
        first_pose = agent_candidates[0][1]

    metadata = {
        "graph_layout": "Spark DSG JSON nodes/edges",
        "top_level_keys": list(graph.keys()),
        "node_type_counts": dict(node_type_counts),
        "layer_counts": dict(layer_counts),
        "edge_count": len(graph_edges(graph)),
        "hydra_region_source": selected_source,
        "explicit_predicted_adjacency_edges": len(pred_adj),
        "place_position_buffer": place_position_buffer,
        "first_pose_xy": list(first_pose) if first_pose is not None else None,
    }
    return regions, pred_adj, metadata


def find_scan_file(scan_root: Path, subdir: str, suffix: str) -> Path:
    directory = scan_root / subdir
    if directory.is_dir():
        matches = sorted(directory.rglob(f"*{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not locate *{suffix} under {directory}")


def parse_house(scan_root: Path, scan_id: str, plane: str, warnings: List[str]) -> Tuple[List[GtRegion], Set[Tuple[int, int]], Dict[str, Any]]:
    house_path = find_scan_file(scan_root, "house_segmentations", ".house")
    logging.info("Loading MP3D house file: %s", house_path)

    regions: Dict[int, Dict[str, Any]] = {}
    surfaces_by_region: Dict[int, List[int]] = defaultdict(list)
    surface_info: Dict[int, Dict[str, Any]] = {}
    vertices_by_surface: Dict[int, List[Point]] = defaultdict(list)
    portals: Set[Tuple[int, int]] = set()

    for line in house_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        record = parts[0]
        if record == "R" and len(parts) >= 16:
            region_index = int(parts[1])
            centroid = project_coord(
                {"x": float(parts[6]), "y": float(parts[7]), "z": float(parts[8])},
                plane,
            )
            regions[region_index] = {
                "region_index": region_index,
                "level_index": int(parts[2]),
                "label_code": parts[5],
                "centroid": centroid,
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
        elif record == "P" and len(parts) >= 4 and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit():
            left, right = int(parts[2]), int(parts[3])
            if left != right:
                portals.add(tuple(sorted((left, right))))

    gt_regions: List[GtRegion] = []
    source_counter: Counter = Counter()
    for region_index in sorted(regions):
        region = regions[region_index]
        floor_points: List[Point] = []
        floor_polygons: List[Any] = []
        floor_surface_polys: List[Poly] = []
        for surface_index in surfaces_by_region.get(region_index, []):
            info = surface_info[surface_index]
            normal = info["normal"]
            is_floor_like = info["label"].upper() == "F" or abs(normal[2]) > 0.85
            if is_floor_like:
                points = vertices_by_surface.get(surface_index, [])
                floor_points.extend(points)
                surface_poly = make_polygon(points, f"house_surface_{surface_index}", allow_convex=False)
                if surface_poly is None:
                    continue
                if ShapelyPolygon is not None:
                    geom = ShapelyPolygon(surface_poly.points)
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    if not geom.is_empty and geom.area > 0.0:
                        floor_polygons.append(geom)
                else:
                    floor_surface_polys.append(surface_poly)

        source = "house_surfaces"
        poly = None
        if floor_polygons and unary_union is not None:
            geom = unary_union(floor_polygons) if len(floor_polygons) > 1 else floor_polygons[0]
            poly = shapely_to_poly(geom, source, warnings, f"GT region {region_index}")
        elif len(floor_surface_polys) == 1:
            poly = Poly(floor_surface_polys[0].points, source)
        elif floor_surface_polys:
            warn(
                warnings,
                f"GT region {region_index} has multiple floor surfaces but Shapely is unavailable; using vertex hull",
            )
            poly = make_polygon(floor_points, source, allow_convex=True)
        if poly is None and not floor_points:
            # Fallback: no floor-labelled surfaces found for this region.
            # data_organization.md states that region extents are prisms whose
            # horizontal cross-section is defined by the vertices of ALL
            # associated surfaces, not just floors.  When no floor-like surface
            # is tagged, approximate the footprint from the full vertex set.
            all_surface_points: List[Point] = []
            for s_idx in surfaces_by_region.get(region_index, []):
                all_surface_points.extend(vertices_by_surface.get(s_idx, []))
            if all_surface_points:
                warn(
                    warnings,
                    f"GT region {region_index}: no floor-like surfaces found; "
                    "footprint approximated from all-surface vertex hull",
                )
                poly = make_polygon(all_surface_points, "house_all_surfaces_hull", allow_convex=True)
                if poly is not None:
                    source = "house_all_surfaces_hull"
        if poly is None:
            source = "missing_house_footprint"
        if poly is None:
            warn(warnings, f"Skipping GT region {region_index}; no valid footprint")
            continue

        source_counter[source] += 1
        label_code = region["label_code"]
        gt_regions.append(
            GtRegion(
                region_index=region_index,
                level_index=region["level_index"],
                label_code=label_code,
                label_name=REGION_LABELS.get(label_code, f"label_{label_code}"),
                polygon=poly,
                centroid=poly.centroid,
                area=poly.area,
                polygon_source=source,
                height=region["height"],
            )
        )

    metadata = {
        "house_source": str(house_path),
        "gt_region_records": len(regions),
        "gt_polygon_sources": dict(source_counter),
        "gt_polygon_source": "mixed" if len(source_counter) > 1 else (next(iter(source_counter)) if source_counter else "none"),
        "portal_edges": len(portals),
        "scan_id": scan_id,
    }
    logging.info("Loaded %d MP3D GT regions", len(gt_regions))
    return gt_regions, portals, metadata


def compute_pairwise(preds: Sequence[HydraRegion], gts: Sequence[GtRegion], pixel_size: float) -> List[Pairwise]:
    pairs: List[Pairwise] = []
    for pred in preds:
        if pred.polygon is None:
            continue
        pred_geom = poly_to_shapely(pred.polygon)
        for gt in gts:
            gt_geom = poly_to_shapely(gt.polygon)
            if pred_geom is not None and gt_geom is not None:
                intersection = float(pred_geom.intersection(gt_geom).area)
                union = float(pred_geom.union(gt_geom).area)
                iou = intersection / union if union > 0.0 else 0.0
            else:
                iou, intersection, union = rasterize_pair(pred.polygon, gt.polygon, pixel_size)
            pairs.append(Pairwise(pred.node_id, gt.region_index, iou, intersection, union))
    return pairs


def match_regions(preds: Sequence[HydraRegion], gts: Sequence[GtRegion], pairwise: Sequence[Pairwise], iou_threshold: float) -> Tuple[Dict[str, int], Dict[Tuple[str, int], Pairwise], str]:
    pair_lookup = {(pair.pred_id, pair.gt_id): pair for pair in pairwise}
    pred_ids = [pred.node_id for pred in preds if pred.polygon is not None]
    gt_ids = [gt.region_index for gt in gts]
    matches: Dict[str, int] = {}

    if pred_ids and gt_ids and linear_sum_assignment is not None and np is not None:
        matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=float)
        for row, pred_id in enumerate(pred_ids):
            for col, gt_id in enumerate(gt_ids):
                pair = pair_lookup.get((pred_id, gt_id))
                matrix[row, col] = pair.iou if pair is not None else 0.0
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            if matrix[row, col] >= iou_threshold:
                matches[pred_ids[row]] = gt_ids[col]
        return matches, pair_lookup, "hungarian"

    used_preds: Set[str] = set()
    used_gts: Set[int] = set()
    for pair in sorted(pairwise, key=lambda item: item.iou, reverse=True):
        if pair.iou < iou_threshold:
            break
        if pair.pred_id in used_preds or pair.gt_id in used_gts:
            continue
        matches[pair.pred_id] = pair.gt_id
        used_preds.add(pair.pred_id)
        used_gts.add(pair.gt_id)
    return matches, pair_lookup, "greedy"


def overlap_counts(preds: Sequence[HydraRegion], gts: Sequence[GtRegion], pair_lookup: Dict[Tuple[str, int], Pairwise], min_overlap_area: float) -> Tuple[Dict[int, List[str]], Dict[str, List[int]]]:
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


def polygon_adjacency_gt(gts: Sequence[GtRegion], tolerance: float) -> Set[Tuple[int, int]]:
    adjacency: Set[Tuple[int, int]] = set()
    for idx, left in enumerate(gts):
        for right in gts[idx + 1 :]:
            if left.level_index != right.level_index:
                continue
            if boundary_distance(left.polygon, right.polygon) <= tolerance:
                adjacency.add(tuple(sorted((left.region_index, right.region_index))))
    return adjacency


def adjacency_to_gt_space(
    pred_adjacency: Set[Tuple[str, str]],
    matches: Dict[str, int],
) -> Tuple[Set[Tuple[int, int]], int]:
    pred_gt_space_adjacency: Set[Tuple[int, int]] = set()
    dropped_edges = 0
    for left, right in pred_adjacency:
        gt_left = matches.get(left)
        gt_right = matches.get(right)
        if gt_left is None or gt_right is None:
            dropped_edges += 1
            continue
        if gt_left == gt_right:
            dropped_edges += 1
            continue
        pred_gt_space_adjacency.add(tuple(sorted((gt_left, gt_right))))
    return pred_gt_space_adjacency, dropped_edges


def polygon_adjacency_pred(preds: Sequence[HydraRegion], tolerance: float) -> Set[Tuple[str, str]]:
    adjacency: Set[Tuple[str, str]] = set()
    usable = [pred for pred in preds if pred.polygon is not None]
    for idx, left in enumerate(usable):
        for right in usable[idx + 1 :]:
            if boundary_distance(left.polygon, right.polygon) <= tolerance:
                edge = normalized_edge(left.node_id, right.node_id)
                if edge is not None:
                    adjacency.add(edge)
    return adjacency


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.mean(values)) if values else None


def safe_median(values: Sequence[float]) -> Optional[float]:
    return float(statistics.median(values)) if values else None


def zero_if_none(value: Optional[float]) -> float:
    """Return 0.0 for undefined match-dependent metrics.

    A no-match scan is a real segmentation failure for aggregate comparison, not
    a missing datum. Matched-only metric names still make the conditioning clear.
    """
    return float(value) if value is not None else 0.0


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio, with 0.0 when the denominator is absent."""
    return float(numerator / denominator) if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def classify_failure_mode(
    gt_count: int,
    pred_count: int,
    matched_count: int,
    gt_region_coverage: float,
    region_count_ratio: float,
) -> Tuple[str, bool]:
    """Categorize scene-level region failures.

    The category is intentionally based on counts and GT coverage rather than
    matched-pair geometry quality. This prevents a single good match from hiding
    a method that missed most rooms in the scan.
    """
    severe = (
        gt_count > 0
        and (
            region_count_ratio < SEVERE_UNDERSEGMENTATION_THRESHOLD
            or gt_region_coverage < SEVERE_UNDERSEGMENTATION_THRESHOLD
        )
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
    preds: Sequence[HydraRegion],
    gts: Sequence[GtRegion],
    matches: Dict[str, int],
    pair_lookup: Dict[Tuple[str, int], Pairwise],
    explicit_pred_adjacency: Set[Tuple[str, str]],
    gt_portals: Set[Tuple[int, int]],
    args: argparse.Namespace,
    warnings: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Tuple[float, float, float]], Dict[int, List[str]], Dict[str, List[int]]]:
    matched_pairs = [pair_lookup[(pred_id, gt_id)] for pred_id, gt_id in matches.items()]
    ious = [pair.iou for pair in matched_pairs]

    gt_ids = {gt.region_index for gt in gts}
    pred_ids = {pred.node_id for pred in preds}
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
    full_penalty_count = len(matches) + missed_gt_region_count + unmatched_predicted_region_count
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
        pred = next(item for item in preds if item.node_id == pred_id)
        gt = next(item for item in gts if item.region_index == gt_id)
        if pred.polygon is not None:
            boundary_by_pred[pred_id] = boundary_scores(
                pred.polygon,
                gt.polygon,
                args.boundary_tolerance,
                args.boundary_sample_step,
            )

    boundary_precisions = [value[0] for value in boundary_by_pred.values()]
    boundary_recalls = [value[1] for value in boundary_by_pred.values()]
    boundary_f1s = [value[2] for value in boundary_by_pred.values()]

    gt_to_preds, pred_to_gts = overlap_counts(preds, gts, pair_lookup, args.min_overlap_area)
    over_segmented = {str(gt_id): fragments for gt_id, fragments in sorted(gt_to_preds.items()) if len(fragments) > 1}
    under_segmented = {pred_id: gt_regions for pred_id, gt_regions in sorted(pred_to_gts.items()) if len(gt_regions) > 1}
    oversegmentation_rate = safe_ratio(len(over_segmented), gt_count)
    undersegmentation_rate = safe_ratio(len(under_segmented), pred_count)
    no_predicted_regions = gt_count > 0 and pred_count == 0
    no_valid_matches = gt_count > 0 and len(matches) == 0
    single_region_failure = gt_count > 1 and pred_count == 1
    region_count_collapse = gt_count > 0 and region_count_ratio <= SEVERE_UNDERSEGMENTATION_THRESHOLD

    gt_adj_source = "house_portals"
    gt_adjacency = set(gt_portals)
    if not gt_adjacency:
        gt_adj_source = "polygon_boundary_proximity"
        gt_adjacency = polygon_adjacency_gt(gts, args.adjacency_tolerance)

    pred_adj_source = "exported_graph_or_derived_components"
    pred_adjacency = set(explicit_pred_adjacency)
    pred_gt_space_adjacency, dropped_pred_adj_edges = adjacency_to_gt_space(pred_adjacency, matches)
    if not pred_adjacency or (matches and pred_adjacency and not pred_gt_space_adjacency):
        if pred_adjacency and not pred_gt_space_adjacency:
            warn(
                warnings,
                "Predicted adjacency edges could not be mapped through matched regions; using polygon proximity",
            )
        else:
            warn(warnings, "No predicted room/region adjacency found; using polygon proximity")
        pred_adj_source = "polygon_boundary_proximity"
        pred_adjacency = polygon_adjacency_pred(preds, args.adjacency_tolerance)
        pred_gt_space_adjacency, dropped_pred_adj_edges = adjacency_to_gt_space(pred_adjacency, matches)

    logging.debug(
        "Region adjacency: gt_source=%s gt_edges=%d pred_source=%s pred_edges=%d "
        "mapped_pred_edges=%d dropped_pred_edges=%d",
        gt_adj_source,
        len(gt_adjacency),
        pred_adj_source,
        len(pred_adjacency),
        len(pred_gt_space_adjacency),
        dropped_pred_adj_edges,
    )

    adjacency_tp = len(pred_gt_space_adjacency & gt_adjacency)
    adjacency_fp = len(pred_gt_space_adjacency - gt_adjacency)
    adjacency_fn = len(gt_adjacency - pred_gt_space_adjacency)
    adjacency_precision = adjacency_tp / (adjacency_tp + adjacency_fp) if adjacency_tp + adjacency_fp else 0.0
    adjacency_recall = adjacency_tp / (adjacency_tp + adjacency_fn) if adjacency_tp + adjacency_fn else 0.0
    adjacency_f1 = (
        2.0 * adjacency_precision * adjacency_recall / (adjacency_precision + adjacency_recall)
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
            "num_pred_regions_with_polygons": sum(1 for pred in preds if pred.polygon is not None),
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
            # Boundary scores are matched-only geometry scores; no-match scans
            # export 0.0 so aggregate tables retain the failed scan.
            "mean_boundary_precision": zero_if_none(safe_mean(boundary_precisions)),
            "mean_boundary_recall": zero_if_none(safe_mean(boundary_recalls)),
            "mean_boundary_f1": zero_if_none(safe_mean(boundary_f1s)),
            "boundary_tolerance": args.boundary_tolerance,
            "boundary_sample_step": args.boundary_sample_step,
        },
        "over_segmentation": {
            "num_over_segmented_gt_regions": len(over_segmented),
            "mean_predicted_fragments_per_gt_region": safe_mean(
                [float(len(gt_to_preds.get(gt.region_index, []))) for gt in gts]
            ),
            "affected_gt_region_ids": sorted(int(key) for key in over_segmented.keys()),
            "details": over_segmented,
        },
        "under_segmentation": {
            "num_merged_predicted_regions": len(under_segmented),
            "mean_gt_regions_per_predicted_region": safe_mean(
                [float(len(pred_to_gts.get(pred.node_id, []))) for pred in preds if pred.polygon is not None]
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
            "num_predicted_adjacency_edges": len(pred_adjacency),
            "num_predicted_gt_space_adjacency_edges": len(pred_gt_space_adjacency),
            "num_dropped_predicted_adjacency_edges": dropped_pred_adj_edges,
        },
    }

    if not matches and preds and gts:
        warn(warnings, "No regions met the IoU threshold. Check coordinate frame alignment and Hydra source selection.")

    return metrics, boundary_by_pred, gt_to_preds, pred_to_gts


def build_export_row(summary: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    region_iou = summary["metrics"]["region_iou"]
    row = {
        "scan_id": summary["scan_id"],
        "method": summary.get("method", "hydra"),
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
        "source_file": str(source_file),
    }
    validate_export_schema(row)
    return row


def validate_export_schema(row: Dict[str, Any]) -> None:
    keys = tuple(row.keys())
    if keys != REGION_EXPORT_FIELDS:
        missing = [key for key in REGION_EXPORT_FIELDS if key not in row]
        extra = [key for key in keys if key not in REGION_EXPORT_FIELDS]
        raise ValueError(
            "Region export schema mismatch: "
            f"missing={missing or 'none'}, extra={extra or 'none'}, order={list(keys)}"
        )


PAIRWISE_FIELDS = (
    "pred_node_id",
    "pred_type",
    "pred_label",
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


def _build_pairwise_rows_hydra(
    preds: Sequence[HydraRegion],
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
            "pred_label": pred.label if pred and pred.label is not None else "",
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


def write_pairwise_csv_hydra(
    path: Path,
    preds: Sequence[HydraRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
) -> None:
    """Write the full N×M pairwise overlap table to a CSV file."""
    rows = _build_pairwise_rows_hydra(preds, gts, pairwise, matches)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=PAIRWISE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_pairwise_json_hydra(
    path: Path,
    preds: Sequence[HydraRegion],
    gts: Sequence[GtRegion],
    pairwise: Sequence[Pairwise],
    matches: Dict[str, int],
    scan_id: str,
) -> None:
    """Write the full N×M pairwise overlap table to a JSON file."""
    rows = _build_pairwise_rows_hydra(preds, gts, pairwise, matches)
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


def write_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    validate_export_schema(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=REGION_EXPORT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerow(row)


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(summary: Dict[str, Any]) -> None:
    metrics = summary["metrics"]
    region_iou = metrics["region_iou"]
    boundary = metrics["boundary_f1"]
    over = metrics["over_segmentation"]
    under = metrics["under_segmentation"]
    adj = metrics["region_adjacency_f1"]

    print("\nHydra vs Matterport3D Region Evaluation")
    print("=======================================")
    print(f"scan_id: {summary['scan_id']}")
    print(f"hydra_source: {summary['graph_metadata'].get('hydra_region_source')}")
    print(f"gt regions: {region_iou['num_gt_regions']}")
    print(f"hydra regions: {region_iou['num_pred_regions']}")
    print(f"matched regions: {region_iou['num_matches']}")
    print(
        "unmatched GT / unmatched Hydra: "
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
        "boundary precision/recall/f1: "
        f"{fmt(boundary['mean_boundary_precision'])} / {fmt(boundary['mean_boundary_recall'])} / {fmt(boundary['mean_boundary_f1'])}"
    )
    print(
        "failure mode / severe undersegmentation: "
        f"{region_iou['failure_mode']} / {region_iou['severe_undersegmentation']}"
    )
    print(f"over_segmentation: {over['num_over_segmented_gt_regions']} GT regions")
    print(f"under_segmentation: {under['num_merged_predicted_regions']} Hydra regions")
    print(
        "region_adjacency precision/recall/f1: "
        f"{fmt(adj['region_adjacency_precision'])} / {fmt(adj['region_adjacency_recall'])} / {fmt(adj['region_adjacency_f1'])}"
    )
    if summary.get("warnings"):
        print("\nWarnings:")
        for message in summary["warnings"]:
            print(f"  - {message}")


# ─────────────────────────── Overlap diagnostics ────────────────────────────


def _poly_bounds(polygon: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return (minx, miny, maxx, maxy) from a Poly object or any object with .bounds."""
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
    preds: Sequence[HydraRegion],
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
            d = distance(pred.centroid, gt.centroid)
            if min_centroid_dist is None or d < min_centroid_dist:
                min_centroid_dist = d

    def _cdist(pred: HydraRegion, gt: GtRegion) -> Optional[float]:
        if pred.centroid is None:
            return None
        return float(distance(pred.centroid, gt.centroid))

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
        print("  (no Hydra regions with valid polygons)")

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
        print("Pred bounds:     n/a (no valid Hydra region polygons)")
    print(f"GT total area:   {fmt(diag['gt_total_area'])}")
    print(f"Pred total area: {fmt(diag['pred_total_area'])}")
    print(f"Bounds overlap:  {'yes' if diag['bounds_overlap'] else 'NO'}")

    if pred_b is None or diag["pred_total_area"] == 0.0:
        issue = "missing/invalid Hydra region polygons"
    elif gt_b is None or diag["gt_total_area"] == 0.0:
        issue = "missing/invalid GT polygons"
    elif not diag["bounds_overlap"]:
        issue = "ALIGNMENT / FRAME MISMATCH — GT and Hydra bounding boxes do not overlap at all"
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
    preds: Sequence[HydraRegion],
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
            f"Validation: {len(preds) - pred_with_poly}/{len(preds)} Hydra regions lack valid polygons",
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Hydra rooms/regions against Matterport3D regions.")
    parser.add_argument("--scan_id", required=True)
    parser.add_argument("--hydra_dir", type=Path, default=Path("/home/crcz/.hydra"))
    parser.add_argument("--mp3d_root", type=Path, default=Path("/mnt/DATA/repos/phd/3dsg/mp3d/dataset/v1/scans"))
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--hydra_graph", type=Path, help="Override automatic Hydra DSG JSON path.")
    parser.add_argument("--method_name", default="hydra")
    parser.add_argument("--hydra_stage", default="backend", choices=("backend", "frontend", "auto"))
    parser.add_argument(
        "--hydra_source",
        default="rooms",
        choices=("rooms",),
        help="Hydra room/region nodes to evaluate from the final DSG.",
    )
    parser.add_argument("--plane", default="xy", choices=("xy", "xz", "yz"))
    parser.add_argument("--iou_threshold", type=float, default=0.15)
    parser.add_argument("--boundary_tolerance", type=float, default=0.25)
    parser.add_argument("--boundary_sample_step", type=float, default=0.10)
    parser.add_argument("--min_overlap_area", type=float, default=0.1)
    parser.add_argument("--adjacency_tolerance", type=float, default=0.30)
    parser.add_argument("--pixel_size", type=float, default=0.03, help="Raster size in meters for IoU area estimates.")
    parser.add_argument("--place_position_buffer", type=float, default=0.25, help="Buffer for room footprints derived from child place positions.")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument(
        "--gt_tx",
        type=float,
        default=0.0,
        help=(
            "Translate GT polygons by this amount in the projected X axis before "
            "computing IoU. Use to compensate for a known offset between the MP3D "
            "house-file frame and the Hydra coordinate frame."
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
            "computing IoU. Uses the first agent/pose node in the Hydra graph as "
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
    coordinate frame and the Hydra/SLAM coordinate frame. Area is invariant under
    translation; centroid and polygon vertices are updated in-place.
    """
    if tx == 0.0 and ty == 0.0:
        return
    for gt in gts:
        gt.polygon = Poly(
            points=[(x + tx, y + ty) for x, y in gt.polygon.points],
            source=gt.polygon.source,
        )
        gt.centroid = (gt.centroid[0] + tx, gt.centroid[1] + ty)


# ─────────────────────── Frame auto-alignment ────────────────────────────────


def _rotate_points_h(
    points: List[Point], angle_deg: float, cx: float, cy: float
) -> List[Point]:
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


def _rotate_poly_h(poly: Poly, angle_deg: float, cx: float, cy: float) -> Poly:
    """Return a new Poly rotated around (cx, cy)."""
    return Poly(
        points=_rotate_points_h(poly.points, angle_deg, cx, cy),
        source=poly.source,
    )


def _alignment_score_hydra(
    preds: List[HydraRegion],
    gts: List[GtRegion],
    tx: float,
    ty: float,
    angle_deg: float,
    cx: float,
    cy: float,
    pixel_size: float,
) -> float:
    """Sum of all pairwise intersection areas after rotating GT around (cx, cy) then translating."""
    total = 0.0
    for gt in gts:
        gt_t = _rotate_poly_h(gt.polygon, angle_deg, cx, cy)
        gt_t = Poly(points=[(x + tx, y + ty) for x, y in gt_t.points], source=gt_t.source)
        for pred in preds:
            if pred.polygon is None:
                continue
            try:
                if ShapelyPolygon is not None:
                    ps = poly_to_shapely(pred.polygon)
                    gs = poly_to_shapely(gt_t)
                    if ps is not None and gs is not None:
                        total += float(ps.intersection(gs).area)
                        continue
                _, inter, _ = rasterize_pair(pred.polygon, gt_t, pixel_size)
                total += inter
            except Exception:
                pass
    return total


def rotate_gt_regions_hydra(
    gts: List[GtRegion], angle_deg: float, cx: float, cy: float
) -> None:
    """Rotate all GT polygon vertices around (cx, cy) in-place."""
    if angle_deg == 0.0:
        return
    for gt in gts:
        gt.polygon = _rotate_poly_h(gt.polygon, angle_deg, cx, cy)
        (gt.centroid,) = _rotate_points_h([gt.centroid], angle_deg, cx, cy)


def auto_align_hydra(
    preds: List[HydraRegion],
    gts: List[GtRegion],
    initial_tx: float,
    initial_ty: float,
    radius: float,
    step: float,
    try_rotations: bool,
    pixel_size: float,
    warnings: List[str],
) -> Tuple[float, float, float, float]:
    """Grid-search the GT-to-pred transform that maximizes total pairwise intersection area.

    Rotation (when enabled) is applied around the GT centroid before translation.
    Returns (best_tx, best_ty, best_rotation_deg, best_overlap_score).
    """
    usable_preds = [p for p in preds if p.polygon is not None]
    usable_gts = [g for g in gts if g.polygon is not None]
    if not usable_preds or not usable_gts:
        warn(warnings, "auto_align: no valid polygons to search over.")
        return initial_tx, initial_ty, 0.0, 0.0

    cx = statistics.mean(g.centroid[0] for g in usable_gts)
    cy = statistics.mean(g.centroid[1] for g in usable_gts)

    rotations = [0.0, 90.0, 180.0, 270.0] if try_rotations else [0.0]
    n_steps = max(1, int(math.ceil(radius / max(step, 1e-9))))

    best_tx, best_ty, best_rot = initial_tx, initial_ty, 0.0
    best_score = _alignment_score_hydra(
        usable_preds, usable_gts, initial_tx, initial_ty, 0.0, cx, cy, pixel_size
    )

    for rot in rotations:
        for di in range(-n_steps, n_steps + 1):
            for dj in range(-n_steps, n_steps + 1):
                tx = initial_tx + di * step
                ty = initial_ty + dj * step
                sc = _alignment_score_hydra(
                    usable_preds, usable_gts, tx, ty, rot, cx, cy, pixel_size
                )
                if sc > best_score:
                    best_score = sc
                    best_tx, best_ty, best_rot = tx, ty, rot

    return best_tx, best_ty, best_rot, best_score


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if cv2 is None or np is None:
        print("ERROR: numpy and opencv-python are required for deterministic raster IoU in this environment.", file=sys.stderr)
        return 2

    warnings: List[str] = []
    if args.plane != "xy":
        warn(warnings, f"Plane {args.plane} requested; xy is the validated/default path for MP3D/Hydra here")

    hydra_graph = load_hydra_graph_path(args)
    scan_root = args.mp3d_root / args.scan_id
    output_csv = args.output_dir / f"region_eval_summary_{args.scan_id}.csv"
    output_json = args.output_dir / f"region_eval_summary_{args.scan_id}.json"

    preds, pred_adjacency, graph_meta = load_hydra_regions(
        hydra_graph,
        args.hydra_source,
        args.plane,
        args.place_position_buffer,
        warnings,
    )
    gts, gt_portals, gt_meta = parse_house(scan_root, args.scan_id, args.plane, warnings)

    # ── coordinate-frame alignment ────────────────────────────────────────────
    if args.auto_align:
        first_pose = graph_meta.get("first_pose_xy")
        pose_tx = -first_pose[0] if first_pose else 0.0
        pose_ty = -first_pose[1] if first_pose else 0.0

        usable_gts = [g for g in gts if g.polygon is not None]
        usable_preds = [p for p in preds if p.polygon is not None]
        if usable_gts and usable_preds:
            gt_cx_mean = statistics.mean(g.centroid[0] for g in usable_gts)
            gt_cy_mean = statistics.mean(g.centroid[1] for g in usable_gts)
            pred_cx_mean = statistics.mean(p.centroid[0] for p in usable_preds if p.centroid)
            pred_cy_mean = statistics.mean(p.centroid[1] for p in usable_preds if p.centroid)
            cent_tx = pred_cx_mean - gt_cx_mean
            cent_ty = pred_cy_mean - gt_cy_mean

            pivot_cx = gt_cx_mean
            pivot_cy = gt_cy_mean
            score_pose = _alignment_score_hydra(
                usable_preds, usable_gts, pose_tx, pose_ty, 0.0, pivot_cx, pivot_cy, args.pixel_size
            )
            score_cent = _alignment_score_hydra(
                usable_preds, usable_gts, cent_tx, cent_ty, 0.0, pivot_cx, pivot_cy, args.pixel_size
            )
            if first_pose and score_pose >= score_cent:
                initial_tx, initial_ty = pose_tx, pose_ty
                init_source = f"first_pose={first_pose} score={score_pose:.3f}"
            else:
                initial_tx, initial_ty = cent_tx, cent_ty
                init_source = f"centroid_diff score={score_cent:.3f}"
        else:
            initial_tx, initial_ty = 0.0, 0.0
            init_source = "no usable polygons"

        tx, ty, rot, score = auto_align_hydra(
            preds, gts,
            initial_tx, initial_ty,
            args.align_radius, args.align_step, args.align_try_rotations,
            args.pixel_size, warnings,
        )
        logging.info(
            "auto_align result: tx=%.3f ty=%.3f rot=%.1fdeg overlap=%.3f (initial: %s)",
            tx, ty, rot, score, init_source,
        )
        warn(
            warnings,
            f"auto_align result: tx={tx:.3f} ty={ty:.3f} rot={rot:.1f}deg "
            f"overlap={score:.3f} (initial: {init_source})",
        )
        if rot != 0.0:
            pivot_cx = statistics.mean(g.centroid[0] for g in gts if g.polygon is not None)
            pivot_cy = statistics.mean(g.centroid[1] for g in gts if g.polygon is not None)
            rotate_gt_regions_hydra(gts, rot, pivot_cx, pivot_cy)
        translate_gt_regions(gts, tx, ty)

    elif args.gt_tx != 0.0 or args.gt_ty != 0.0:
        warn(
            warnings,
            f"GT polygons translated by ({args.gt_tx:.3f}, {args.gt_ty:.3f}) "
            "before IoU computation (--gt_tx / --gt_ty). Ensure this offset "
            "correctly maps the MP3D frame to the Hydra frame.",
        )
        translate_gt_regions(gts, args.gt_tx, args.gt_ty)

    logging.info("Computing pairwise region overlaps")
    pairwise = compute_pairwise(preds, gts, args.pixel_size)

    validate_run(gt_meta["gt_region_records"], gts, preds, pairwise, warnings)

    matches, pair_lookup, assignment_method = match_regions(preds, gts, pairwise, args.iou_threshold)
    logging.info("Matching method: %s", assignment_method)
    logging.info("Matched %d regions", len(matches))

    metrics, boundary_by_pred, gt_to_preds, pred_to_gts = compute_metrics(
        preds,
        gts,
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
        "graph_json": str(hydra_graph),
        "hydra_dir": str(args.hydra_dir),
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
            "Matterport and Hydra coordinates are assumed to share the same metric global frame.",
            "Hydra room nodes without explicit footprints are approximated from connected child place positions.",
            "IoU/intersection/union are exact Shapely polygon areas when Shapely is available; otherwise they are deterministic raster estimates.",
            "mean_iou_matched is undefined (null) when there are no valid matches; penalized IoU and count metrics remain finite.",
            "overlap_diagnostics.best_pred_to_gt and best_gt_to_pred list the closest pairs by IoU regardless of threshold; they are NOT valid matches.",
        ],
    }

    export_row = build_export_row(summary, hydra_graph)
    summary["region_eval_summary"] = export_row
    write_summary_csv(output_csv, export_row)
    write_summary_json(output_json, summary)

    pairwise_csv = args.output_dir / f"region_eval_pairwise_{args.scan_id}.csv"
    pairwise_json = args.output_dir / f"region_eval_pairwise_{args.scan_id}.json"
    write_pairwise_csv_hydra(pairwise_csv, preds, gts, pairwise, matches)
    write_pairwise_json_hydra(pairwise_json, preds, gts, pairwise, matches, args.scan_id)

    print_summary(summary)
    print_overlap_diagnostics(overlap_diag, args.iou_threshold)
    print(f"\nWrote summary CSV: {output_csv}")
    print(f"Wrote summary JSON: {output_json}")
    print(f"Wrote pairwise CSV: {pairwise_csv}")
    print(f"Wrote pairwise JSON: {pairwise_json}")
    logging.info("Unmatched GT regions: %s", metrics["region_iou"]["unmatched_gt_regions"])
    logging.info("Unmatched Hydra regions: %s", metrics["region_iou"]["unmatched_predicted_regions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
