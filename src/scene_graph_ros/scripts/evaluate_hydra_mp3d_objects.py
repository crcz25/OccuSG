#!/usr/bin/env python3
"""Evaluate Hydra DSG objects against Matterport3D object ground truth.

This script mirrors the object-level metrics and one-to-one matching used by
`evaluate_mp3d_objects.py`, but reads Hydra/Hydra-ROS Dynamic Scene Graph
outputs directly from a scan directory such as `/home/crcz/.hydra/2t7WUuJeko7`.

Example:
  python3 evaluate_hydra_mp3d_objects.py --scan_id 2t7WUuJeko7

Outputs:
  object_eval_matches_<scan_id>.csv
  object_eval_summary_<scan_id>.json

Ground truth:
  Object centers and labels are read from the Matterport3D `.house` file in
  `house_segmentations`. The `region_segmentations` semseg files are loaded for
  diagnostics when present; they contain useful per-region segment labels but do
  not by themselves provide object world centroids needed by these metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


SYNONYMS = {
    "couch": "sofa",
    "dining table": "table",
    "coffee table": "table",
    "side table": "table",
    "potted plant": "plant",
    "tv": "tv monitor",
    "television": "tv monitor",
    "tv monitor": "tv monitor",
    "chest of drawers": "chest of drawers",
    "dresser": "chest of drawers",
    "night stand": "chest of drawers",
    "nightstand": "chest of drawers",
    "bookshelf": "shelving",
    "shelf": "shelving",
    "bookcase": "shelving",
    "light": "lighting",
    "lamp": "lighting",
    "door frame": "door",
    "sofa chair": "seating",
}

CONFIDENCE_KEYS = (
    "confidence",
    "score",
    "detection_score",
    "probability",
    "semantic_score",
)

POSITION_KEYS = (
    "position",
    "centroid",
    "center",
    "translation",
    "xyz",
)


def normalize_label(label: Any, apply_synonyms: bool = True) -> Optional[str]:
    if label is None:
        return None
    text = str(label).strip().lower()
    if not text:
        return None
    text = text.replace("_", " ").replace("-", " ").replace("#", " ")
    text = re.sub(r"[/\\]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if apply_synonyms:
        text = SYNONYMS.get(text, text)
    return text or None


def to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_xyz(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None

    if isinstance(value, dict):
        if {"x", "y"}.issubset(value.keys()):
            x = to_float(value.get("x"))
            y = to_float(value.get("y"))
            z = to_float(value.get("z"))
            if x is not None and y is not None:
                return (x, y) if z is None else (x, y, z)
        for key in POSITION_KEYS:
            if key in value:
                nested = extract_xyz(value.get(key))
                if nested is not None:
                    return nested

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        vals = [to_float(v) for v in value[:3]]
        if vals[0] is not None and vals[1] is not None:
            if len(vals) >= 3 and vals[2] is not None:
                return (vals[0], vals[1], vals[2])
            return (vals[0], vals[1])

    return None


def first_value(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * pct / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_vals[low]
    frac = pos - low
    return sorted_vals[low] * (1.0 - frac) + sorted_vals[high] * frac


def xyz_fields(prefix: str, xyz: Optional[Sequence[float]]) -> Dict[str, Any]:
    if xyz is None:
        return {f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": ""}
    return {
        f"{prefix}_x": xyz[0] if len(xyz) > 0 else "",
        f"{prefix}_y": xyz[1] if len(xyz) > 1 else "",
        f"{prefix}_z": xyz[2] if len(xyz) > 2 else "",
    }


def resolve_graph_json(args: argparse.Namespace) -> Path:
    if args.graph_json:
        return args.graph_json

    scan_dir = args.hydra_dir / args.scan_id
    stage_candidates = [args.hydra_stage]
    for stage in ("backend", "frontend"):
        if stage not in stage_candidates:
            stage_candidates.append(stage)

    for stage in stage_candidates:
        for name in ("dsg.json", "dsg_with_mesh.json"):
            candidate = scan_dir / stage / name
            if candidate.exists():
                return candidate

    searched = ", ".join(str(scan_dir / s / n) for s in stage_candidates for n in ("dsg.json", "dsg_with_mesh.json"))
    raise FileNotFoundError(f"No Hydra DSG JSON found. Searched: {searched}")


def resolve_scan_root(args: argparse.Namespace) -> Path:
    return args.mp3d_root / args.scan_id


def graph_labelspaces(graph: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    metadata = graph.get("metadata", {})
    raw_spaces = metadata.get("labelspaces", {}) if isinstance(metadata, dict) else {}
    labelspaces: Dict[str, Dict[int, str]] = {}
    for name, entries in raw_spaces.items():
        mapping: Dict[int, str] = {}
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    try:
                        mapping[int(entry[0])] = str(entry[1])
                    except (TypeError, ValueError):
                        continue
        if mapping:
            labelspaces[str(name)] = mapping
    return labelspaces


def labelspace_key_for_node(node: Dict[str, Any], labelspaces: Dict[str, Dict[int, str]]) -> Optional[str]:
    key = f"_l{node.get('layer')}p{node.get('partition')}"
    if key in labelspaces:
        return key
    if "mesh" in labelspaces:
        return "mesh"
    return next(iter(labelspaces.keys()), None)


def semantic_label_name(
    semantic_label: Any,
    node: Dict[str, Any],
    attrs: Dict[str, Any],
    labelspaces: Dict[str, Dict[int, str]],
) -> Tuple[Any, Optional[str], Optional[str]]:
    labelspace_key = labelspace_key_for_node(node, labelspaces)
    label_index: Optional[int] = None
    try:
        label_index = int(semantic_label)
    except (TypeError, ValueError):
        pass

    if label_index is not None and labelspace_key:
        label = labelspaces.get(labelspace_key, {}).get(label_index)
        if label is not None:
            return semantic_label, label, labelspace_key

    name = attrs.get("name")
    if name:
        return semantic_label, str(name), labelspace_key

    if semantic_label not in (None, ""):
        return semantic_label, str(semantic_label), labelspace_key

    return semantic_label, None, labelspace_key


def extract_bbox(attrs: Dict[str, Any]) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]], Optional[str]]:
    bbox = attrs.get("bounding_box")
    if not isinstance(bbox, dict):
        return None, None, None
    center = extract_xyz(bbox.get("world_P_center"))
    dims = extract_xyz(bbox.get("dimensions"))
    bbox_type = bbox.get("type")
    return center, dims, None if bbox_type is None else str(bbox_type)


def object_position(attrs: Dict[str, Any], position_source: str) -> Tuple[Optional[Tuple[float, ...]], str]:
    bbox_center, _, _ = extract_bbox(attrs)
    attr_position = extract_xyz(attrs.get("position"))

    if position_source == "bbox_center":
        return bbox_center or attr_position, "bbox_center" if bbox_center else "position_fallback"
    if position_source == "auto":
        return attr_position or bbox_center, "position" if attr_position else "bbox_center_fallback"
    return attr_position or bbox_center, "position" if attr_position else "bbox_center_fallback"


def infer_confidence(attrs: Dict[str, Any]) -> Optional[float]:
    value = first_value(attrs, CONFIDENCE_KEYS)
    confidence = to_float(value)
    if confidence is not None:
        return confidence

    probs = attrs.get("semantic_class_probabilities")
    if isinstance(probs, dict) and probs:
        numeric = [to_float(v) for v in probs.values()]
        numeric = [v for v in numeric if v is not None]
        if numeric:
            return max(numeric)
    return None


def load_hydra_objects(
    graph_json: Path,
    min_confidence: Optional[float],
    position_source: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logging.info("Loading Hydra graph: %s", graph_json)
    with graph_json.open("r", encoding="utf-8") as infile:
        graph = json.load(infile)

    if not isinstance(graph, dict):
        raise ValueError(f"Hydra graph JSON has unsupported top-level type: {type(graph).__name__}")

    labelspaces = graph_labelspaces(graph)
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("Hydra graph JSON does not contain a top-level nodes list")

    preds: List[Dict[str, Any]] = []
    type_counts = Counter()
    layer_counts = Counter()
    skipped_low_conf = 0
    skipped_no_pos = 0
    skipped_non_object = 0

    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        attrs = node.get("attributes", {})
        if not isinstance(attrs, dict):
            attrs = {}
        attr_type = attrs.get("type")
        type_counts[str(attr_type)] += 1
        layer_counts[f"layer_{node.get('layer')}_partition_{node.get('partition')}"] += 1

        if attr_type not in {"ObjectNodeAttributes", "KhronosObjectAttributes"}:
            skipped_non_object += 1
            continue

        confidence = infer_confidence(attrs)
        if min_confidence is not None and confidence is not None and confidence < min_confidence:
            skipped_low_conf += 1
            continue

        position, used_position_source = object_position(attrs, position_source)
        if position is None:
            skipped_no_pos += 1
            continue

        semantic_label = attrs.get("semantic_label")
        _, raw_label, labelspace_key = semantic_label_name(semantic_label, node, attrs, labelspaces)
        bbox_center, bbox_dims, bbox_type = extract_bbox(attrs)
        hydra_id = node.get("id", attrs.get("external_key", idx))

        preds.append(
            {
                "pred_node_id": str(hydra_id),
                "hydra_object_id": str(hydra_id),
                "raw_label": raw_label,
                "label": normalize_label(raw_label),
                "semantic_label": semantic_label,
                "labelspace_key": labelspace_key,
                "confidence": confidence,
                "position": position,
                "position_source": used_position_source,
                "bbox_center": bbox_center,
                "bbox_dimensions": bbox_dims,
                "bbox_type": bbox_type,
                "layer": node.get("layer"),
                "partition": node.get("partition"),
                "is_active": attrs.get("is_active"),
                "is_predicted": attrs.get("is_predicted"),
            }
        )

    diagnostics = {
        "graph_layout": "Hydra/Spark DSG top-level nodes list",
        "graph_top_level_keys": sorted(graph.keys()),
        "graph_json": str(graph_json),
        "layer_names": graph.get("layer_names", {}),
        "labelspaces": {k: dict(v) for k, v in labelspaces.items()},
        "labelspace_keys": sorted(labelspaces.keys()),
        "node_attribute_type_counts": dict(type_counts.most_common()),
        "node_layer_partition_counts": dict(layer_counts.most_common()),
        "num_node_entries_seen": len(nodes),
        "num_skipped_non_object": skipped_non_object,
        "num_skipped_low_confidence": skipped_low_conf,
        "num_skipped_missing_position": skipped_no_pos,
        "hydra_object_fields": {
            "id": "node.id",
            "semantic_label": "node.attributes.semantic_label mapped through graph.metadata.labelspaces",
            "position": "node.attributes.position, with AABB center fallback",
            "bounding_box": "node.attributes.bounding_box.{world_P_center,dimensions,type}",
            "confidence": "not present in observed Hydra outputs; populated only if a known score field exists",
        },
    }
    logging.info("Found %d Hydra object nodes with positions", len(preds))
    return preds, diagnostics


def candidate_house_files(scan_root: Path, scan_id: str) -> List[Path]:
    patterns = [
        f"house_segmentations/**/{scan_id}.house",
        f"**/{scan_id}.house",
        "house_segmentations/**/*.house",
        "**/*.house",
    ]
    seen = set()
    results: List[Path] = []
    for pattern in patterns:
        for path in scan_root.glob(pattern):
            if path not in seen:
                seen.add(path)
                results.append(path)
    return results


def read_house_lines(scan_root: Path, scan_id: str) -> Tuple[List[str], str]:
    files = candidate_house_files(scan_root, scan_id)
    if files:
        source = files[0]
        logging.info("Loading Matterport house file: %s", source)
        return source.read_text(encoding="utf-8").splitlines(), str(source)

    zip_path = scan_root / "house_segmentations.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith(f"/{scan_id}.house") or name.endswith(".house")
            ]
            if members:
                member = sorted(members, key=lambda name: (not name.endswith(f"/{scan_id}.house"), name))[0]
                logging.info("Loading Matterport house file from zip: %s!%s", zip_path, member)
                text = archive.read(member).decode("utf-8")
                return text.splitlines(), f"{zip_path}!{member}"

    raise FileNotFoundError(f"No .house file found under {scan_root} or {zip_path}")


def parse_house_lines(lines: Iterable[str], use_mpcat40: bool) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    categories: Dict[int, Dict[str, Any]] = {}
    objects: List[Dict[str, Any]] = []

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        record_type = parts[0]

        if record_type == "C":
            if len(parts) < 6:
                logging.warning("Skipping malformed C record on line %d", line_number)
                continue
            category_index = int(parts[1])
            categories[category_index] = {
                "category_index": category_index,
                "category_mapping_index": int(parts[2]),
                "category_mapping_name": parts[3],
                "mpcat40_index": int(parts[4]),
                "mpcat40_name": parts[5],
            }

        elif record_type == "O":
            if len(parts) < 18:
                logging.warning("Skipping malformed O record on line %d", line_number)
                continue
            object_index = int(parts[1])
            region_index = int(parts[2])
            category_index = int(parts[3])
            center = tuple(float(x) for x in parts[4:7])
            axis0 = tuple(float(x) for x in parts[7:10])
            axis1 = tuple(float(x) for x in parts[10:13])
            radii = tuple(float(x) for x in parts[13:16])
            category = categories.get(category_index, {})
            label_name = (
                category.get("mpcat40_name")
                if use_mpcat40
                else category.get("category_mapping_name")
            )

            objects.append(
                {
                    "gt_object_index": object_index,
                    "region_index": region_index,
                    "category_index": category_index,
                    "category_mapping_name": category.get("category_mapping_name"),
                    "mpcat40_name": category.get("mpcat40_name"),
                    "raw_label": label_name,
                    "label": normalize_label(label_name),
                    "center_xyz": center,
                    "radii": radii,
                    "obb_axes": [axis0, axis1],
                }
            )

    return objects, categories


def load_region_segmentation_diagnostics(scan_root: Path, scan_id: str) -> Dict[str, Any]:
    semseg_payloads: List[Tuple[str, Dict[str, Any]]] = []
    extracted_root = scan_root / "region_segmentations"
    if extracted_root.exists():
        for path in sorted(extracted_root.glob("**/*.semseg.json")):
            try:
                semseg_payloads.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
            except Exception as exc:
                logging.warning("Could not read region semseg file %s: %s", path, exc)

    zip_path = scan_root / "region_segmentations.zip"
    if not semseg_payloads and zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            for member in sorted(name for name in archive.namelist() if name.endswith(".semseg.json")):
                try:
                    semseg_payloads.append((f"{zip_path}!{member}", json.loads(archive.read(member))))
                except Exception as exc:
                    logging.warning("Could not read region semseg member %s: %s", member, exc)

    label_counts = Counter()
    object_counts_by_region: Dict[str, int] = {}
    total_objects = 0
    for source, payload in semseg_payloads:
        groups = payload.get("segGroups", []) if isinstance(payload, dict) else []
        total_objects += len(groups)
        region_match = re.search(r"region(\d+)\.semseg\.json", source)
        region_key = region_match.group(1) if region_match else source
        object_counts_by_region[region_key] = len(groups)
        for group in groups:
            if isinstance(group, dict):
                label_counts[group.get("label") or "<missing>"] += 1

    return {
        "scan_id": scan_id,
        "num_region_semseg_files": len(semseg_payloads),
        "num_region_semseg_objects": total_objects,
        "region_semseg_object_counts": object_counts_by_region,
        "region_semseg_label_counts": dict(label_counts.most_common()),
        "sources": [source for source, _ in semseg_payloads],
        "usage_note": (
            "Loaded for diagnostics. Matching uses house_segmentations .house O records "
            "because region semseg labels do not include object world centroids."
        ),
    }


def labels_compatible(pred: Dict[str, Any], gt: Dict[str, Any], ignore_labels: bool) -> bool:
    if ignore_labels:
        return True
    if not pred.get("label") or not gt.get("label"):
        return False
    return pred["label"] == gt["label"]


def distance(pred_pos: Sequence[float], gt_pos: Sequence[float]) -> Tuple[float, str]:
    if len(pred_pos) >= 3 and len(gt_pos) >= 3:
        return (
            math.sqrt(sum((float(pred_pos[i]) - float(gt_pos[i])) ** 2 for i in range(3))),
            "3d",
        )
    return (
        math.sqrt(sum((float(pred_pos[i]) - float(gt_pos[i])) ** 2 for i in range(2))),
        "2d",
    )


def greedy_assignment(candidates: List[Tuple[float, int, int]]) -> List[Tuple[int, int, float]]:
    matches = []
    used_pred = set()
    used_gt = set()
    for cost, pred_idx, gt_idx in sorted(candidates):
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, cost))
    return matches


def match_objects(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    threshold: float,
    ignore_labels: bool,
) -> Tuple[List[Tuple[int, int, float]], str]:
    candidates = []
    for pred_idx, pred in enumerate(preds):
        for gt_idx, gt in enumerate(gts):
            if not labels_compatible(pred, gt, ignore_labels):
                continue
            dist, _ = distance(pred["position"], gt["center_xyz"])
            if dist <= threshold:
                candidates.append((dist, pred_idx, gt_idx))

    logging.info("Built %d compatible match candidates within %.3f m", len(candidates), threshold)
    if not candidates:
        return [], "none"

    if linear_sum_assignment is not None and np is not None:
        big_cost = threshold + 1_000_000.0
        cost_matrix = np.full((len(preds), len(gts)), big_cost, dtype=float)
        for dist, pred_idx, gt_idx in candidates:
            cost_matrix[pred_idx, gt_idx] = dist
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for pred_idx, gt_idx in zip(row_ind, col_ind):
            cost = float(cost_matrix[pred_idx, gt_idx])
            if cost <= threshold:
                matches.append((int(pred_idx), int(gt_idx), cost))
        return matches, "hungarian"

    return greedy_assignment(candidates), "greedy"


def per_class_metrics(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
) -> Dict[str, Dict[str, Any]]:
    gt_counts = Counter(gt.get("label") or "<missing>" for gt in gts)
    pred_counts = Counter(pred.get("label") or "<missing>" for pred in preds)
    matched_gt_counts = Counter((gts[gt_idx].get("label") or "<missing>") for _, gt_idx, _ in matches)
    matched_pred_counts = Counter((preds[pred_idx].get("label") or "<missing>") for pred_idx, _, _ in matches)

    labels = sorted(set(gt_counts) | set(pred_counts))
    metrics = {}
    for label in labels:
        metrics[label] = {
            "num_gt": gt_counts[label],
            "num_pred": pred_counts[label],
            "num_matched_gt": matched_gt_counts[label],
            "num_matched_pred": matched_pred_counts[label],
            "percent_found": safe_percent(matched_gt_counts[label], gt_counts[label]),
            "percent_correct": safe_percent(matched_pred_counts[label], pred_counts[label]),
        }
    return metrics


def unmatched_label_diagnostics(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matched_pred: set,
    matched_gt: set,
) -> Dict[str, Any]:
    return {
        "unmatched_pred_labels": dict(
            Counter(
                preds[idx].get("label") or "<missing>"
                for idx in range(len(preds))
                if idx not in matched_pred
            ).most_common()
        ),
        "unmatched_gt_labels": dict(
            Counter(
                gts[idx].get("label") or "<missing>"
                for idx in range(len(gts))
                if idx not in matched_gt
            ).most_common()
        ),
    }


def match_reason(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    threshold: float,
    ignore_labels: bool,
) -> str:
    if not ignore_labels and not labels_compatible(pred, gt, ignore_labels):
        return "label_mismatch"
    dist, mode = distance(pred["position"], gt["center_xyz"])
    if dist > threshold:
        return f"nearest_compatible_gt_over_threshold_{mode}"
    return "not_selected_by_one_to_one_assignment"


def format_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(
    output_csv: Path,
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    threshold: float,
    ignore_labels: bool,
) -> None:
    fieldnames = [
        "gt_object_index",
        "hydra_object_id",
        "gt_label",
        "hydra_label",
        "match_status",
        "position_error_m",
        "class_correct",
        "hydra_confidence",
        "pred_node_id",
        "pred_label",
        "distance_m",
        "matched",
        "reason",
        "gt_label_normalized",
        "hydra_label_normalized",
        "hydra_semantic_label",
        "hydra_labelspace_key",
        "hydra_position_source",
        "hydra_bbox_type",
        "hydra_bbox_dim_x",
        "hydra_bbox_dim_y",
        "hydra_bbox_dim_z",
        "pred_x",
        "pred_y",
        "pred_z",
        "gt_x",
        "gt_y",
        "gt_z",
        "hydra_bbox_x",
        "hydra_bbox_y",
        "hydra_bbox_z",
        "gt_region_index",
        "gt_category_index",
        "gt_category_mapping_name",
        "gt_mpcat40_name",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}

    def common_row(pred: Optional[Dict[str, Any]], gt: Optional[Dict[str, Any]], dist: Optional[float]) -> Dict[str, Any]:
        class_correct = ""
        if pred is not None and gt is not None:
            class_correct = str(labels_compatible(pred, gt, False)).lower()
        row = {
            "gt_object_index": "" if gt is None else gt["gt_object_index"],
            "hydra_object_id": "" if pred is None else pred["hydra_object_id"],
            "gt_label": "" if gt is None else (gt.get("raw_label") or ""),
            "hydra_label": "" if pred is None else (pred.get("raw_label") or ""),
            "position_error_m": format_float(dist),
            "class_correct": class_correct,
            "hydra_confidence": "" if pred is None else format_float(pred.get("confidence")),
            "pred_node_id": "" if pred is None else pred["pred_node_id"],
            "pred_label": "" if pred is None else (pred.get("raw_label") or ""),
            "distance_m": format_float(dist),
            "gt_label_normalized": "" if gt is None else (gt.get("label") or ""),
            "hydra_label_normalized": "" if pred is None else (pred.get("label") or ""),
            "hydra_semantic_label": "" if pred is None else pred.get("semantic_label", ""),
            "hydra_labelspace_key": "" if pred is None else (pred.get("labelspace_key") or ""),
            "hydra_position_source": "" if pred is None else pred.get("position_source", ""),
            "hydra_bbox_type": "" if pred is None else (pred.get("bbox_type") or ""),
            "gt_region_index": "" if gt is None else gt.get("region_index", ""),
            "gt_category_index": "" if gt is None else gt.get("category_index", ""),
            "gt_category_mapping_name": "" if gt is None else (gt.get("category_mapping_name") or ""),
            "gt_mpcat40_name": "" if gt is None else (gt.get("mpcat40_name") or ""),
        }
        row.update(xyz_fields("pred", None if pred is None else pred["position"]))
        row.update(xyz_fields("gt", None if gt is None else gt["center_xyz"]))
        row.update(xyz_fields("hydra_bbox", None if pred is None else pred.get("bbox_center")))
        bbox_dims = None if pred is None else pred.get("bbox_dimensions")
        row["hydra_bbox_dim_x"] = "" if bbox_dims is None or len(bbox_dims) < 1 else bbox_dims[0]
        row["hydra_bbox_dim_y"] = "" if bbox_dims is None or len(bbox_dims) < 2 else bbox_dims[1]
        row["hydra_bbox_dim_z"] = "" if bbox_dims is None or len(bbox_dims) < 3 else bbox_dims[2]
        return row

    with output_csv.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for pred_idx, gt_idx, dist in sorted(matches, key=lambda item: item[2]):
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            row = common_row(pred, gt, dist)
            row.update({"match_status": "matched", "matched": "true", "reason": "matched"})
            writer.writerow(row)

        for pred_idx, pred in enumerate(preds):
            if pred_idx in matched_pred:
                continue
            nearest_gt_idx = None
            nearest_dist = math.inf
            for gt_idx, gt in enumerate(gts):
                if not ignore_labels and not labels_compatible(pred, gt, ignore_labels):
                    continue
                dist, _ = distance(pred["position"], gt["center_xyz"])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_gt_idx = gt_idx

            gt = None if nearest_gt_idx is None else gts[nearest_gt_idx]
            dist_value = None if nearest_gt_idx is None else nearest_dist
            row = common_row(pred, None, dist_value)
            row.update(
                {
                    "match_status": "false_positive",
                    "matched": "false",
                    "reason": "false_positive_no_compatible_gt"
                    if gt is None
                    else match_reason(pred, gt, threshold, ignore_labels),
                }
            )
            writer.writerow(row)

        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue
            nearest_pred_idx = None
            nearest_dist = math.inf
            for pred_idx, pred in enumerate(preds):
                if not ignore_labels and not labels_compatible(pred, gt, ignore_labels):
                    continue
                dist, _ = distance(pred["position"], gt["center_xyz"])
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pred_idx = pred_idx

            row = common_row(None, gt, None if nearest_pred_idx is None else nearest_dist)
            row.update(
                {
                    "match_status": "missed",
                    "matched": "false",
                    "reason": "missed_no_compatible_pred"
                    if nearest_pred_idx is None
                    else "missed_nearest_pred_over_threshold",
                }
            )
            writer.writerow(row)


def build_summary(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    match_method: str,
    args: argparse.Namespace,
    graph_json: Path,
    scan_root: Path,
    house_source: str,
    graph_diag: Dict[str, Any],
    region_diag: Dict[str, Any],
) -> Dict[str, Any]:
    errors = [dist for _, _, dist in matches]
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}
    z_available = all(len(pred["position"]) >= 3 for pred in preds) and all(len(gt["center_xyz"]) >= 3 for gt in gts)

    missing_confidence = sum(1 for pred in preds if pred.get("confidence") is None)

    return {
        "scan_id": args.scan_id,
        "graph_json": str(graph_json),
        "hydra_dir": str(args.hydra_dir),
        "hydra_scan_dir": str(args.hydra_dir / args.scan_id),
        "hydra_stage_requested": args.hydra_stage,
        "scan_root": str(scan_root),
        "house_file": house_source,
        "region_segmentations": region_diag,
        "distance_threshold_m": args.distance_threshold,
        "ignore_labels": args.ignore_labels,
        "use_mpcat40": args.use_mpcat40,
        "min_confidence": args.min_confidence,
        "hydra_position_source": args.position_source,
        "match_method": match_method,
        "coordinate_assumption": (
            "Hydra OBJECT positions and Matterport .house centers are assumed to already "
            "be in the same metric scan frame. No transform was applied."
        ),
        "used_distance_dimension": "3d" if z_available else "2d",
        "num_gt_objects": len(gts),
        "num_pred_objects": len(preds),
        "num_hydra_objects": len(preds),
        "num_matched": len(matches),
        "num_false_positive": len(preds) - len(matches),
        "num_missed": len(gts) - len(matches),
        "num_false_negative": len(gts) - len(matches),
        "percent_found": safe_percent(len(matches), len(gts)),
        "percent_correct": safe_percent(len(matches), len(preds)),
        "avg_pos_error_m": statistics.mean(errors) if errors else None,
        "median_pos_error_m": statistics.median(errors) if errors else None,
        "p90_pos_error_m": percentile(errors, 90.0),
        "per_class": per_class_metrics(preds, gts, matches),
        "unmatched_label_diagnostics": unmatched_label_diagnostics(preds, gts, matched_pred, matched_gt),
        "graph_diagnostics": graph_diag,
        "hydra_limitations": {
            "missing_confidence_count": missing_confidence,
            "confidence_note": (
                "Observed Hydra ObjectNodeAttributes do not include detector confidence; "
                "CSV hydra_confidence is blank unless a known confidence field appears."
            ),
            "region_segmentation_note": region_diag.get("usage_note"),
        },
        "label_synonyms_applied": SYNONYMS,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nHydra vs Matterport3D Object Evaluation")
    print("=======================================")
    print(f"scan_id: {summary['scan_id']}")
    print(f"Hydra graph: {summary['graph_json']}")
    print(f"GT house: {summary['house_file']}")
    print(f"GT objects: {summary['num_gt_objects']}")
    print(f"Hydra objects: {summary['num_hydra_objects']}")
    print(f"Matched: {summary['num_matched']} ({summary['match_method']})")
    print(f"False positives: {summary['num_false_positive']}")
    print(f"Missed: {summary['num_missed']}")
    print(f"percent_found: {summary['percent_found']:.2f}%")
    print(f"percent_correct: {summary['percent_correct']:.2f}%")
    avg = summary["avg_pos_error_m"]
    med = summary["median_pos_error_m"]
    p90 = summary["p90_pos_error_m"]
    print(f"avg_pos_error_m: {'n/a' if avg is None else f'{avg:.3f}'}")
    print(f"median_pos_error_m: {'n/a' if med is None else f'{med:.3f}'}")
    print(f"p90_pos_error_m: {'n/a' if p90 is None else f'{p90:.3f}'}")
    print(f"distance mode: {summary['used_distance_dimension']}")

    matched_classes = [
        (label, vals)
        for label, vals in summary["per_class"].items()
        if vals["num_matched_gt"] or vals["num_pred"]
    ]
    print("\nPer-class highlights:")
    for label, vals in sorted(
        matched_classes,
        key=lambda item: (item[1]["num_matched_gt"], item[1]["num_pred"]),
        reverse=True,
    )[:12]:
        print(
            f"  {label}: found {vals['num_matched_gt']}/{vals['num_gt']} "
            f"({vals['percent_found']:.1f}%), correct "
            f"{vals['num_matched_pred']}/{vals['num_pred']} "
            f"({vals['percent_correct']:.1f}%)"
        )


def dry_run_schema(args: argparse.Namespace) -> None:
    graph_json = resolve_graph_json(args)
    scan_root = resolve_scan_root(args)
    preds, graph_diag = load_hydra_objects(graph_json, args.min_confidence, args.position_source)
    house_lines, house_source = read_house_lines(scan_root, args.scan_id)
    gts, _ = parse_house_lines(house_lines, args.use_mpcat40)
    region_diag = load_region_segmentation_diagnostics(scan_root, args.scan_id)

    print("Dry Run Schema")
    print("==============")
    print(f"Hydra graph: {graph_json}")
    print(f"Graph top-level keys: {graph_diag['graph_top_level_keys']}")
    print(f"Node attribute type counts: {graph_diag['node_attribute_type_counts']}")
    print(f"Labelspace keys: {graph_diag['labelspace_keys']}")
    print(f"Hydra OBJECT nodes found: {len(preds)}")
    print("Sample Hydra OBJECT:")
    print(json.dumps(preds[0] if preds else None, indent=2, default=str))
    print(f"\nHouse file: {house_source}")
    print(f"GT object count: {len(gts)}")
    print("Sample GT object:")
    print(json.dumps(gts[0] if gts else None, indent=2, default=str))
    print(f"\nRegion semseg files: {region_diag['num_region_semseg_files']}")
    print(f"Region semseg objects: {region_diag['num_region_semseg_objects']}")
    print("Top region semseg labels:")
    print(json.dumps(dict(list(region_diag["region_semseg_label_counts"].items())[:20]), indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Hydra-generated OBJECT nodes against Matterport3D ground truth."
    )
    parser.add_argument("--scan_id", required=True, help="Matterport3D scan id, e.g. 2t7WUuJeko7")
    parser.add_argument("--hydra_dir", type=Path, default=Path("/home/crcz/.hydra"))
    parser.add_argument(
        "--mp3d_root",
        type=Path,
        default=Path("/mnt/DATA/repos/phd/3dsg/mp3d/dataset/v1/scans"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("."))
    parser.add_argument("--graph_json", type=Path, default=None, help="Override Hydra DSG JSON path")
    parser.add_argument("--hydra_stage", choices=("backend", "frontend"), default="backend")
    parser.add_argument("--distance_threshold", type=float, default=1.0)
    parser.add_argument("--ignore_labels", action="store_true")
    parser.add_argument(
        "--use_mpcat40",
        dest="use_mpcat40",
        action="store_true",
        default=True,
        help="Use Matterport mpcat40 names for GT labels (default).",
    )
    parser.add_argument(
        "--use_category_mapping",
        dest="use_mpcat40",
        action="store_false",
        help="Use raw Matterport category mapping names instead of mpcat40 names.",
    )
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument(
        "--position_source",
        choices=("position", "bbox_center", "auto"),
        default="position",
        help="Hydra position field to evaluate; default matches ObjectNodeAttributes.position.",
    )
    parser.add_argument("--dry_run_schema", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if args.distance_threshold <= 0:
        raise ValueError("--distance_threshold must be positive")
    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min_confidence should be in [0, 1]")
    if not args.hydra_dir.exists() and args.graph_json is None:
        raise FileNotFoundError(f"Hydra directory does not exist: {args.hydra_dir}")
    scan_root = resolve_scan_root(args)
    if not scan_root.exists():
        raise FileNotFoundError(f"Matterport3D scan root does not exist: {scan_root}")
    if args.graph_json is not None and not args.graph_json.exists():
        raise FileNotFoundError(f"Hydra graph JSON does not exist: {args.graph_json}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        validate_inputs(args)
        if args.dry_run_schema:
            dry_run_schema(args)
            return 0

        graph_json = resolve_graph_json(args)
        scan_root = resolve_scan_root(args)
        preds, graph_diag = load_hydra_objects(graph_json, args.min_confidence, args.position_source)
        house_lines, house_source = read_house_lines(scan_root, args.scan_id)
        gts, _ = parse_house_lines(house_lines, args.use_mpcat40)
        region_diag = load_region_segmentation_diagnostics(scan_root, args.scan_id)

        logging.info("Loaded %d Matterport3D .house objects", len(gts))
        logging.info(
            "Loaded %d region semseg objects across %d files for diagnostics",
            region_diag["num_region_semseg_objects"],
            region_diag["num_region_semseg_files"],
        )

        if not preds:
            logging.warning("No Hydra OBJECT nodes with positions were found")
        if not gts:
            logging.warning("No GT O records were found in the .house file")

        matches, match_method = match_objects(preds, gts, args.distance_threshold, args.ignore_labels)
        summary = build_summary(
            preds,
            gts,
            matches,
            match_method,
            args,
            graph_json,
            scan_root,
            house_source,
            graph_diag,
            region_diag,
        )
        print_summary(summary)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = args.output_dir / f"object_eval_matches_{args.scan_id}.csv"
        output_json = args.output_dir / f"object_eval_summary_{args.scan_id}.json"

        write_csv(output_csv, preds, gts, matches, args.distance_threshold, args.ignore_labels)
        logging.info("Wrote match CSV: %s", output_csv)
        with output_json.open("w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True, default=str)
        logging.info("Wrote summary JSON: %s", output_json)

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
