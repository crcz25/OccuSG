#!/usr/bin/env python3
"""Evaluate scene graph OBJECT nodes against Matterport3D .house objects.

Metric definitions:
  percent_found = 100 * matched_predictions / ground_truth_objects
  percent_correct = 100 * matched_predictions / predicted_objects
  avg_pos_error_m = mean center distance over matched one-to-one pairs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only on minimal systems
    np = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only on minimal systems
    linear_sum_assignment = None


SYNONYMS = {
    "couch": "sofa",
    "dining table": "table",
    "potted plant": "plant",
    "tv": "tv monitor",
    "television": "tv monitor",
    "tv monitor": "tv monitor",
    "chest of drawers": "chest of drawers",
    "night stand": "chest of drawers",
    "nightstand": "chest of drawers",
    "bookshelf": "shelving",
    "shelf": "shelving",
    "bookcase": "shelving",
    "light": "lighting",
    "lamp": "lighting",
}


LABEL_KEYS = (
    "semantic_label",
    "class_name",
    "class_id",
    "label",
    "name",
    "category",
    "category_name",
    "mpcat40_name",
)

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
    """Normalize labels for lightweight semantic matching."""
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
    """Extract a 2D or 3D coordinate from common JSON encodings."""
    if value is None:
        return None

    if isinstance(value, dict):
        if "position" in value:
            nested = extract_xyz(value.get("position"))
            if nested is not None:
                return nested
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


def nested_dicts(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    yield node
    for key in ("attributes", "semantic", "metadata", "data", "properties"):
        value = node.get(key)
        if isinstance(value, dict):
            yield value


def first_nested_value(node: Dict[str, Any], keys: Sequence[str]) -> Any:
    for scope in nested_dicts(node):
        for key in keys:
            if key in scope and scope.get(key) not in (None, ""):
                return scope.get(key)
    return None


def extract_position(node: Dict[str, Any]) -> Optional[Tuple[float, ...]]:
    for scope in nested_dicts(node):
        for key in POSITION_KEYS:
            if key in scope:
                xyz = extract_xyz(scope.get(key))
                if xyz is not None:
                    return xyz
        if "pose" in scope:
            xyz = extract_xyz(scope.get("pose"))
            if xyz is not None:
                return xyz
    return None


def looks_like_object_node(node: Dict[str, Any]) -> bool:
    for scope in nested_dicts(node):
        for key in ("type", "node_type", "layer"):
            value = scope.get(key)
            if isinstance(value, str) and value.strip().upper() in {
                "OBJECT",
                "OBJECTS",
                "OBJECT_NODE",
            }:
                return True
    return False


def iter_graph_nodes(graph: Any) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """Return candidate node dictionaries and a short description of the layout."""
    if isinstance(graph, dict):
        nodes = graph.get("nodes")
        if isinstance(nodes, list):
            return [(str(n.get("id", i)), n) for i, n in enumerate(nodes) if isinstance(n, dict)], (
                "top-level nodes list"
            )
        if isinstance(nodes, dict):
            return [(str(k), v) for k, v in nodes.items() if isinstance(v, dict)], (
                "top-level nodes dict"
            )

        node_like = [
            (str(k), v)
            for k, v in graph.items()
            if isinstance(v, dict) and looks_like_object_node(v)
        ]
        if node_like:
            return node_like, "top-level dict of node-like entries"

    if isinstance(graph, list):
        return [(str(i), n) for i, n in enumerate(graph) if isinstance(n, dict)], (
            "top-level list"
        )

    return [], "unrecognized graph layout"


def load_graph_objects(
    graph_json: Path, min_confidence: Optional[float], verbose: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with graph_json.open("r", encoding="utf-8") as infile:
        graph = json.load(infile)

    node_entries, layout = iter_graph_nodes(graph)
    preds = []
    skipped_low_conf = 0
    skipped_no_pos = 0

    for fallback_id, node in node_entries:
        if not looks_like_object_node(node):
            continue

        raw_conf = first_nested_value(node, CONFIDENCE_KEYS)
        confidence = to_float(raw_conf)
        if min_confidence is not None and confidence is not None and confidence < min_confidence:
            skipped_low_conf += 1
            continue

        position = extract_position(node)
        if position is None:
            skipped_no_pos += 1
            continue

        raw_label = first_nested_value(node, LABEL_KEYS)
        pred_id = first_nested_value(node, ("id", "node_id", "object_id", "uuid")) or fallback_id
        region = first_nested_value(
            node,
            (
                "region",
                "region_id",
                "room",
                "room_id",
                "parent_region",
                "parent_room",
            ),
        )

        preds.append(
            {
                "pred_node_id": str(pred_id),
                "raw_label": raw_label,
                "label": normalize_label(raw_label),
                "confidence": confidence,
                "position": position,
                "region": region,
            }
        )

    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else {}
    diagnostics = {
        "graph_layout": layout,
        "graph_top_level_keys": sorted(graph.keys()) if isinstance(graph, dict) else [],
        "graph_metadata": metadata,
        "frame_id": metadata.get("frame_id") if isinstance(metadata, dict) else None,
        "num_node_entries_seen": len(node_entries),
        "num_skipped_low_confidence": skipped_low_conf,
        "num_skipped_missing_position": skipped_no_pos,
    }
    if verbose and skipped_no_pos:
        print(f"Skipped {skipped_no_pos} OBJECT nodes without usable positions.")
    return preds, diagnostics


def locate_house_file(scan_root: Path, scan_id: Optional[str]) -> Path:
    candidates = []
    if scan_id:
        candidates.extend(scan_root.glob(f"**/{scan_id}.house"))
    candidates.extend(scan_root.glob("**/*.house"))
    unique = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    if not unique:
        raise FileNotFoundError(f"No .house file found under {scan_root}")
    if scan_id:
        exact = [p for p in unique if p.name == f"{scan_id}.house"]
        if exact:
            return exact[0]
    return unique[0]


def parse_house_file(
    house_file: Path, use_mpcat40: bool
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    categories: Dict[int, Dict[str, Any]] = {}
    objects: List[Dict[str, Any]] = []

    with house_file.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            record_type = parts[0]

            if record_type == "C":
                if len(parts) < 6:
                    print(
                        f"Warning: skipping malformed C record at {house_file}:{line_number}",
                        file=sys.stderr,
                    )
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
                    print(
                        f"Warning: skipping malformed O record at {house_file}:{line_number}",
                        file=sys.stderr,
                    )
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


def greedy_assignment(
    candidates: List[Tuple[float, int, int]]
) -> List[Tuple[int, int, float]]:
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


def safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def append_scan_id_to_path(path: Path, scan_id: Optional[str]) -> Path:
    """Insert the scan id before the file suffix unless it is already present."""
    if not scan_id:
        return path
    if scan_id in path.stem:
        return path
    return path.with_name(f"{path.stem}_{scan_id}{path.suffix}")


def infer_scan_id(args: argparse.Namespace) -> Optional[str]:
    return args.scan_id or (args.scan_root.name if args.scan_root else None)


def per_class_metrics(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
) -> Dict[str, Dict[str, Any]]:
    gt_counts = Counter(gt.get("label") or "<missing>" for gt in gts)
    pred_counts = Counter(pred.get("label") or "<missing>" for pred in preds)
    matched_gt_counts = Counter((gts[gt_idx].get("label") or "<missing>") for _, gt_idx, _ in matches)
    matched_pred_counts = Counter(
        (preds[pred_idx].get("label") or "<missing>") for pred_idx, _, _ in matches
    )

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


def xyz_fields(prefix: str, xyz: Optional[Sequence[float]]) -> Dict[str, Any]:
    if xyz is None:
        return {f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": ""}
    return {
        f"{prefix}_x": xyz[0] if len(xyz) > 0 else "",
        f"{prefix}_y": xyz[1] if len(xyz) > 1 else "",
        f"{prefix}_z": xyz[2] if len(xyz) > 2 else "",
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


def write_csv(
    output_csv: Path,
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    threshold: float,
    ignore_labels: bool,
) -> None:
    fieldnames = [
        "pred_node_id",
        "pred_label",
        "pred_x",
        "pred_y",
        "pred_z",
        "gt_object_index",
        "gt_label",
        "gt_x",
        "gt_y",
        "gt_z",
        "distance_m",
        "matched",
        "reason",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}

    with output_csv.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for pred_idx, gt_idx, dist in sorted(matches, key=lambda item: item[2]):
            pred = preds[pred_idx]
            gt = gts[gt_idx]
            row = {
                "pred_node_id": pred["pred_node_id"],
                "pred_label": pred.get("raw_label") or "",
                "gt_object_index": gt["gt_object_index"],
                "gt_label": gt.get("raw_label") or "",
                "distance_m": f"{dist:.6f}",
                "matched": "true",
                "reason": "matched",
            }
            row.update(xyz_fields("pred", pred["position"]))
            row.update(xyz_fields("gt", gt["center_xyz"]))
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
            row = {
                "pred_node_id": pred["pred_node_id"],
                "pred_label": pred.get("raw_label") or "",
                "gt_object_index": "",
                "gt_label": "",
                "distance_m": "" if nearest_gt_idx is None else f"{nearest_dist:.6f}",
                "matched": "false",
                "reason": "false_positive_no_compatible_gt"
                if nearest_gt_idx is None
                else match_reason(pred, gts[nearest_gt_idx], threshold, ignore_labels),
            }
            row.update(xyz_fields("pred", pred["position"]))
            row.update(xyz_fields("gt", None))
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
            row = {
                "pred_node_id": "",
                "pred_label": "",
                "gt_object_index": gt["gt_object_index"],
                "gt_label": gt.get("raw_label") or "",
                "distance_m": "" if nearest_pred_idx is None else f"{nearest_dist:.6f}",
                "matched": "false",
                "reason": "missed_no_compatible_pred"
                if nearest_pred_idx is None
                else "missed_nearest_pred_over_threshold",
            }
            row.update(xyz_fields("pred", None))
            row.update(xyz_fields("gt", gt["center_xyz"]))
            writer.writerow(row)


def build_summary(
    preds: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    match_method: str,
    args: argparse.Namespace,
    house_file: Path,
    graph_diag: Dict[str, Any],
) -> Dict[str, Any]:
    errors = [dist for _, _, dist in matches]
    matched_pred = {pred_idx for pred_idx, _, _ in matches}
    matched_gt = {gt_idx for _, gt_idx, _ in matches}
    z_available = all(len(pred["position"]) >= 3 for pred in preds) and all(
        len(gt["center_xyz"]) >= 3 for gt in gts
    )

    summary = {
        "scan_id": args.scan_id,
        "graph_json": str(args.graph_json),
        "scan_root": str(args.scan_root),
        "house_file": str(house_file),
        "distance_threshold_m": args.distance_threshold,
        "ignore_labels": args.ignore_labels,
        "use_mpcat40": args.use_mpcat40,
        "min_confidence": args.min_confidence,
        "match_method": match_method,
        "coordinate_assumption": (
            "Predicted OBJECT positions and Matterport .house centers are assumed "
            "to already be in the same metric scan frame. No transform was applied."
        ),
        "frame_id_reported_by_graph": graph_diag.get("frame_id"),
        "used_distance_dimension": "3d" if z_available else "2d",
        "num_gt_objects": len(gts),
        "num_pred_objects": len(preds),
        "num_matched": len(matches),
        "num_false_positive": len(preds) - len(matches),
        "num_missed": len(gts) - len(matches),
        "percent_found": safe_percent(len(matches), len(gts)),
        "percent_correct": safe_percent(len(matches), len(preds)),
        "avg_pos_error_m": statistics.mean(errors) if errors else None,
        "median_pos_error_m": statistics.median(errors) if errors else None,
        "p90_pos_error_m": percentile(errors, 90.0),
        "per_class": per_class_metrics(preds, gts, matches),
        "unmatched_label_diagnostics": unmatched_label_diagnostics(
            preds, gts, matched_pred, matched_gt
        ),
        "graph_diagnostics": graph_diag,
        "label_synonyms_applied": SYNONYMS,
    }
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nObject Detection Evaluation")
    print("===========================")
    print(f"scan_id: {summary['scan_id']}")
    print(f"GT objects: {summary['num_gt_objects']}")
    print(f"Predicted objects: {summary['num_pred_objects']}")
    print(f"Matched: {summary['num_matched']} ({summary['match_method']})")
    print(f"False positives: {summary['num_false_positive']}")
    print(f"Missed: {summary['num_missed']}")
    print(f"percent_found: {summary['percent_found']:.2f}%")
    print(f"percent_correct: {summary['percent_correct']:.2f}%")
    avg = summary["avg_pos_error_m"]
    print(f"avg_pos_error_m: {'n/a' if avg is None else f'{avg:.3f}'}")
    med = summary["median_pos_error_m"]
    p90 = summary["p90_pos_error_m"]
    print(f"median_pos_error_m: {'n/a' if med is None else f'{med:.3f}'}")
    print(f"p90_pos_error_m: {'n/a' if p90 is None else f'{p90:.3f}'}")
    print(f"distance mode: {summary['used_distance_dimension']}")
    print(f"graph frame_id: {summary['frame_id_reported_by_graph']}")

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


def dry_run_schema(
    graph_json: Path,
    scan_root: Path,
    scan_id: Optional[str],
    use_mpcat40: bool,
    min_confidence: Optional[float],
    verbose: bool,
) -> None:
    preds, graph_diag = load_graph_objects(graph_json, min_confidence, verbose)
    house_file = locate_house_file(scan_root, scan_id)
    gts, _ = parse_house_file(house_file, use_mpcat40)

    print("Dry Run Schema")
    print("==============")
    print(f"Graph JSON: {graph_json}")
    print(f"Graph top-level keys: {graph_diag['graph_top_level_keys']}")
    print(f"Graph layout: {graph_diag['graph_layout']}")
    print(f"Graph metadata: {graph_diag['graph_metadata']}")
    print(f"Candidate OBJECT nodes found: {len(preds)}")
    print("Sample OBJECT node:")
    print(json.dumps(preds[0] if preds else None, indent=2, default=str))
    print(f"\nHouse file: {house_file}")
    print(f"GT object count: {len(gts)}")
    print("Sample GT object:")
    print(json.dumps(gts[0] if gts else None, indent=2, default=str))
    print("\nSynonyms applied during normalization:")
    print(json.dumps(SYNONYMS, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scene graph OBJECT nodes against Matterport3D .house objects."
    )
    parser.add_argument("--graph_json", type=Path, required=True)
    parser.add_argument("--scan_root", type=Path, required=True)
    parser.add_argument("--scan_id", default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
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
    parser.add_argument("--dry_run_schema", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.graph_json.exists():
        raise FileNotFoundError(f"Graph JSON does not exist: {args.graph_json}")
    if not args.scan_root.exists():
        raise FileNotFoundError(f"Scan root does not exist: {args.scan_root}")
    if args.distance_threshold <= 0:
        raise ValueError("--distance_threshold must be positive")
    if args.min_confidence is not None and not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min_confidence should be in [0, 1]")


def main() -> int:
    args = parse_args()
    try:
        validate_inputs(args)
        if args.dry_run_schema:
            dry_run_schema(
                args.graph_json,
                args.scan_root,
                args.scan_id,
                args.use_mpcat40,
                args.min_confidence,
                args.verbose,
            )
            return 0

        preds, graph_diag = load_graph_objects(
            args.graph_json, args.min_confidence, args.verbose
        )
        house_file = locate_house_file(args.scan_root, args.scan_id)
        gts, _ = parse_house_file(house_file, args.use_mpcat40)

        if not preds:
            print("Warning: no predicted OBJECT nodes with positions were found.", file=sys.stderr)
        if not gts:
            print("Warning: no GT O records were found in the .house file.", file=sys.stderr)

        matches, match_method = match_objects(
            preds,
            gts,
            args.distance_threshold,
            args.ignore_labels,
        )
        summary = build_summary(
            preds, gts, matches, match_method, args, house_file, graph_diag
        )
        print_summary(summary)
        output_scan_id = infer_scan_id(args)

        if args.output_csv:
            output_csv = append_scan_id_to_path(args.output_csv, output_scan_id)
            write_csv(
                output_csv,
                preds,
                gts,
                matches,
                args.distance_threshold,
                args.ignore_labels,
            )
            print(f"\nWrote match CSV: {output_csv}")

        if args.output_json:
            output_json = append_scan_id_to_path(args.output_json, output_scan_id)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with output_json.open("w", encoding="utf-8") as outfile:
                json.dump(summary, outfile, indent=2, sort_keys=True, default=str)
            print(f"Wrote summary JSON: {output_json}")

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
