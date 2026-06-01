#!/usr/bin/env python3
"""Collect Matterport3D evaluation outputs into CSV tables.

This script only aggregates existing evaluation artifacts. It does not import
ROS packages and does not re-run the object or region evaluators.

python3 /workspace/occusg_ws/src/scene_graph_ros/scripts/collect_mp3d_eval_tables.py \
  --scans_root /workspace/occusg_ws/mp3d/dataset/v1/scans \
  --output_dir /workspace/occusg_ws/mp3d/dataset/v1/ \
  --verbose
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


OBJECT_CSV_COLUMNS = [
    "scan_id",
    "configuration",
    "percent_found",
    "percent_correct",
    "avg_pos_error_m",
    "num_gt_objects",
    "num_pred_objects",
    "num_matched",
    "num_false_positive",
    "num_missed",
    "source_file",
]

REGION_CSV_COLUMNS = [
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
]

REGION_DATASET_SUMMARY_COLUMNS = [
    "method",
    "num_scans_evaluated",
    "num_successful_scans",
    "num_failed_scans",
    "avg_num_gt_regions",
    "avg_num_pred_regions",
    "mean_region_count_error",
    "median_region_count_error",
    "mean_region_recall",
    "median_region_recall",
    "mean_region_precision",
    "median_region_precision",
    "mean_region_f1",
    "median_region_f1",
    "mean_penalized_iou",
    "median_penalized_iou",
    "failure_rate",
    "single_region_failure_rate",
    "zero_match_failure_rate",
    "severe_undersegmentation_rate",
]

OBJECT_METRIC_KEYS = {
    "percent_found": ("percent_found", "found_percent", "recall_percent"),
    "percent_correct": ("percent_correct", "correct_percent", "precision_percent"),
    "avg_pos_error_m": (
        "avg_pos_error_m",
        "mean_pos_error_m",
        "average_position_error_m",
        "mean_position_error_m",
    ),
    "num_gt_objects": ("num_gt_objects", "gt_objects", "n_gt_objects"),
    "num_pred_objects": ("num_pred_objects", "pred_objects", "predicted_objects", "n_pred_objects"),
    "num_matched": ("num_matched", "matched_objects", "num_matched_objects", "n_matched"),
    "num_false_positive": (
        "num_false_positive",
        "num_false_positives",
        "false_positive",
        "false_positives",
    ),
    "num_missed": ("num_missed", "missed_objects", "num_missed_objects", "false_negatives"),
}

REGION_METRIC_KEYS = {key: (key,) for key in REGION_CSV_COLUMNS if key != "source_file"}

MISSING = float("nan")


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if value == "":
        return True
    return False


def to_float(value: Any) -> float:
    if value is None or value == "":
        return MISSING
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return MISSING
    return parsed if math.isfinite(parsed) else MISSING


def to_int(value: Any) -> Any:
    if value is None or value == "":
        return MISSING
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return MISSING
    return parsed


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "matched"}


def zero_if_missing(value: Any) -> float:
    return 0.0 if is_missing(value) else float(value)


def safe_ratio(numerator: Any, denominator: Any) -> float:
    if is_missing(numerator) or is_missing(denominator) or float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def classify_failure_mode(gt_count: Any, pred_count: Any, matched_count: Any, gt_coverage: Any, count_ratio: Any) -> Tuple[str, bool]:
    if is_missing(gt_count):
        return "", False
    gt = int(float(gt_count))
    pred = 0 if is_missing(pred_count) else int(float(pred_count))
    matched = 0 if is_missing(matched_count) else int(float(matched_count))
    coverage = zero_if_missing(gt_coverage)
    ratio = zero_if_missing(count_ratio)
    severe = gt > 0 and (ratio < 0.25 or coverage < 0.25)
    if gt == 0:
        return "NO_GT_REGIONS", False
    if pred == 0:
        return "NO_PREDICTIONS", severe
    if matched == 0:
        return "NO_MATCHES", severe
    if severe:
        return "SEVERE_UNDER_SEGMENTATION", True
    if coverage < 1.0:
        return "PARTIAL_MATCH_ONLY", False
    return "VALID_MATCHES", False


def nested_get_any(data: Any, key_candidates: Sequence[str]) -> Any:
    """Search dictionaries and nested structures for the first candidate key."""
    candidates = {key.lower() for key in key_candidates}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in candidates and value not in (None, ""):
                    if isinstance(value, (dict, list)):
                        nested = walk(value)
                        if nested is not None:
                            return nested
                        continue
                    return value
            for value in node.values():
                found = walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = walk(value)
                if found is not None:
                    return found
        return None

    return walk(data)


def flatten_keys(data: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            yield full_key
            yield from flatten_keys(value, full_key)
    elif isinstance(data, list):
        for item in data[:3]:
            yield from flatten_keys(item, prefix)


def file_priority(path: Path, kind: str) -> Tuple[int, int, int, int, float]:
    name = path.name.lower()
    suffix_score = 0 if path.suffix.lower() == ".json" else 1
    summary_score = 0 if "summary" in name else 1
    eval_score = 0 if "eval" in name else 1
    kind_terms = ("object", "detection") if kind == "object" else ("region", "segmentation")
    kind_score = 0 if any(term in name for term in kind_terms) else 1
    try:
        mtime = -path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (suffix_score, summary_score, eval_score, kind_score, mtime)


def find_candidate_files(directory: Path, kind: str) -> List[Path]:
    if not directory.exists() or not directory.is_dir():
        return []

    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv"}
    ]
    preferred_terms = ("object", "detection", "eval", "summary", "match", "metric") if kind == "object" else (
        "region",
        "segmentation",
        "eval",
        "summary",
        "match",
        "metric",
    )
    likely = [path for path in files if any(term in path.name.lower() for term in preferred_terms)]
    candidates = likely or files
    return sorted(candidates, key=lambda path: file_priority(path, kind))


def extract_object_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for output_key, key_candidates in OBJECT_METRIC_KEYS.items():
        value = nested_get_any(data, key_candidates)
        if output_key.startswith("num_"):
            metrics[output_key] = to_int(value)
        else:
            metrics[output_key] = to_float(value)
    return metrics


def extract_region_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for output_key, key_candidates in REGION_METRIC_KEYS.items():
        value = nested_get_any(data, key_candidates)
        if output_key in {"scan_id", "method"}:
            metrics[output_key] = "" if value is None or isinstance(value, (dict, list)) else str(value)
        elif output_key == "failure_mode":
            metrics[output_key] = "" if value is None or isinstance(value, (dict, list)) else str(value)
        elif output_key in {
            "severe_undersegmentation",
            "no_predicted_regions",
            "no_valid_matches",
            "single_region_failure",
            "region_count_collapse",
        }:
            metrics[output_key] = MISSING if value is None or isinstance(value, (dict, list)) else truthy(value)
        elif output_key in {
            "num_gt_regions",
            "num_pred_regions",
            "num_matches",
            "num_unmatched_gt",
            "num_unmatched_pred",
            "region_count_error_abs",
        }:
            metrics[output_key] = to_int(value)
        else:
            metrics[output_key] = to_float(value)
    return metrics


def value_to_string(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def extract_object_metrics_from_csv(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    pred_ids = {
        row.get("pred_node_id", "").strip()
        for row in rows
        if row.get("pred_node_id", "").strip()
    }
    gt_ids = {
        row.get("gt_object_index", "").strip()
        for row in rows
        if row.get("gt_object_index", "").strip()
    }
    matched_rows = [row for row in rows if truthy(row.get("matched", ""))]
    false_positive_rows = [
        row
        for row in rows
        if not truthy(row.get("matched", ""))
        and row.get("pred_node_id", "").strip()
        and not row.get("gt_object_index", "").strip()
    ]
    missed_rows = [
        row
        for row in rows
        if not truthy(row.get("matched", ""))
        and row.get("gt_object_index", "").strip()
        and not row.get("pred_node_id", "").strip()
    ]
    distances = [to_float(row.get("distance_m")) for row in matched_rows]
    distances = [value for value in distances if not is_missing(value)]

    num_gt = len(gt_ids) if gt_ids else MISSING
    num_pred = len(pred_ids) if pred_ids else MISSING
    num_matched = len(matched_rows)
    return {
        "percent_found": 100.0 * num_matched / num_gt if not is_missing(num_gt) and num_gt else MISSING,
        "percent_correct": 100.0 * num_matched / num_pred if not is_missing(num_pred) and num_pred else MISSING,
        "avg_pos_error_m": statistics.mean(distances) if distances else MISSING,
        "num_gt_objects": num_gt,
        "num_pred_objects": num_pred,
        "num_matched": num_matched,
        "num_false_positive": len(false_positive_rows) if false_positive_rows else MISSING,
        "num_missed": len(missed_rows) if missed_rows else MISSING,
    }


def extract_region_metrics_from_csv(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    missing = [key for key in REGION_CSV_COLUMNS if key != "source_file" and key not in rows[0]]
    if missing:
        raise RuntimeError(f"{path}: not a canonical region summary CSV; missing {', '.join(missing)}")
    return extract_region_metrics(rows[0])


def metric_score(metrics: Dict[str, Any], required_keys: Sequence[str]) -> int:
    return sum(0 if is_missing(metrics.get(key)) else 1 for key in required_keys)


def load_metrics_from_candidate(path: Path, kind: str) -> Tuple[Dict[str, Any], List[str]]:
    keys: List[str] = []
    if path.suffix.lower() == ".json":
        data = load_json(path)
        keys = sorted(flatten_keys(data))
        if kind == "object":
            return extract_object_metrics(data), keys
        return extract_region_metrics(data), keys
    if kind == "object":
        return extract_object_metrics_from_csv(path), []
    return extract_region_metrics_from_csv(path), []


def select_metrics_file(
    directory: Path,
    kind: str,
    required_keys: Sequence[str],
    label: str,
    strict: bool,
) -> Tuple[Optional[Path], Dict[str, Any], List[str]]:
    candidates = find_candidate_files(directory, kind)
    if not candidates:
        message = f"{label}: no {kind} JSON/CSV result files found under {directory}"
        if strict:
            raise FileNotFoundError(message)
        warn(message)
        return None, {}, []

    best_path: Optional[Path] = None
    best_metrics: Dict[str, Any] = {}
    best_keys: List[str] = []
    best_score = -1
    load_errors: List[str] = []

    for path in candidates:
        try:
            metrics, keys = load_metrics_from_candidate(path, kind)
        except (OSError, json.JSONDecodeError, csv.Error) as exc:
            load_errors.append(f"{path}: {exc}")
            continue
        score = metric_score(metrics, required_keys)
        if score > best_score:
            best_path = path
            best_metrics = metrics
            best_keys = keys
            best_score = score
        if score == len(required_keys):
            break

    if best_path is None or best_score <= 0:
        message = f"{label}: found files but could not extract {kind} metrics"
        if load_errors:
            message += f"; load errors: {'; '.join(load_errors[:3])}"
        if strict:
            raise RuntimeError(message)
        warn(message)
        return None, {}, []

    ambiguous = [
        path
        for path in candidates[:5]
        if path != best_path and path.suffix.lower() == best_path.suffix.lower()
    ]
    if ambiguous:
        warn(f"{label}: selected {best_path}; other nearby candidates: {', '.join(str(p) for p in ambiguous[:3])}")
    return best_path, best_metrics, best_keys


def parse_object_models(values: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise argparse.ArgumentTypeError(
                f"Invalid --object_models entry '{value}'. Expected directory:configuration."
            )
        directory, configuration = value.split(":", 1)
        if not directory or not configuration:
            raise argparse.ArgumentTypeError(
                f"Invalid --object_models entry '{value}'. Expected directory:configuration."
            )
        pairs.append((directory, configuration))
    return pairs


def discover_scans(scans_root: Path) -> List[Path]:
    return sorted([path for path in scans_root.iterdir() if path.is_dir()], key=lambda p: p.name)


def empty_object_row(scan_id: str, configuration: str) -> Dict[str, Any]:
    return {
        "scan_id": scan_id,
        "configuration": configuration,
        "percent_found": MISSING,
        "percent_correct": MISSING,
        "avg_pos_error_m": MISSING,
        "num_gt_objects": MISSING,
        "num_pred_objects": MISSING,
        "num_matched": MISSING,
        "num_false_positive": MISSING,
        "num_missed": MISSING,
        "source_file": "",
    }


def empty_region_row(scan_id: str, method: str) -> Dict[str, Any]:
    return {
        "scan_id": scan_id,
        "method": method,
        "num_gt_regions": MISSING,
        "num_pred_regions": MISSING,
        "num_matches": MISSING,
        "num_unmatched_gt": MISSING,
        "num_unmatched_pred": MISSING,
        "region_count_error_abs": MISSING,
        "region_count_error_rel": MISSING,
        "region_count_ratio": MISSING,
        "region_precision": MISSING,
        "region_recall": MISSING,
        "region_f1": MISSING,
        "gt_coverage": MISSING,
        "mean_iou_matched": MISSING,
        "mean_iou_gt_penalized": MISSING,
        "mean_iou_full_penalized": MISSING,
        "oversegmentation_rate": MISSING,
        "undersegmentation_rate": MISSING,
        "no_predicted_regions": False,
        "no_valid_matches": False,
        "single_region_failure": False,
        "severe_undersegmentation": False,
        "region_count_collapse": False,
        "failure_mode": "",
        "source_file": "",
    }


def collect_object_rows(
    scans: Sequence[Path],
    object_models: Sequence[Tuple[str, str]],
    strict: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    keys_used: Dict[str, List[str]] = {}
    required = ("percent_found", "percent_correct", "avg_pos_error_m")

    for scan in scans:
        for model_dir, configuration in object_models:
            row = empty_object_row(scan.name, configuration)
            result_dir = scan / model_dir
            label = f"{scan.name}/{model_dir}"
            if not result_dir.is_dir():
                message = f"{label}: missing model result directory"
                if strict:
                    raise FileNotFoundError(message)
                warn(message)
                rows.append(row)
                continue

            source, metrics, keys = select_metrics_file(result_dir, "object", required, label, strict)
            if source is not None:
                row.update(metrics)
                row["source_file"] = str(source)
                keys_used[str(source)] = keys or list(metrics.keys())
                print(f"Selected object source for {scan.name}/{configuration}: {source}")
                missing = [key for key in required if is_missing(row.get(key))]
                if missing:
                    warn(f"{label}: missing object metrics {', '.join(missing)} in {source}")
            elif verbose:
                print(f"No object source selected for {scan.name}/{configuration}")
            rows.append(row)
    return rows, keys_used


def collect_region_rows(
    scans: Sequence[Path],
    strict: bool,
    verbose: bool,
    method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    keys_used: Dict[str, List[str]] = {}
    required = (
        "num_gt_regions",
        "num_pred_regions",
        "num_matches",
        "region_recall",
        "mean_iou_gt_penalized",
        "failure_mode",
    )

    for scan in scans:
        row = empty_region_row(scan.name, method)
        result_dir = scan / "regions"
        label = f"{scan.name}/regions"
        if not result_dir.is_dir():
            message = f"{label}: missing region result directory"
            if strict:
                raise FileNotFoundError(message)
            warn(message)
            rows.append(row)
            continue

        source, metrics, keys = select_metrics_file(result_dir, "region", required, label, strict)
        if source is not None:
            row.update(metrics)
            if not row.get("method"):
                row["method"] = method
            row["source_file"] = str(source)
            keys_used[str(source)] = keys or list(metrics.keys())
            print(f"Selected region source for {scan.name}: {source}")
            missing = [key for key in required if is_missing(row.get(key))]
            if missing:
                warn(f"{label}: missing region metrics {', '.join(missing)} in {source}")
        elif verbose:
            print(f"No region source selected for {scan.name}")
        rows.append(row)
    return rows, keys_used


def csv_value(value: Any) -> Any:
    if value == "":
        return ""
    if is_missing(value):
        return "NaN"
    return value


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column, "")) for column in columns})


def preview_csv(path: Path, rows: int = 5) -> None:
    print(f"\nPreview {path}:")
    with path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        for index, row in enumerate(reader):
            if index > rows:
                break
            print(",".join(row))


def validate_outputs(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Output is empty or missing: {path}")
        print(f"Confirmed non-empty output: {path}")


def finite_values(rows: Sequence[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if is_missing(value):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            values.append(parsed)
    return values


def mean_or_missing(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else MISSING


def median_or_missing(values: Sequence[float]) -> float:
    return statistics.median(values) if values else MISSING


def bool_rate(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if truthy(row.get(key))) / len(rows) if rows else MISSING


def build_region_dataset_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-scan region metrics without dropping quality failures."""
    summaries: List[Dict[str, Any]] = []
    methods = sorted({str(row.get("method") or "mine") for row in rows})
    for method in methods:
        method_rows = [row for row in rows if str(row.get("method") or "mine") == method]
        successful = [
            row
            for row in method_rows
            if not is_missing(row.get("num_gt_regions"))
            and not is_missing(row.get("num_pred_regions"))
            and not is_missing(row.get("num_matches"))
        ]
        failed_count = len(method_rows) - len(successful)
        count_error = finite_values(successful, "region_count_error_abs")
        recall = finite_values(successful, "region_recall")
        precision = finite_values(successful, "region_precision")
        f1 = finite_values(successful, "region_f1")
        penalized_iou = finite_values(successful, "mean_iou_gt_penalized")
        summaries.append(
            {
                "method": method,
                "num_scans_evaluated": len(method_rows),
                "num_successful_scans": len(successful),
                "num_failed_scans": failed_count,
                "avg_num_gt_regions": mean_or_missing(finite_values(successful, "num_gt_regions")),
                "avg_num_pred_regions": mean_or_missing(finite_values(successful, "num_pred_regions")),
                "mean_region_count_error": mean_or_missing(count_error),
                "median_region_count_error": median_or_missing(count_error),
                "mean_region_recall": mean_or_missing(recall),
                "median_region_recall": median_or_missing(recall),
                "mean_region_precision": mean_or_missing(precision),
                "median_region_precision": median_or_missing(precision),
                "mean_region_f1": mean_or_missing(f1),
                "median_region_f1": median_or_missing(f1),
                "mean_penalized_iou": mean_or_missing(penalized_iou),
                "median_penalized_iou": median_or_missing(penalized_iou),
                "failure_rate": failed_count / len(method_rows) if method_rows else MISSING,
                "single_region_failure_rate": bool_rate(successful, "single_region_failure"),
                "zero_match_failure_rate": bool_rate(successful, "no_valid_matches"),
                "severe_undersegmentation_rate": bool_rate(successful, "severe_undersegmentation"),
            }
        )
    return summaries


def print_key_note(title: str, keys_by_source: Dict[str, List[str]]) -> None:
    print(f"\n{title}:")
    if not keys_by_source:
        print("  No source files selected.")
        return
    for source, keys in sorted(keys_by_source.items()):
        selected_keys = [key for key in keys if re.search(r"(percent|error|num_|iou|boundary|adjacency|segmentation|matched)", key)]
        selected_keys = selected_keys[:12] or keys[:12]
        print(f"  {source}")
        print(f"    keys: {', '.join(selected_keys)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate existing MP3D object and region evaluation outputs into tables."
    )
    parser.add_argument("--scans_root", type=Path, required=True)
    parser.add_argument(
        "--scripts_dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the evaluation scripts; used for provenance checks only.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--object_models",
        nargs="+",
        default=["y11:yolo11x", "y26:yolo26x"],
        help="Object model directory/name mappings such as y11:yolo11x y26:yolo26x.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--method_name", default="mine")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    object_models = parse_object_models(args.object_models)

    if not args.scans_root.is_dir():
        raise FileNotFoundError(f"scans_root does not exist or is not a directory: {args.scans_root}")
    if args.verbose:
        print(f"Using scripts_dir: {args.scripts_dir}")
        for script_name in ("evaluate_mp3d_objects.py", "evaluate_mp3d_regions.py"):
            script = args.scripts_dir / script_name
            print(f"  {script_name}: {'found' if script.exists() else 'missing'}")

    scans = discover_scans(args.scans_root)
    print(f"Discovered {len(scans)} scan directories under {args.scans_root}")
    if args.verbose or args.dry_run:
        for scan in scans:
            present = []
            for model_dir, _ in object_models:
                if (scan / model_dir).is_dir():
                    present.append(model_dir)
            if (scan / "regions").is_dir():
                present.append("regions")
            print(f"  {scan.name}: {', '.join(present) if present else 'no eval result dirs'}")

    object_rows, object_keys = collect_object_rows(scans, object_models, args.strict, args.verbose)
    region_rows, region_keys = collect_region_rows(scans, args.strict, args.verbose, args.method_name)

    print_key_note("Detected object result files and metric keys", object_keys)
    print_key_note("Detected region result files and metric keys", region_keys)

    if args.dry_run:
        print("\nDry run requested; no files written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_csv = args.output_dir / "object_detection_results.csv"
    region_csv = args.output_dir / "region_segmentation_results.csv"
    region_summary_csv = args.output_dir / "region_segmentation_dataset_summary.csv"

    write_csv(object_csv, OBJECT_CSV_COLUMNS, object_rows)
    write_csv(region_csv, REGION_CSV_COLUMNS, region_rows)
    write_csv(region_summary_csv, REGION_DATASET_SUMMARY_COLUMNS, build_region_dataset_summary(region_rows))

    preview_csv(object_csv)
    preview_csv(region_csv)
    preview_csv(region_summary_csv)
    validate_outputs([object_csv, region_csv, region_summary_csv])

    print("\nSummary:")
    print(f"  scans discovered: {len(scans)}")
    print(f"  object result rows written: {len(object_rows)}")
    print(f"  region result rows written: {len(region_rows)}")
    print(f"  wrote: {object_csv}")
    print(f"  wrote: {region_csv}")
    print(f"  wrote: {region_summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
