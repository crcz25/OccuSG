#!/usr/bin/env python3
"""Collect Hydra MP3D evaluator outputs into CSV tables.

This script only aggregates existing evaluator CSV/JSON files. It does not
launch ROS nodes, run evaluators, inspect Hydra outputs, or read DSG files.

Example:
  python3 /mnt/DATA/repos/phd/3dsg/src/scene_graph_ros/scripts/collect_hydra_mp3d_eval_tables.py \
    --results_dir /mnt/DATA/repos/phd/hydra_ws/results/evals \
    --output_dir /mnt/DATA/repos/phd/hydra_ws/results \
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
from dataclasses import dataclass
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
    "missing_artifact_warnings",
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
    "missing_artifact_warnings",
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
    "percent_found": ("percent_found", "found_percent", "recall_percent", "object_recall"),
    "percent_correct": ("percent_correct", "correct_percent", "precision_percent", "object_precision"),
    "avg_pos_error_m": (
        "avg_pos_error_m",
        "mean_pos_error_m",
        "average_position_error_m",
        "mean_position_error_m",
        "avg_position_error_m",
    ),
    "num_gt_objects": ("num_gt_objects", "gt_objects", "n_gt_objects", "ground_truth_objects"),
    "num_pred_objects": (
        "num_pred_objects",
        "pred_objects",
        "predicted_objects",
        "n_pred_objects",
        "num_hydra_objects",
        "hydra_objects",
        "num_objects",
        "objects",
    ),
    "num_matched": ("num_matched", "matched_objects", "num_matched_objects", "n_matched"),
    "num_false_positive": (
        "num_false_positive",
        "num_false_positives",
        "false_positive",
        "false_positives",
    ),
    "num_missed": (
        "num_missed",
        "missed_objects",
        "num_missed_objects",
        "num_false_negative",
        "false_negative",
        "false_negatives",
    ),
}

REGION_METRIC_KEYS = {
    key: (key,)
    for key in REGION_CSV_COLUMNS
    if key not in {"source_file", "missing_artifact_warnings"}
}

EVAL_FILE_PATTERN = re.compile(
    r"^(object|region)_eval_(summary|matches)_(?P<scan_id>[^.]+)\.(json|csv)$"
)
MISSING = float("nan")
@dataclass(frozen=True)
class EvalArtifact:
    path: Path
    kind: str
    artifact_type: str
    scan_id: str


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


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
    if isinstance(value, str) and value.strip().lower() in {"nan", "none", "null", "--"}:
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
        return int(float(value))
    except (TypeError, ValueError):
        return MISSING


def value_to_string(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value)


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


def normalize_key(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def nested_get_any(data: Any, key_candidates: Sequence[str]) -> Any:
    candidates = {normalize_key(key) for key in key_candidates}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key, value in node.items():
                if normalize_key(key) in candidates and value not in (None, ""):
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def extract_object_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for output_key, key_candidates in OBJECT_METRIC_KEYS.items():
        value = nested_get_any(data, key_candidates)
        metrics[output_key] = to_int(value) if output_key.startswith("num_") else to_float(value)
    return normalize_object_metric_definitions(metrics)


def normalize_object_metric_definitions(metrics: Dict[str, Any]) -> Dict[str, Any]:
    num_gt = metrics.get("num_gt_objects")
    num_pred = metrics.get("num_pred_objects")
    num_matched = metrics.get("num_matched")

    if is_missing(metrics.get("percent_found")) and not is_missing(num_gt) and num_gt:
        metrics["percent_found"] = 100.0 * num_matched / num_gt if not is_missing(num_matched) else MISSING
    if is_missing(metrics.get("percent_correct")) and not is_missing(num_pred) and num_pred:
        metrics["percent_correct"] = 100.0 * num_matched / num_pred if not is_missing(num_matched) else MISSING
    if is_missing(metrics.get("num_false_positive")) and not is_missing(num_pred) and not is_missing(num_matched):
        metrics["num_false_positive"] = int(num_pred) - int(num_matched)
    if is_missing(metrics.get("num_missed")) and not is_missing(num_gt) and not is_missing(num_matched):
        metrics["num_missed"] = int(num_gt) - int(num_matched)
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
        "num_false_positive": len(false_positive_rows),
        "num_missed": len(missed_rows),
    }


def extract_region_metrics_from_csv(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}
    missing = [
        key
        for key in REGION_CSV_COLUMNS
        if key not in {"source_file", "missing_artifact_warnings"} and key not in rows[0]
    ]
    if missing:
        raise RuntimeError(f"{path}: not a canonical region summary CSV; missing {', '.join(missing)}")
    return extract_region_metrics(rows[0])


def scan_id_from_eval_filename(path: Path) -> Optional[str]:
    match = EVAL_FILE_PATTERN.match(path.name)
    return match.group("scan_id") if match else None


def discover_eval_artifacts(results_dir: Path) -> List[EvalArtifact]:
    artifacts: List[EvalArtifact] = []
    files = (p for p in results_dir.rglob("*") if p.is_file())
    for path in sorted(files, key=lambda p: p.relative_to(results_dir).as_posix()):
        match = EVAL_FILE_PATTERN.match(path.name)
        if not match:
            continue
        kind, artifact_type = match.group(1), match.group(2)
        if artifact_type == "summary" and path.suffix.lower() not in {".json", ".csv"}:
            continue
        if artifact_type == "matches" and path.suffix.lower() != ".csv":
            continue
        artifacts.append(
            EvalArtifact(
                path=path,
                kind=kind,
                artifact_type=artifact_type,
                scan_id=match.group("scan_id"),
            )
        )
    return artifacts


def group_artifacts(
    artifacts: Sequence[EvalArtifact],
) -> Dict[str, Dict[str, Dict[str, EvalArtifact]]]:
    grouped: Dict[str, Dict[str, Dict[str, EvalArtifact]]] = {}
    for artifact in artifacts:
        scan_group = grouped.setdefault(artifact.scan_id, {"object": {}, "region": {}})
        kind_group = scan_group[artifact.kind]
        existing = kind_group.get(artifact.artifact_type)
        if existing is not None:
            if artifact.artifact_type == "summary" and existing.path.suffix.lower() != artifact.path.suffix.lower():
                if artifact.path.suffix.lower() == ".json":
                    kind_group[artifact.artifact_type] = artifact
                continue
            raise RuntimeError(
                f"Duplicate {artifact.kind} {artifact.artifact_type} artifact for scan {artifact.scan_id}\n"
                f"first source: {existing.path}\n"
                f"second source: {artifact.path}"
            )
        kind_group[artifact.artifact_type] = artifact
    return grouped


def select_source(
    grouped_scan: Dict[str, Dict[str, EvalArtifact]],
    kind: str,
) -> Optional[EvalArtifact]:
    return grouped_scan[kind].get("summary") or grouped_scan[kind].get("matches")


def validate_summary_scan_id(data: Any, artifact: EvalArtifact) -> None:
    if artifact.artifact_type != "summary" or artifact.path.suffix.lower() != ".json":
        return
    if not isinstance(data, dict) or "scan_id" not in data:
        return
    summary_scan_id = str(data["scan_id"])
    if summary_scan_id != artifact.scan_id:
        raise RuntimeError(
            f"{artifact.kind} summary scan_id mismatch\n"
            f"filename scan_id: {artifact.scan_id}\n"
            f"summary scan_id: {summary_scan_id}\n"
            f"source: {artifact.path}"
        )


def validate_csv_scan_id(artifact: EvalArtifact) -> None:
    rows = read_csv_rows(artifact.path)
    scan_ids = {row.get("scan_id", "").strip() for row in rows if row.get("scan_id", "").strip()}
    unexpected = sorted(scan_id for scan_id in scan_ids if scan_id != artifact.scan_id)
    if unexpected:
        raise RuntimeError(
            f"{artifact.kind} match CSV scan_id mismatch\n"
            f"filename scan_id: {artifact.scan_id}\n"
            f"CSV scan_id values: {', '.join(unexpected)}\n"
            f"source: {artifact.path}"
        )


def load_metrics_from_artifact(artifact: EvalArtifact) -> Tuple[Dict[str, Any], List[str]]:
    suffix = artifact.path.suffix.lower()
    if suffix == ".json":
        data = read_json(artifact.path)
        validate_summary_scan_id(data, artifact)
        keys = sorted(flatten_keys(data))
        metrics = extract_object_metrics(data) if artifact.kind == "object" else extract_region_metrics(data)
        if artifact.kind == "object":
            configuration = configuration_from_summary(data)
            if configuration:
                metrics["configuration"] = configuration
        return metrics, keys
    if suffix == ".csv":
        validate_csv_scan_id(artifact)
        metrics = (
            extract_object_metrics_from_csv(artifact.path)
            if artifact.kind == "object"
            else extract_region_metrics_from_csv(artifact.path)
        )
        return metrics, list(metrics.keys())
    raise RuntimeError(f"Unsupported evaluator artifact suffix: {artifact.path}")


def configuration_from_summary(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    value = nested_get_any(
        data,
        (
            "configuration",
            "config",
            "eval_configuration",
            "stage",
            "hydra_stage",
            "hydra_stage_requested",
        ),
    )
    if value is None or isinstance(value, (dict, list)):
        return ""
    text = str(value).strip()
    return text or ""


def empty_object_row(scan_id: str, warnings: Sequence[str]) -> Dict[str, Any]:
    return {
        "scan_id": scan_id,
        "configuration": "hydra",
        "percent_found": MISSING,
        "percent_correct": MISSING,
        "avg_pos_error_m": MISSING,
        "num_gt_objects": MISSING,
        "num_pred_objects": MISSING,
        "num_matched": MISSING,
        "num_false_positive": MISSING,
        "num_missed": MISSING,
        "source_file": "",
        "missing_artifact_warnings": "; ".join(dict.fromkeys(warnings)),
    }


def empty_region_row(scan_id: str, warnings: Sequence[str]) -> Dict[str, Any]:
    return {
        "scan_id": scan_id,
        "method": "hydra",
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
        "missing_artifact_warnings": "; ".join(dict.fromkeys(warnings)),
    }


def missing_metrics(row: Dict[str, Any], keys: Sequence[str]) -> List[str]:
    return [key for key in keys if is_missing(row.get(key))]


def collect_object_rows(
    grouped: Dict[str, Dict[str, Dict[str, EvalArtifact]]],
    strict: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    rows: List[Dict[str, Any]] = []
    keys_used: Dict[str, List[str]] = {}
    required = ("percent_found", "percent_correct", "avg_pos_error_m")

    for scan_id in sorted(grouped):
        warnings: List[str] = []
        source = select_source(grouped[scan_id], "object")
        if source is None:
            message = "missing object evaluator artifact"
            if strict:
                raise FileNotFoundError(f"{scan_id}: {message}")
            warnings.append(message)
            row = empty_object_row(scan_id, warnings)
            if verbose:
                print(f"No object source selected for {scan_id}/hydra")
            rows.append(row)
            continue

        if source.artifact_type == "matches":
            warnings.append("missing object summary JSON; used match CSV fallback")

        metrics, keys = load_metrics_from_artifact(source)
        configuration = metrics.pop("configuration", "hydra") or "hydra"
        row = empty_object_row(scan_id, warnings)
        row["configuration"] = configuration
        row.update(metrics)
        row["source_file"] = str(source.path)
        validate_row_source(row, source)
        keys_used[str(source.path)] = keys or list(metrics.keys())

        missing = missing_metrics(row, required)
        if missing:
            message = f"missing object evaluator metrics: {', '.join(missing)}"
            warnings.append(message)
            if strict:
                raise RuntimeError(f"{scan_id}: {message} in {source.path}")
            warn(f"{scan_id}: {message} in {source.path}")
        row["missing_artifact_warnings"] = "; ".join(dict.fromkeys(warnings))
        if verbose:
            print(f"Selected object source for {scan_id}/{configuration}: {source.path}")
        rows.append(row)
    return rows, keys_used


def collect_region_rows(
    grouped: Dict[str, Dict[str, Dict[str, EvalArtifact]]],
    strict: bool,
    verbose: bool,
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

    for scan_id in sorted(grouped):
        warnings: List[str] = []
        source = select_source(grouped[scan_id], "region")
        if source is None:
            message = "missing region evaluator artifact"
            if strict:
                raise FileNotFoundError(f"{scan_id}: {message}")
            warnings.append(message)
            row = empty_region_row(scan_id, warnings)
            if verbose:
                print(f"No region source selected for {scan_id}")
            rows.append(row)
            continue

        metrics, keys = load_metrics_from_artifact(source)
        row = empty_region_row(scan_id, warnings)
        row.update(metrics)
        if not row.get("method"):
            row["method"] = "hydra"
        row["source_file"] = str(source.path)
        validate_row_source(row, source)
        keys_used[str(source.path)] = keys or list(metrics.keys())

        missing = missing_metrics(row, required)
        if missing:
            message = f"missing region evaluator metrics: {', '.join(missing)}"
            warnings.append(message)
            if strict:
                raise RuntimeError(f"{scan_id}: {message} in {source.path}")
            warn(f"{scan_id}: {message} in {source.path}")
        row["missing_artifact_warnings"] = "; ".join(dict.fromkeys(warnings))
        if verbose:
            print(f"Selected region source for {scan_id}: {source.path}")
        rows.append(row)
    return rows, keys_used


def validate_row_source(row: Dict[str, Any], artifact: EvalArtifact) -> None:
    if row.get("scan_id") != artifact.scan_id:
        raise RuntimeError(
            f"{artifact.kind} row/source scan mismatch\n"
            f"row scan_id: {row.get('scan_id')}\n"
            f"filename scan_id: {artifact.scan_id}\n"
            f"source: {artifact.path}"
        )


def assert_unique_rows(rows: Sequence[Dict[str, Any]], key_fields: Sequence[str], table_name: str) -> None:
    seen: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key in seen:
            raise RuntimeError(
                f"{table_name}: duplicate row key {key}\n"
                f"first source: {seen[key].get('source_file')}\n"
                f"second source: {row.get('source_file')}"
            )
        seen[key] = row


def assert_sources_not_reused_across_scan_ids(rows: Sequence[Dict[str, Any]], table_name: str) -> None:
    sources: Dict[str, str] = {}
    for row in rows:
        source = row.get("source_file")
        if not source:
            continue
        scan_id = row.get("scan_id")
        existing = sources.get(source)
        if existing is not None and existing != scan_id:
            raise RuntimeError(
                f"{table_name}: source file reused across scan IDs\n"
                f"source: {source}\n"
                f"first scan_id: {existing}\n"
                f"second scan_id: {scan_id}"
            )
        sources[source] = scan_id


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
    methods = sorted({str(row.get("method") or "hydra") for row in rows})
    for method in methods:
        method_rows = [row for row in rows if str(row.get("method") or "hydra") == method]
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
        selected_keys = [
            key
            for key in keys
            if re.search(r"(percent|error|num_|iou|boundary|adjacency|segmentation|matched)", key)
        ]
        selected_keys = selected_keys[:12] or keys[:12]
        print(f"  {source}")
        print(f"    keys: {', '.join(selected_keys)}")


def artifact_counts(artifacts: Sequence[EvalArtifact]) -> Dict[Tuple[str, str], int]:
    counts: Dict[Tuple[str, str], int] = {}
    for artifact in artifacts:
        key = (artifact.kind, artifact.artifact_type)
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_verbose_report(
    results_dir: Path,
    artifacts: Sequence[EvalArtifact],
    object_rows: Sequence[Dict[str, Any]],
    region_rows: Sequence[Dict[str, Any]],
) -> None:
    counts = artifact_counts(artifacts)
    print("\nVerbose report:")
    print(f"Results directory: {results_dir}")
    print(f"Discovered object summary JSONs: {counts.get(('object', 'summary'), 0)}")
    print(f"Discovered object match CSVs: {counts.get(('object', 'matches'), 0)}")
    print(f"Discovered region summary JSONs: {counts.get(('region', 'summary'), 0)}")
    print(f"Discovered region match CSVs: {counts.get(('region', 'matches'), 0)}")
    print(f"Object rows written: {len(object_rows)}")
    print(f"Region rows written: {len(region_rows)}")
    print("Duplicate object rows: 0")
    print("Duplicate region rows: 0")
    print("Cross-scan source mismatches: 0")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate existing Hydra MP3D evaluator outputs into tables."
    )
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist or is not a directory: {args.results_dir}")

    artifacts = discover_eval_artifacts(args.results_dir)
    if not artifacts:
        raise RuntimeError(f"No evaluator artifacts discovered under {args.results_dir}")

    grouped = group_artifacts(artifacts)
    object_rows, object_keys = collect_object_rows(grouped, args.strict, args.verbose)
    region_rows, region_keys = collect_region_rows(grouped, args.strict, args.verbose)

    assert_unique_rows(object_rows, ["scan_id", "configuration"], "object table")
    assert_unique_rows(region_rows, ["scan_id"], "region table")
    assert_sources_not_reused_across_scan_ids(object_rows, "object table")
    assert_sources_not_reused_across_scan_ids(region_rows, "region table")

    if args.verbose:
        print_verbose_report(args.results_dir, artifacts, object_rows, region_rows)
        print_key_note("Detected object source files and metric keys", object_keys)
        print_key_note("Detected region source files and metric keys", region_keys)

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

    validate_outputs([object_csv, region_csv, region_summary_csv])
    preview_csv(object_csv)
    preview_csv(region_csv)
    preview_csv(region_summary_csv)

    print("\nSummary:")
    print(f"  scans discovered: {len(grouped)}")
    print(f"  object result rows written: {len(object_rows)}")
    print(f"  region result rows written: {len(region_rows)}")
    print(f"  wrote: {object_csv}")
    print(f"  wrote: {region_csv}")
    print(f"  wrote: {region_summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
