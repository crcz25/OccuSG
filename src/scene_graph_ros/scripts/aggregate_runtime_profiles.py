#!/usr/bin/env python3
"""Aggregate runtime profiling JSON files into JSON and CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional


SUMMARY_ROWS = [
    ("Point-cloud generation", "point_cloud_generation_ms"),
    ("3D occupancy integration", "occupancy_integration_ms"),
    ("2D free-space projection", "free_space_projection_ms"),
    ("DUDE decomposition", "dude_decomposition_ms"),
    ("Region tracking", "region_tracking_ms"),
    ("Entity assignment & graph assembly", "entity_assignment_graph_assembly_ms"),
]

SUMMARY_JSON = "runtime_summary.json"
SUMMARY_CSV = "runtime_summary.csv"


def _sample_value(sample: Any) -> Optional[float]:
    if isinstance(sample, (int, float)):
        return float(sample)
    if isinstance(sample, dict) and "elapsed_ms" in sample:
        return float(sample["elapsed_ms"])
    return None


def _summary_from_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean_ms": mean,
        "std_ms": math.sqrt(max(0.0, variance)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _format_ms(value: Any, empty: str = "--") -> str:
    if value is None:
        return empty
    return f"{float(value):.2f}"


def _zero_sample_note(values: list[float], stage_file_count: int) -> str:
    if values:
        return ""
    if stage_file_count <= 0:
        return "no profiling file contained this stage"
    return "stage present, but no samples after warmup discard"


def _json_files_for_single_run(profiling_dir: Path, run_name: str) -> list[Path]:
    return sorted(
        path
        for path in profiling_dir.glob(f"{run_name}.*.json")
        if path.name not in {SUMMARY_JSON}
    )


def _json_files_for_dataset(root_dir: Path, run_name: Optional[str]) -> list[Path]:
    if root_dir.name == "profiling":
        pattern = f"{run_name}.*.json" if run_name else "*.json"
        return sorted(
            path
            for path in root_dir.glob(pattern)
            if not path.name.startswith("runtime_summary")
        )

    files: list[Path] = []
    for profiling_dir in sorted(root_dir.glob("*/profiling")):
        scan_id = profiling_dir.parent.name
        pattern = f"{run_name}.*.json" if run_name else f"{scan_id}_region.*.json"
        matched = sorted(profiling_dir.glob(pattern))
        if not matched and run_name is None:
            matched = sorted(profiling_dir.glob("*.json"))
        files.extend(path for path in matched if not path.name.startswith("runtime_summary"))
    return files


def _load_profiles(paths: list[Path], root_dir: Path) -> list[dict[str, Any]]:
    profiles = []
    for path in paths:
        try:
            profile = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse {path}: {exc}") from exc

        profiling_dir = path.parent
        scan_id = profiling_dir.parent.name if profiling_dir.name == "profiling" else None
        profile["_source_file"] = str(path.relative_to(root_dir))
        profile["_scan_id"] = scan_id
        profiles.append(profile)
    return profiles


def _stage_samples(profile: dict[str, Any], stage_name: str) -> list[float]:
    stage = profile.get("stages", {}).get(stage_name)
    if not isinstance(stage, dict):
        return []

    raw_samples = stage.get("samples_ms", [])
    values = [
        value
        for value in (_sample_value(sample) for sample in raw_samples)
        if value is not None and math.isfinite(value)
    ]
    discard = max(0, int(profile.get("discarded_warmup_count") or 0))
    return values[discard:]


def aggregate_dataset(root_dir: Path, run_name: Optional[str] = None) -> dict[str, Any]:
    root_dir = root_dir.resolve()
    files = _json_files_for_dataset(root_dir, run_name)
    profiles = _load_profiles(files, root_dir)
    scan_ids = sorted(
        {
            str(profile["_scan_id"])
            for profile in profiles
            if profile.get("_scan_id") is not None
        }
    )

    rows = []
    pooled_values_by_stage: dict[str, list[float]] = {}
    source_files_by_stage: dict[str, list[str]] = {}
    for label, stage_name in SUMMARY_ROWS:
        values: list[float] = []
        source_files: list[str] = []
        stage_file_count = 0
        for profile in profiles:
            if stage_name in profile.get("stages", {}):
                stage_file_count += 1
            samples = _stage_samples(profile, stage_name)
            if not samples:
                continue
            values.extend(samples)
            source_files.append(profile["_source_file"])

        summary = _summary_from_values(values)
        pooled_values_by_stage[stage_name] = values
        source_files_by_stage[stage_name] = source_files
        rows.append(
            {
                "row": label,
                "stage": stage_name,
                "n": summary["n"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "min_ms": summary["min_ms"],
                "max_ms": summary["max_ms"],
                "source_file_count": len(source_files),
                "note": _zero_sample_note(values, stage_file_count),
            }
        )

    measured_rows = [row for row in rows if row["mean_ms"] is not None]
    if len(measured_rows) == len(SUMMARY_ROWS):
        total_mean = sum(float(row["mean_ms"]) for row in measured_rows)
        total_std_sum = sum(float(row["std_ms"] or 0.0) for row in measured_rows)
        rows.append(
            {
                "row": "Total per update",
                "stage": "total_per_update_sum_of_measured_stages_ms",
                "n": None,
                "mean_ms": total_mean,
                "std_ms": total_std_sum,
                "min_ms": None,
                "max_ms": None,
                "source_file_count": None,
                "note": "mean and std columns are sums of measured stage statistics",
            }
        )

    machines = []
    for profile in profiles:
        metadata = profile.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        machine = {
            "hostname": metadata.get("hostname"),
            "cpu_model": metadata.get("cpu_model"),
            "logical_cores": metadata.get("logical_cores"),
            "ram_gb": metadata.get("ram_gb"),
            "ros_distro": metadata.get("ros_distro"),
        }
        if machine not in machines:
            machines.append(machine)

    return {
        "root_dir": str(root_dir),
        "run_name_filter": run_name,
        "scan_ids": scan_ids,
        "scan_count": len(scan_ids),
        "profile_file_count": len(profiles),
        "input_files": [profile["_source_file"] for profile in profiles],
        "rows": rows,
        "machine_metadata": machines,
        "notes": [
            "Stage means and standard deviations are pooled over processed update samples from all discovered scans.",
            "The first discarded_warmup_count samples are removed separately for each process file and stage before pooling.",
            "Stage timings are callback execution times measured with monotonic wall-clock timers, not end-to-end message latency.",
            "Total per update is reported as the sum of measured stage mean/std columns because per-update cross-process correlation is not available.",
        ],
        "source_files_by_stage": source_files_by_stage,
    }


def aggregate(profiling_dir: Path, run_name: str) -> dict[str, Any]:
    """Backward-compatible single-run entry point."""
    profiling_dir = profiling_dir.resolve()
    files = _json_files_for_single_run(profiling_dir, run_name)
    profiles = _load_profiles(files, profiling_dir)

    rows = []
    for label, stage_name in SUMMARY_ROWS:
        values: list[float] = []
        source_files: list[str] = []
        stage_file_count = 0
        for profile in profiles:
            if stage_name in profile.get("stages", {}):
                stage_file_count += 1
            samples = _stage_samples(profile, stage_name)
            if not samples:
                continue
            values.extend(samples)
            source_files.append(profile["_source_file"])
        summary = _summary_from_values(values)
        rows.append(
            {
                "row": label,
                "stage": stage_name,
                "n": summary["n"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "min_ms": summary["min_ms"],
                "max_ms": summary["max_ms"],
                "source_file_count": len(source_files),
                "note": _zero_sample_note(values, stage_file_count),
            }
        )

    measured_rows = [row for row in rows if row["mean_ms"] is not None]
    if len(measured_rows) == len(SUMMARY_ROWS):
        rows.append(
            {
                "row": "Total per update",
                "stage": "total_per_update_sum_of_measured_stages_ms",
                "n": None,
                "mean_ms": sum(float(row["mean_ms"]) for row in measured_rows),
                "std_ms": sum(float(row["std_ms"] or 0.0) for row in measured_rows),
                "min_ms": None,
                "max_ms": None,
                "source_file_count": None,
                "note": "mean and std columns are sums of measured stage statistics",
            }
        )

    return {
        "root_dir": str(profiling_dir),
        "run_name_filter": run_name,
        "scan_ids": [profiling_dir.parent.name] if profiling_dir.name == "profiling" else [],
        "scan_count": 1 if profiling_dir.name == "profiling" else 0,
        "profile_file_count": len(profiles),
        "input_files": [profile["_source_file"] for profile in profiles],
        "rows": rows,
        "machine_metadata": [],
        "notes": [
            "Stage means and standard deviations are pooled over processed update samples.",
            "The first discarded_warmup_count samples are removed separately for each process file and stage before pooling.",
            "Stage timings are callback execution times measured with monotonic wall-clock timers, not end-to-end message latency.",
            "Total per update is reported as the sum of measured stage mean/std columns because per-update cross-process correlation is not available.",
        ],
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / SUMMARY_JSON
    csv_path = output_dir / SUMMARY_CSV

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    fieldnames = [
        "row",
        "stage",
        "n",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "source_file_count",
        "note",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row.get(key) for key in fieldnames})


def print_markdown(summary: dict[str, Any]) -> None:
    print("| Stage | n | mean ms | std ms | source files | Note |")
    print("|---|---:|---:|---:|---:|---|")
    for row in summary["rows"]:
        print(
            "| {row} | {n} | {mean} | {std} | {files} | {note} |".format(
                row=row["row"],
                n="" if row["n"] is None else row["n"],
                mean=_format_ms(row["mean_ms"], empty=""),
                std=_format_ms(row["std_ms"], empty=""),
                files="" if row.get("source_file_count") is None else row["source_file_count"],
                note=row.get("note") or "",
            )
        )
    print()
    for note in summary["notes"]:
        print(f"- {note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate one profiling directory or a dataset directory containing "
            "scan_id/profiling/*.json runtime files."
        )
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Profiling directory, or dataset/bag root containing scan_id/profiling dirs.",
    )
    parser.add_argument(
        "run_name",
        nargs="?",
        default=None,
        help="Optional run name filter. Preserve old usage with: profiling_dir run_name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for runtime_summary outputs. Defaults to root_dir.",
    )
    args = parser.parse_args()

    direct_files = (
        _json_files_for_single_run(args.root_dir, args.run_name)
        if args.run_name
        else []
    )
    if args.run_name and direct_files:
        summary = aggregate(args.root_dir, args.run_name)
    else:
        summary = aggregate_dataset(args.root_dir, args.run_name)

    output_dir = args.output_dir or args.root_dir
    write_outputs(summary, output_dir)
    print_markdown(summary)


if __name__ == "__main__":
    main()
