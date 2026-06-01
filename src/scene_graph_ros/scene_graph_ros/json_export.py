"""Thin ROS-side helpers for triggering core JSON export."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Optional

import rclpy

from scene_graph_core.serialization import SceneGraphJsonSerializer


def get_bool_param(param_dict: dict, param_name: str, default: bool = False) -> bool:
    """Return a bool parameter, accepting native bools and string values."""
    value = param_dict.get(param_name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def resolve_export_json_path(node, param_dict: dict) -> Optional[Path]:
    """Return a concrete JSON file path for shutdown export."""
    raw_path = str(param_dict.get("export_json_path", "") or "").strip()
    if not raw_path:
        return None

    export_path = Path(raw_path).expanduser()
    if export_path.exists() and export_path.is_dir():
        resolved_path = export_path / "scene_graph.json"
        node.get_logger().warning(
            "export_json_path points to a directory; writing to "
            f"{resolved_path}"
        )
        return resolved_path

    if raw_path.endswith(("/", "\\")) or export_path.suffix == "":
        resolved_path = export_path / "scene_graph.json"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        node.get_logger().warning(
            "export_json_path does not include a file name; writing to "
            f"{resolved_path}"
        )
        return resolved_path

    export_path.parent.mkdir(parents=True, exist_ok=True)
    return export_path


def prepare_scene_graph_for_shutdown(orchestrator) -> None:
    """Best-effort final drain before persisting the graph."""
    if getattr(orchestrator, "_shutdown_prepare_completed", False):
        return
    orchestrator._shutdown_prepare_completed = True

    logger = orchestrator.get_logger()
    logger.info("Preparing scene graph for shutdown export")

    for _ in range(5):
        pose_manager = getattr(orchestrator, "pose_manager", None)
        detection_queue = getattr(orchestrator, "detection_queue", None)
        pending_poses = (
            pose_manager.pending_count() if pose_manager is not None else 0
        )
        pending_detections = (
            detection_queue.pending_count() if detection_queue is not None else 0
        )
        if pending_poses <= 0 and pending_detections <= 0:
            break

        for method_name in ("_pose_flush_tick", "_detection_flush_tick"):
            method = getattr(orchestrator, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception as exc:
                logger.warning(f"Shutdown {method_name} failed: {exc}")

    for method_name in ("_maintenance_tick", "_pipeline_tick", "_maintenance_tick"):
        method = getattr(orchestrator, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception as exc:
            logger.warning(f"Shutdown {method_name} failed: {exc}")


def export_scene_graph_json_if_configured(orchestrator) -> None:
    """Export one orchestrator's persisted core graph when configured."""
    if getattr(orchestrator, "_json_export_completed", False):
        return

    param_dict = getattr(orchestrator, "_param_dict", {})
    if not get_bool_param(param_dict, "export_json_on_shutdown", True):
        return

    export_path = resolve_export_json_path(orchestrator, param_dict)
    if export_path is None:
        orchestrator.get_logger().warning(
            "export_json_on_shutdown is enabled but export_json_path is empty; "
            "skipping JSON export"
        )
        return

    prepare_scene_graph_for_shutdown(orchestrator)

    compact = get_bool_param(param_dict, "export_json_compact", False)

    try:
        with orchestrator._sg_lock:
            path = SceneGraphJsonSerializer().export_json(
                orchestrator.sg,
                export_path,
                compact=compact,
            )
        orchestrator._json_export_completed = True
        orchestrator.get_logger().info(f"Exported scene graph JSON to {path}")
    except Exception as exc:
        orchestrator.get_logger().error(f"Failed to export scene graph JSON: {exc}")


def register_shutdown_export(orchestrator) -> None:
    """Register JSON export with rclpy shutdown when supported."""
    context = rclpy.get_default_context()
    on_shutdown = getattr(context, "on_shutdown", None)
    if callable(on_shutdown):
        on_shutdown(lambda: export_scene_graph_json_if_configured(orchestrator))


def register_signal_export(orchestrator) -> None:
    """Export before process-level launch shutdown signals can terminate Python."""
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(signum)

        def _handler(received_signum, frame, previous_handler=previous_handler):
            export_scene_graph_json_if_configured(orchestrator)
            if callable(previous_handler):
                previous_handler(received_signum, frame)
                return
            if previous_handler == signal.SIG_IGN:
                return
            if received_signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + int(received_signum))

        signal.signal(signum, _handler)


def register_export_triggers(orchestrator) -> None:
    """Register all shutdown/export hooks for one orchestrator."""
    register_shutdown_export(orchestrator)
    register_signal_export(orchestrator)
