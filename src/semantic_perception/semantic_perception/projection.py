"""Camera-to-graph frame resolution and projection diagnostics.

Object geometry is projected into the target frame inside this package, using the
transform valid **at the frame's own timestamp**. Doing it here — rather than
applying the newest available transform later in the graph node — is what keeps a
proposal's position tied to the pose the camera actually had when the image was
captured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray | None:
    """Return the 3x3 rotation for a quaternion, or ``None`` when degenerate."""
    values = np.asarray([x, y, z, w], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    qx, qy, qz, qw = values / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def transform_to_matrix(transform_stamped) -> np.ndarray | None:
    """Convert ``geometry_msgs/TransformStamped`` into a 4x4 homogeneous matrix.

    The result maps a point expressed in the transform's *child* frame into its
    *header* frame: ``point_target = matrix @ [x, y, z, 1]``. This is the same
    direction ``tf2`` uses for ``lookup_transform(target, source, ...)``.
    """
    if transform_stamped is None:
        return None
    transform = getattr(transform_stamped, "transform", None)
    if transform is None:
        return None
    translation = getattr(transform, "translation", None)
    rotation = getattr(transform, "rotation", None)
    if translation is None or rotation is None:
        return None
    rotation_matrix = quaternion_to_rotation_matrix(
        float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
    )
    if rotation_matrix is None:
        return None
    offset = np.asarray(
        [float(translation.x), float(translation.y), float(translation.z)],
        dtype=np.float64,
    )
    if not np.isfinite(offset).all():
        return None
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = offset
    return matrix


@dataclass
class ProjectionDiagnostics:
    """One frame's projection trace, rendered only when diagnostics are enabled."""

    sequence: int
    source_frame: str
    target_frame: str
    frame_stamp_sec: float
    transform_stamp_sec: float
    transform: np.ndarray | None = None
    entries: list[str] = field(default_factory=list)

    def add_detection(
        self,
        detection_id: int,
        class_name: str,
        camera_position: Sequence[float] | None,
        graph_position: Sequence[float] | None,
        mask_pixels: int,
        depth_statistics: tuple[float, float, float, int] | None,
    ) -> None:
        def render(point: Sequence[float] | None) -> str:
            if point is None:
                return "n/a"
            return "(" + ", ".join(f"{float(value):.3f}" for value in point) + ")"

        if depth_statistics is None:
            depth_text = "depth n/a"
        else:
            median, minimum, maximum, rejected = depth_statistics
            depth_text = (
                f"depth median={median:.3f} min={minimum:.3f} max={maximum:.3f} "
                f"rejected={rejected}"
            )
        self.entries.append(
            f"  id={detection_id} class={class_name!r} "
            f"camera={render(camera_position)} graph={render(graph_position)} "
            f"mask_px={mask_pixels} {depth_text}"
        )

    def render(self) -> str:
        latency = self.frame_stamp_sec - self.transform_stamp_sec
        header = (
            f"[projection] seq={self.sequence} {self.source_frame}->{self.target_frame} "
            f"frame_stamp={self.frame_stamp_sec:.6f} "
            f"tf_stamp={self.transform_stamp_sec:.6f} "
            f"tf_age={latency:+.3f}s"
        )
        if self.transform is not None:
            translation = self.transform[:3, 3]
            header += (
                "  translation=("
                + ", ".join(f"{float(value):.3f}" for value in translation)
                + ")"
            )
        return "\n".join([header, *self.entries])


def depth_statistics(
    depth_m: np.ndarray,
    mask: np.ndarray | None,
    box_xyxy: Sequence[float],
    max_depth_m: float,
) -> tuple[float, float, float, int] | None:
    """Return ``(median, minimum, maximum, rejected_count)`` over the selection."""
    if depth_m.ndim != 2:
        return None
    height, width = depth_m.shape
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return None
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    selection = np.zeros((height, width), dtype=bool)
    selection[y1:y2, x1:x2] = True
    if mask is not None and mask.shape == depth_m.shape:
        selection &= np.asarray(mask, dtype=bool)
    candidates = depth_m[selection]
    if candidates.size == 0:
        return None
    usable = candidates[
        np.isfinite(candidates) & (candidates > 0.0) & (candidates <= max_depth_m)
    ]
    rejected = int(candidates.size - usable.size)
    if usable.size == 0:
        return 0.0, 0.0, 0.0, rejected
    return (
        float(np.median(usable)),
        float(usable.min()),
        float(usable.max()),
        rejected,
    )
