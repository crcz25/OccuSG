"""Depth projection and axis-aligned 3D extent calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Geometry3D:
    valid: bool
    centroid: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def invalid(cls) -> "Geometry3D":
        zero = np.zeros(3, dtype=np.float32)
        return cls(False, zero, zero.copy(), zero.copy())


def compute_geometry(
    depth_m: np.ndarray,
    mask: np.ndarray | None,
    box_xyxy: Sequence[float],
    intrinsics: Sequence[float],
    min_valid_points: int = 20,
    max_depth_m: float = 10.0,
) -> Geometry3D:
    """Project valid masked depths with pinhole intrinsics in the camera frame."""
    if depth_m.ndim != 2 or min_valid_points <= 0 or max_depth_m <= 0.0:
        return Geometry3D.invalid()
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return Geometry3D.invalid()
    fx, fy, cx, cy = values
    if fx <= 0.0 or fy <= 0.0:
        return Geometry3D.invalid()

    height, width = depth_m.shape
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        return Geometry3D.invalid()
    x1, y1 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x2, y2 = min(width, int(np.ceil(box[2]))), min(height, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return Geometry3D.invalid()

    selection = np.zeros((height, width), dtype=bool)
    selection[y1:y2, x1:x2] = True
    if mask is not None:
        mask_value = np.asarray(mask, dtype=bool)
        if mask_value.shape != depth_m.shape:
            return Geometry3D.invalid()
        selection &= mask_value
    valid = selection & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    rows, columns = np.nonzero(valid)
    if rows.size < min_valid_points:
        return Geometry3D.invalid()
    z = depth_m[rows, columns].astype(np.float64)
    x = (columns.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    if not np.isfinite(points).all():
        return Geometry3D.invalid()
    return Geometry3D(
        True,
        points.mean(axis=0).astype(np.float32),
        points.min(axis=0).astype(np.float32),
        points.max(axis=0).astype(np.float32),
    )
