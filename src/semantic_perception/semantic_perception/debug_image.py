"""Rendering for the optional ROS inference debug image."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


_COLORS_BGR = (
    (61, 214, 255),
    (255, 144, 30),
    (87, 255, 126),
    (255, 93, 177),
    (210, 180, 70),
    (120, 120, 255),
)


def render_debug_image(
    rgb: np.ndarray,
    proposals: Sequence,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Return a BGR image with masks, labels, and model confidence scores."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Debug RGB input must have shape HxWx3 and dtype uint8")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("Debug mask alpha must be between 0 and 1")

    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = image.copy()
    height, width = image.shape[:2]

    for index, proposal in enumerate(proposals):
        mask = proposal.mask
        if mask is None:
            continue
        mask_value = np.asarray(mask, dtype=bool)
        if mask_value.shape == (height, width):
            overlay[mask_value] = _COLORS_BGR[index % len(_COLORS_BGR)]
    if proposals and mask_alpha > 0.0:
        image = cv2.addWeighted(overlay, mask_alpha, image, 1.0 - mask_alpha, 0.0)

    for index, proposal in enumerate(proposals):
        color = _COLORS_BGR[index % len(_COLORS_BGR)]
        box = np.asarray(proposal.detection.box_xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1 = int(np.clip(round(box[0]), 0, max(0, width - 1)))
        y1 = int(np.clip(round(box[1]), 0, max(0, height - 1)))
        x2 = int(np.clip(round(box[2]), 0, max(0, width - 1)))
        y2 = int(np.clip(round(box[3]), 0, max(0, height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        class_name = str(proposal.detection.class_name or "object")
        caption = (
            f"{class_name} | Det {_format_score(proposal.detection.confidence)} "
            f"| SAM {_format_score(proposal.sam_score)}"
        )
        _draw_caption(image, caption, x1, y1, color)
    return image


def _format_score(value: object) -> str:
    """Render an optional finite model score for a compact debug caption."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{score:.2f}" if np.isfinite(score) else "n/a"


def _draw_caption(
    image: np.ndarray, caption: str, x: int, y: int, color: tuple[int, int, int]
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        caption, font, scale, thickness
    )
    height, width = image.shape[:2]
    left = min(max(0, x), max(0, width - text_width - 6))
    top = max(0, y - text_height - baseline - 6)
    right = min(width - 1, left + text_width + 6)
    bottom = min(height - 1, top + text_height + baseline + 6)
    cv2.rectangle(image, (left, top), (right, bottom), color, -1)
    cv2.putText(
        image,
        caption,
        (left + 3, top + text_height + 2),
        font,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )
