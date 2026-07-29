"""Crop creation, batched visual encoding, and three-part embedding fusion.

Each detection produces three unit-norm CLIP vectors of the same dimension ``D``:
the masked crop, the raw detector-box crop, and the cached text embedding of the
resolved class label. They are concatenated in that fixed order and re-normalized
into a ``3D`` representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class CropEmbedding:
    mask_embedding: np.ndarray
    bbox_embedding: np.ndarray
    label_embedding: np.ndarray
    fused_embedding: np.ndarray

    @classmethod
    def empty(cls) -> "CropEmbedding":
        empty = np.empty(0, dtype=np.float32)
        return cls(empty, empty.copy(), empty.copy(), empty.copy())

    @property
    def valid(self) -> bool:
        return self.fused_embedding.size > 0


def normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        return None
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return (value / norm).astype(np.float32)


def fuse_embeddings(
    mask_embedding: np.ndarray | None,
    bbox_embedding: np.ndarray | None,
    label_embedding: np.ndarray | None,
) -> np.ndarray:
    """Return the L2-normalized concatenation ``[mask | bbox | label]``.

    Any missing, non-finite, zero-norm, or dimensionally inconsistent component
    rejects the detection by returning an empty array.
    """
    components = [
        normalize(mask_embedding),
        normalize(bbox_embedding),
        normalize(label_embedding),
    ]
    if any(component is None for component in components):
        return np.empty(0, dtype=np.float32)
    dimensions = {component.shape[0] for component in components}
    if len(dimensions) != 1:
        return np.empty(0, dtype=np.float32)
    fused = normalize(np.concatenate(components))
    return fused if fused is not None else np.empty(0, dtype=np.float32)


def make_crops(
    rgb: np.ndarray, box_xyxy: Sequence[float], mask: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return a bbox crop and a black-background masked crop.

    Both crops use the same detector box, so the two CLIP views differ only by
    whether the background is preserved.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB frame must have shape HxWx3")
    height, width = rgb.shape[:2]
    values = np.asarray(box_xyxy, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None, None
    x1 = max(0, min(width, int(np.floor(values[0]))))
    y1 = max(0, min(height, int(np.floor(values[1]))))
    x2 = max(0, min(width, int(np.ceil(values[2]))))
    y2 = max(0, min(height, int(np.ceil(values[3]))))
    if x2 <= x1 or y2 <= y1:
        return None, None
    bbox_crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
    if mask is None or mask.shape != (height, width):
        return bbox_crop, None
    object_mask = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
    if not object_mask.any():
        return bbox_crop, None
    masked_crop = np.zeros_like(bbox_crop)
    masked_crop[object_mask] = bbox_crop[object_mask]
    return bbox_crop, masked_crop


def encode_object_crops(
    rgb: np.ndarray,
    boxes: Sequence[Sequence[float]],
    masks: Sequence[np.ndarray | None],
    label_embeddings: Sequence[np.ndarray | None],
    encode_batch: Callable[[Sequence[np.ndarray]], np.ndarray],
    warning: Callable[[str], None] | None = None,
) -> list[CropEmbedding]:
    """Encode all bbox crops in one batch and all valid mask crops in one batch."""
    if not (len(boxes) == len(masks) == len(label_embeddings)):
        raise ValueError(
            "Each detection must have exactly one mask and one label embedding"
        )
    crops = [make_crops(rgb, box, mask) for box, mask in zip(boxes, masks)]
    bbox_indices = [i for i, pair in enumerate(crops) if pair[0] is not None]
    mask_indices = [i for i, pair in enumerate(crops) if pair[1] is not None]

    bbox_vectors = _run_batch(crops, bbox_indices, 0, encode_batch, "bbox", warning)
    mask_vectors = _run_batch(crops, mask_indices, 1, encode_batch, "mask", warning)

    output: list[CropEmbedding] = []
    empty = np.empty(0, dtype=np.float32)
    for index in range(len(boxes)):
        mask_vector = mask_vectors.get(index)
        bbox_vector = bbox_vectors.get(index)
        label_vector = normalize(label_embeddings[index])
        fused = fuse_embeddings(mask_vector, bbox_vector, label_vector)
        if fused.size == 0 and warning:
            warning(
                f"Object {index} rejected: incomplete embedding components "
                f"(mask={mask_vector is not None}, bbox={bbox_vector is not None}, "
                f"label={label_vector is not None})"
            )
        output.append(
            CropEmbedding(
                mask_vector if mask_vector is not None else empty.copy(),
                bbox_vector if bbox_vector is not None else empty.copy(),
                label_vector if label_vector is not None else empty.copy(),
                fused,
            )
        )
    return output


def _run_batch(crops, indices, component, encode_batch, label, warning):
    if not indices:
        return {}
    encoded = np.asarray(
        encode_batch([crops[index][component] for index in indices]), dtype=np.float32
    )
    if encoded.ndim != 2 or encoded.shape[0] != len(indices):
        if warning:
            warning(f"CLIP {label} batch returned an invalid shape: {encoded.shape}")
        return {}
    result = {}
    for index, vector in zip(indices, encoded):
        normalized = normalize(vector)
        if normalized is not None:
            result[index] = normalized
        elif warning:
            warning(f"Object {index} has an invalid {label} embedding")
    return result
