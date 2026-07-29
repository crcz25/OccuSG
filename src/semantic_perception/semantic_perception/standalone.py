"""Standalone RGB-D-pose inference, intentionally independent of ROS."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from semantic_perception.inference.crop_embeddings import CropEmbedding, encode_object_crops
from semantic_perception.inference.embedding_cache import build_label_embedding_cache
from semantic_perception.inference.geometry import Geometry3D, compute_geometry
from semantic_perception.inference.models import (
    Detection,
    ModelBundle,
    non_maximum_suppression,
)
from semantic_perception.inference.vocabulary import (
    build_detector_prompt,
    parse_excluded_labels,
    resolve_global_index,
)


@dataclass
class StandaloneProposal:
    detection: Detection
    mask: np.ndarray | None
    embeddings: CropEmbedding
    camera_geometry: Geometry3D
    world_geometry: Geometry3D
    class_name: str
    class_score: float | None


class StandaloneInferencePipeline:
    """Synchronous GroundingDINO -> MobileSAM -> OpenCLIP RGB-D pipeline."""

    def __init__(
        self,
        config: dict,
        warning: Callable[[str], None] = print,
        info: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.models = ModelBundle(config, str(config["device"]), warning, info)
        self.labels = build_label_embedding_cache(
            config["class_labels_path"],
            config["class_embedding_cache_path"],
            self.models.clip.cache_key,
            self.models.clip.encode_texts,
            info,
        )
        info(self.labels.summary())
        self.prompt = build_detector_prompt(
            self.labels.vocabulary,
            int(config["detector_vocabulary_size"]),
            parse_excluded_labels(config.get("excluded_class_labels")),
        )
        self.warning = warning

    def infer(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Sequence[float],
        camera_to_world: np.ndarray,
    ) -> list[StandaloneProposal]:
        _validate_sample(rgb, depth_m, intrinsics, camera_to_world)
        detections = self._detect(rgb)
        if not detections:
            return []
        boxes = [item.box_xyxy for item in detections]
        masks = self.models.segmenter.segment(rgb, boxes)
        if len(masks) != len(boxes):
            raise RuntimeError("MobileSAM returned a different number of masks than boxes")
        embeddings = encode_object_crops(
            rgb,
            boxes,
            masks,
            [self.labels.by_index(item.class_index) for item in detections],
            self.models.clip.encode_images,
            self.warning,
        )
        output = []
        for detection, mask, embedding in zip(detections, masks, embeddings):
            camera_geometry = compute_geometry(
                depth_m,
                mask,
                detection.box_xyxy,
                intrinsics,
                int(self.config["min_valid_depth_points"]),
                float(self.config["max_depth_m"]),
            )
            world_geometry = compute_geometry(
                depth_m,
                mask,
                detection.box_xyxy,
                intrinsics,
                int(self.config["min_valid_depth_points"]),
                float(self.config["max_depth_m"]),
                camera_to_world,
            )
            class_name = detection.class_name
            class_score = (
                float(np.dot(embedding.mask_embedding, label_vector))
                if embedding.mask_embedding.size
                and (label_vector := self.labels.by_index(detection.class_index))
                is not None
                else None
            )
            output.append(
                StandaloneProposal(
                    detection,
                    mask,
                    embedding,
                    camera_geometry,
                    world_geometry,
                    class_name,
                    class_score,
                )
            )
        return output

    def _detect(self, rgb: np.ndarray) -> list:
        """Detect the whole vocabulary in one pass and label each result."""
        candidates: list = []
        for box, confidence, local_index, phrase in (
            self.models.detector.detect_with_classes(
                rgb,
                self.prompt.labels,
                self.prompt.caption,
                float(self.config["detection_threshold"]),
                float(self.config["text_threshold"]),
            )
        ):
            global_index = resolve_global_index(
                self.prompt, self.labels.vocabulary, local_index, phrase
            )
            class_name = (
                self.labels.vocabulary.label_at(global_index)
                if global_index is not None
                else None
            )
            if not class_name:
                continue
            candidates.append(
                Detection(
                    box_xyxy=box,
                    confidence=confidence,
                    class_index=int(global_index),
                    class_name=class_name,
                    phrase=phrase,
                )
            )
        if len(candidates) < 2:
            return candidates
        kept = non_maximum_suppression(
            np.asarray([item.box_xyxy for item in candidates], dtype=np.float64),
            np.asarray([item.confidence for item in candidates], dtype=np.float64),
            float(self.config["detector_merge_iou_threshold"]),
        )
        return [candidates[index] for index in kept]


def load_sample(
    rgb_path: str | os.PathLike[str],
    depth_path: str | os.PathLike[str],
    pose_path: str | os.PathLike[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    raw_depth = np.asarray(Image.open(depth_path))
    if raw_depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got {raw_depth.shape}")
    depth_m = raw_depth.astype(np.float32)
    if np.issubdtype(raw_depth.dtype, np.integer):
        depth_m *= 0.001
    pose_values = np.loadtxt(pose_path, dtype=np.float64).reshape(-1)
    if pose_values.size != 16:
        raise ValueError(f"Pose must contain 16 values, got {pose_values.size}")
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth_m), pose_values.reshape(4, 4)


def intrinsics_from_horizontal_fov(
    width: int, height: int, horizontal_fov_degrees: float
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0 or not 0.0 < horizontal_fov_degrees < 180.0:
        raise ValueError("Image dimensions and horizontal field of view are invalid")
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    return focal, focal, (width - 1.0) / 2.0, (height - 1.0) / 2.0


def optical_camera_to_world(pose: np.ndarray, convention: str) -> np.ndarray:
    """Convert a dataset pose into the optical frame used by depth projection."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Pose must be a finite 4x4 matrix")
    if convention == "optical":
        return value
    if convention == "habitat":
        # Habitat/OpenGL cameras use +x right, +y up, and look along -z;
        # pinhole RGB-D optical coordinates use +x right, +y down, +z forward.
        return value @ np.diag([1.0, -1.0, -1.0, 1.0])
    raise ValueError(f"Unknown pose convention: {convention}")


def _validate_sample(rgb, depth_m, intrinsics, camera_to_world) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB input must be an HxWx3 uint8 array")
    if depth_m.ndim != 2 or depth_m.shape != rgb.shape[:2]:
        raise ValueError("Aligned depth must have the same height and width as RGB")
    values = np.asarray(intrinsics, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[:2] <= 0):
        raise ValueError("Intrinsics must be finite (fx, fy, cx, cy) with positive focal lengths")
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")


def _default_paths() -> tuple[Path, Path]:
    package_root = Path(__file__).resolve().parents[1]
    workspace = next(
        (parent for parent in package_root.parents if (parent / "models").is_dir()),
        Path.cwd(),
    )
    return package_root, workspace


def _arguments() -> argparse.Namespace:
    package_root, workspace = _default_paths()
    sample = package_root / "test" / "Dd4bFSTQ8gi_000018"
    models = workspace / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, default=sample.with_name(sample.name + "_rgb.png"))
    parser.add_argument("--depth", type=Path, default=sample.with_name(sample.name + "_depth.png"))
    parser.add_argument("--pose", type=Path, default=sample.with_suffix(".txt"))
    parser.add_argument("--models-dir", type=Path, default=models)
    parser.add_argument(
        "--class-labels", type=Path, default=package_root / "prompts/example_classes.csv"
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache/semantic_perception/example_embeddings.bin",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--openclip-device",
        default="cpu",
        help="Keep ViT-H on CPU by default so the complete pipeline fits on 4 GB GPUs",
    )
    parser.add_argument("--horizontal-fov", type=float, default=90.0)
    parser.add_argument(
        "--pose-convention",
        choices=("habitat", "optical"),
        default="habitat",
        help="The supplied HM3D fixture uses Habitat/OpenGL camera axes",
    )
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--vocabulary-size",
        type=int,
        default=0,
        help="Detector labels to prompt with; 0 uses the whole vocabulary",
    )
    parser.add_argument(
        "--min-proposals",
        type=int,
        default=1,
        help="Fail the smoke run unless at least this many proposals are produced",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.min_proposals < 0:
        raise ValueError("--min-proposals must be non-negative")
    rgb, depth, pose = load_sample(args.rgb, args.depth, args.pose)
    camera_to_world = optical_camera_to_world(pose, args.pose_convention)
    intrinsics = intrinsics_from_horizontal_fov(rgb.shape[1], rgb.shape[0], args.horizontal_fov)
    config = {
        "device": args.device,
        "openclip_device": args.openclip_device,
        "groundingdino_device": args.device,
        "sam_device": args.device,
        "openclip_model": "ViT-H-14",
        "openclip_checkpoint_path": str(args.models_dir / "laion2b_s32b_b79k.bin"),
        "groundingdino_config_path": str(args.models_dir / "GroundingDINO_SwinT_OGC.py"),
        "groundingdino_model": str(args.models_dir / "groundingdino_swint_ogc.pth"),
        "sam_model": str(args.models_dir / "mobile_sam.pt"),
        "sam_model_type": "vit_t",
        "class_labels_path": str(args.class_labels),
        "class_embedding_cache_path": str(args.cache),
        "detection_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "detector_vocabulary_size": args.vocabulary_size,
        "excluded_class_labels": args.exclude_classes,
        "detector_merge_iou_threshold": 0.7,
        "min_valid_depth_points": 20,
        "max_depth_m": 10.0,
    }
    pipeline = StandaloneInferencePipeline(config)
    proposals = pipeline.infer(rgb, depth, intrinsics, camera_to_world)
    if len(proposals) < args.min_proposals:
        raise RuntimeError(
            f"Expected at least {args.min_proposals} proposals, got {len(proposals)}"
        )
    if args.min_proposals > 0 and not any(
        item.mask is not None
        and item.embeddings.fused_embedding.size > 0
        and item.world_geometry.valid
        for item in proposals
    ):
        raise RuntimeError(
            "No proposal contained a MobileSAM mask, OpenCLIP embedding, and valid 3D geometry"
        )
    report = {
        "sample": {"rgb": str(args.rgb), "depth": str(args.depth), "pose": str(args.pose)},
        "image_shape": list(rgb.shape),
        "depth_range_m": [float(depth[depth > 0].min()), float(depth.max())],
        "intrinsics": list(intrinsics),
        "pose_convention": args.pose_convention,
        "proposal_count": len(proposals),
        "proposals": [
            {
                "box_xyxy": item.detection.box_xyxy.tolist(),
                "detection_confidence": item.detection.confidence,
                "class_name": item.class_name,
                "class_score": item.class_score,
                "mask_pixels": int(item.mask.sum()) if item.mask is not None else 0,
                "embedding_dimension": int(item.embeddings.fused_embedding.size),
                "valid_3d": item.world_geometry.valid,
                "centroid_camera": item.camera_geometry.centroid.tolist(),
                "centroid_world": item.world_geometry.centroid.tolist(),
            }
            for item in proposals
        ],
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
