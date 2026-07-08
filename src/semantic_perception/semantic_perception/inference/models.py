"""Thin adapters around GroundingDINO, SAM/MobileSAM, and OpenCLIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image


@dataclass
class Detection:
    box_xyxy: np.ndarray
    confidence: float
    source: str = "groundingdino"


def select_device(requested: str, warning: Callable[[str], None]) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for inference; install a build appropriate for this system"
        ) from exc
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warning(f"CUDA device '{requested}' is unavailable; falling back to CPU")
        return "cpu"
    if requested.startswith("cuda"):
        try:
            index = int(requested.split(":", 1)[1]) if ":" in requested else 0
            if index >= torch.cuda.device_count():
                warning(f"CUDA device '{requested}' does not exist; falling back to CPU")
                return "cpu"
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested}") from exc
    return requested


def _require_file(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"Parameter '{label}' must name a model file")
    value = Path(path).expanduser()
    if not value.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {value}")
    return str(value)


class OpenClipAdapter:
    def __init__(self, model_name: str, checkpoint: str, device: str):
        if not model_name:
            raise ValueError("Parameter 'openclip_model' must not be empty")
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Install open_clip_torch and PyTorch to use OpenCLIP") from exc

        pretrained = checkpoint or None
        if checkpoint and ("/" in checkpoint or checkpoint.startswith(".")):
            pretrained = _require_file(checkpoint, "openclip_checkpoint_path")
        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load OpenCLIP model '{model_name}': {exc}") from exc
        self._model.eval()
        self._device = device
        self._torch = torch

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        tensors = [self._preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8))) for image in images]
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.inference_mode():
            values = self._model.encode_image(batch)
        return values.float().cpu().numpy().astype(np.float32)

    def encode_texts(self, prompts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(prompts)).to(self._device)
        with self._torch.inference_mode():
            values = self._model.encode_text(tokens)
        return values.float().cpu().numpy().astype(np.float32)


class GroundingDinoAdapter:
    def __init__(self, config_path: str, weights_path: str, device: str):
        config = _require_file(config_path, "groundingdino_config_path")
        weights = _require_file(weights_path, "groundingdino_model")
        try:
            from groundingdino.util.inference import Model
        except ImportError as exc:
            raise RuntimeError(
                "GroundingDINO is not importable; install the official GroundingDINO package"
            ) from exc
        try:
            self._model = Model(
                model_config_path=config,
                model_checkpoint_path=weights,
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load GroundingDINO: {exc}") from exc

    def detect(
        self, rgb: np.ndarray, prompt: str, box_threshold: float, text_threshold: float
    ) -> list[Detection]:
        try:
            detections, _ = self._model.predict_with_caption(
                image=rgb,
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"GroundingDINO inference failed: {exc}") from exc
        boxes = np.asarray(getattr(detections, "xyxy", []), dtype=np.float32).reshape(-1, 4)
        confidences = np.asarray(
            getattr(detections, "confidence", np.ones(len(boxes))), dtype=np.float32
        ).reshape(-1)
        output = []
        for box, confidence in zip(boxes, confidences):
            if np.isfinite(box).all() and np.isfinite(confidence):
                output.append(Detection(box, float(confidence)))
        return output


class SamAdapter:
    def __init__(self, weights_path: str, model_type: str, device: str):
        weights = _require_file(weights_path, "sam_model")
        registries = []
        errors = []
        for module_name in ("mobile_sam", "segment_anything"):
            try:
                module = __import__(module_name, fromlist=["SamPredictor", "sam_model_registry"])
                registries.append((module.SamPredictor, module.sam_model_registry))
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}: {exc}")
        if not registries:
            raise RuntimeError("Neither MobileSAM nor Segment Anything is importable: " + "; ".join(errors))
        last_error: Exception | None = None
        for predictor_type, registry in registries:
            if model_type not in registry:
                continue
            try:
                model = registry[model_type](checkpoint=weights)
                model.to(device=device)
                model.eval()
                self._predictor = predictor_type(model)
                return
            except Exception as exc:
                last_error = exc
        available = sorted({key for _, registry in registries for key in registry.keys()})
        if model_type not in available:
            raise ValueError(f"Unknown SAM model type '{model_type}'; available: {available}")
        raise RuntimeError(f"Failed to load SAM model: {last_error}")

    def segment(self, rgb: np.ndarray, boxes: Sequence[np.ndarray]) -> list[np.ndarray | None]:
        if not boxes:
            return []
        self._predictor.set_image(rgb)
        output: list[np.ndarray | None] = []
        for box in boxes:
            try:
                masks, scores, _ = self._predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                )
                if len(masks) == 0:
                    output.append(None)
                else:
                    output.append(np.asarray(masks[int(np.argmax(scores))], dtype=bool))
            except Exception:
                output.append(None)
        return output


class ModelBundle:
    """Models owned by one worker/device and loaded once before its thread starts."""

    def __init__(
        self,
        config: dict,
        device: str,
        warning: Callable[[str], None],
        info: Callable[[str], None],
    ):
        self.device = select_device(device, warning)
        info(f"Loading inference models on {self.device}")
        self.clip = OpenClipAdapter(
            config["openclip_model"], config["openclip_checkpoint_path"], self.device
        )
        self.detector = GroundingDinoAdapter(
            config["groundingdino_config_path"], config["groundingdino_model"], self.device
        )
        self.segmenter = SamAdapter(config["sam_model"], config["sam_model_type"], self.device)
        info(f"Inference models ready on {self.device}")
