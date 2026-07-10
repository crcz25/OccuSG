"""Thin adapters around GroundingDINO, SAM/MobileSAM, and OpenCLIP."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
import traceback
from typing import Callable, Sequence
import warnings

import numpy as np
from PIL import Image

_GROUNDINGDINO_FALLBACK_REPORTED = False


@contextmanager
def _log_warnings(warning: Callable[[str], None]):
    """Forward Python warnings raised in this block to ``warning`` instead of hiding them."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        yield
    for item in captured:
        warning(
            f"{item.category.__name__} from {Path(item.filename).name}:{item.lineno}: "
            f"{item.message}"
        )


@contextmanager
def _quiet_known_upstream_stdout():
    """Hide noisy informational prints from third-party model constructors."""
    stream = io.StringIO()
    with redirect_stdout(stream):
        yield


@dataclass
class Detection:
    box_xyxy: np.ndarray
    confidence: float
    source: str = "groundingdino"


def bind_thread_to_device(device: str) -> None:
    """Make ``device`` the calling thread's current CUDA device.

    Upstream model code occasionally allocates on the *current* device rather
    than an explicit one; binding each worker thread once keeps every implicit
    allocation on that worker's own GPU.
    """
    if not device.startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(device))


_MEMORY_BUDGETS_APPLIED: set[tuple[str, float]] = set()


def apply_gpu_memory_budget(
    device: str,
    budget_gb: float,
    warning: Callable[[str], None],
    info: Callable[[str], None],
) -> None:
    """Cap this process's allocator on ``device`` to ``budget_gb`` GiB.

    Enforced through PyTorch's per-process CUDA memory fraction, so exceeding
    the budget raises a recoverable out-of-memory error for one frame instead
    of destabilizing the GPU. Backends that cannot enforce a strict limit get
    a clear warning and continue without one.
    """
    if budget_gb <= 0.0:
        _warn_if_device_is_shared(device, warning)
        return
    key = (device, float(budget_gb))
    if key in _MEMORY_BUDGETS_APPLIED:
        return
    _MEMORY_BUDGETS_APPLIED.add(key)
    if not device.startswith("cuda"):
        warning(
            f"gpu_memory_budget_gb={budget_gb:g} cannot be enforced on device "
            f"'{device}'; strict memory limiting is only available for CUDA"
        )
        return
    import torch

    try:
        target = torch.device(device)
        index = target.index if target.index is not None else torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(index).total_memory
        budget_bytes = budget_gb * 1024**3
        # The budget is a fraction of TOTAL memory, but other processes (for
        # example a display server on this GPU) already hold part of it. A
        # budget above the free amount would let this process starve them,
        # which can freeze the whole machine, so clamp against FREE memory
        # and keep one GiB of headroom for the co-resident processes.
        reserve_bytes = 1024**3
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = None
        if free_bytes is not None and budget_bytes > free_bytes - reserve_bytes:
            clamped_bytes = float(max(reserve_bytes, free_bytes - reserve_bytes))
            warning(
                f"gpu_memory_budget_gb={budget_gb:g} exceeds the "
                f"{free_bytes / 1024**3:.1f} GiB currently free on cuda:{index} "
                f"({(total_bytes - free_bytes) / 1024**3:.1f} GiB is already used "
                "by other processes, e.g. a display server); clamping this "
                f"process to {clamped_bytes / 1024**3:.1f} GiB so they are not starved"
            )
            budget_bytes = clamped_bytes
        fraction = budget_bytes / total_bytes
        if fraction >= 1.0:
            warning(
                f"gpu_memory_budget_gb={budget_gb:g} is not below cuda:{index}'s "
                f"{total_bytes / 1024**3:.1f} GiB capacity; no limit applied"
            )
            return
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        info(
            f"Limited this process to {budget_bytes / 1024**3:.1f} GiB of "
            f"cuda:{index}'s {total_bytes / 1024**3:.1f} GiB ({fraction:.0%}); "
            "allocations beyond the budget raise a recoverable out-of-memory error"
        )
    except AttributeError:
        warning(
            "This PyTorch build lacks set_per_process_memory_fraction; "
            "strict GPU memory limiting is unavailable"
        )
    except Exception as exc:
        warning(f"Could not apply GPU memory budget on {device}: {exc}")


_SHARED_DEVICE_WARNED: set[str] = set()


def _warn_if_device_is_shared(device: str, warning: Callable[[str], None]) -> None:
    """Warn when no budget protects other users of an already-busy GPU."""
    if not device.startswith("cuda") or device in _SHARED_DEVICE_WARNED:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        target = torch.device(device)
        index = target.index if target.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    except Exception:
        return
    used_bytes = total_bytes - free_bytes
    if used_bytes < 0.05 * total_bytes:
        return
    _SHARED_DEVICE_WARNED.add(device)
    warning(
        f"cuda:{index} already has {used_bytes / 1024**3:.1f} GiB in use by other "
        "processes (a display server or another job may share this GPU); set "
        "gpu_memory_budget_gb so this node cannot starve them"
    )


def cuda_memory_summary(devices) -> str:
    """One-line allocated/reserved VRAM summary for the given CUDA devices."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    parts = []
    seen: set[int] = set()
    for device in devices:
        name = str(device)
        if not name.startswith("cuda"):
            continue
        index = torch.device(name).index or 0
        if index in seen:
            continue
        seen.add(index)
        allocated = torch.cuda.memory_allocated(index) / 1024**3
        reserved = torch.cuda.memory_reserved(index) / 1024**3
        entry = f"cuda:{index} {allocated:.1f}/{reserved:.1f} GiB alloc/reserved"
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
            entry += f", {free_bytes / 1024**3:.1f} GiB free"
        except Exception:
            pass
        parts.append(entry)
    return ", ".join(parts)


def release_cuda_memory(devices, warning: Callable[[str], None] | None = None) -> None:
    """Return cached allocator blocks to the driver, e.g. during shutdown."""
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    for device in {str(item) for item in devices if str(item).startswith("cuda")}:
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        except Exception:
            if warning:
                warning(
                    f"Failed to release cached CUDA memory on {device}:\n"
                    f"{traceback.format_exc()}"
                )


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
    def __init__(
        self,
        model_name: str,
        checkpoint: str,
        device: str,
        warning: Callable[[str], None],
        text_batch_size: int = 64,
    ):
        if not model_name:
            raise ValueError("Parameter 'openclip_model' must not be empty")
        if text_batch_size <= 0:
            raise ValueError("Parameter 'text_embedding_batch_size' must be positive")
        try:
            with _log_warnings(warning):
                import open_clip
                import torch
        except ImportError as exc:
            raise RuntimeError("Install open_clip_torch and PyTorch to use OpenCLIP") from exc

        pretrained = checkpoint or None
        if checkpoint and ("/" in checkpoint or checkpoint.startswith(".")):
            pretrained = _require_file(checkpoint, "openclip_checkpoint_path")
        try:
            with _log_warnings(warning):
                self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=device
                )
                self._tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load OpenCLIP model '{model_name}': {exc}") from exc
        self._model.eval()
        self._device = device
        self._torch = torch
        self._text_batch_size = text_batch_size
        self.cache_key = f"{model_name}:{checkpoint or 'untrained'}"

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        tensors = [
            self._preprocess(Image.fromarray(np.asarray(image, dtype=np.uint8)))
            for image in images
        ]
        batch = self._torch.stack(tensors)
        if self._device.startswith("cuda"):
            # Pinned staging memory keeps the host-to-device copy asynchronous.
            batch = batch.pin_memory().to(self._device, non_blocking=True)
        else:
            batch = batch.to(self._device)
        with self._torch.inference_mode():
            values = self._model.encode_image(batch)
        return values.float().cpu().numpy().astype(np.float32)

    def encode_texts(self, prompts: Sequence[str]) -> np.ndarray:
        prompt_list = list(prompts)
        if not prompt_list:
            return np.empty((0, 0), dtype=np.float32)
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise RuntimeError("Install tqdm to display text embedding progress") from exc

        encoded_batches = []
        with self._torch.inference_mode():
            with tqdm(
                total=len(prompt_list),
                desc="OpenCLIP text embeddings",
                unit="prompt",
                dynamic_ncols=True,
            ) as progress:
                for start in range(0, len(prompt_list), self._text_batch_size):
                    batch_prompts = prompt_list[start : start + self._text_batch_size]
                    tokens = self._tokenizer(batch_prompts).to(self._device)
                    values = self._model.encode_text(tokens)
                    encoded_batches.append(values.float().cpu())
                    progress.update(len(batch_prompts))
        return (
            self._torch.cat(encoded_batches, dim=0)
            .numpy()
            .astype(np.float32, copy=False)
        )


class GroundingDinoAdapter:
    def __init__(
        self,
        config_path: str,
        weights_path: str,
        device: str,
        warning: Callable[[str], None],
    ):
        self._warning = warning
        config = _require_file(config_path, "groundingdino_config_path")
        weights = _require_file(weights_path, "groundingdino_model")
        try:
            with _log_warnings(warning):
                import torch
                from groundingdino.util.inference import Model
        except ImportError as exc:
            raise RuntimeError(
                "GroundingDINO is not importable; install the official GroundingDINO package"
            ) from exc
        _enable_groundingdino_fallback(device, warning)
        try:
            with _log_warnings(warning), _quiet_known_upstream_stdout():
                self._model = Model(
                    model_config_path=config,
                    model_checkpoint_path=weights,
                    device=device,
                )
        except Exception as exc:
            raise RuntimeError(f"Failed to load GroundingDINO: {exc}") from exc
        self._torch = torch

    def detect(
        self, rgb: np.ndarray, prompt: str, box_threshold: float, text_threshold: float
    ) -> list[Detection]:
        # The official high-level GroundingDINO API accepts OpenCV/BGR input,
        # while the rest of this package deliberately uses RGB.
        with _log_warnings(self._warning), self._torch.inference_mode():
            detections, _ = self._model.predict_with_caption(
                image=np.ascontiguousarray(rgb[..., ::-1]),
                caption=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
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
    def __init__(
        self,
        weights_path: str,
        model_type: str,
        device: str,
        warning: Callable[[str], None],
    ):
        weights = _require_file(weights_path, "sam_model")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for SAM inference") from exc
        self._torch = torch
        self._warning = warning
        registries = []
        errors = []
        for module_name in ("mobile_sam", "segment_anything"):
            try:
                with _log_warnings(warning):
                    module = __import__(
                        module_name, fromlist=["SamPredictor", "sam_model_registry"]
                    )
                registries.append((module.SamPredictor, module.sam_model_registry))
            except (ImportError, AttributeError) as exc:
                errors.append(f"{module_name}: {exc}")
        if not registries:
            raise RuntimeError(
                "Neither MobileSAM nor Segment Anything is importable: " + "; ".join(errors)
            )
        last_error: Exception | None = None
        for predictor_type, registry in registries:
            if model_type not in registry:
                continue
            try:
                with _log_warnings(warning):
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
        box_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        with _log_warnings(self._warning), self._torch.inference_mode():
            self._predictor.set_image(rgb)
            return self._segment_batched(rgb.shape[:2], box_array)

    def _segment_batched(
        self, image_hw: tuple[int, int], box_array: np.ndarray
    ) -> list[np.ndarray | None]:
        """Decode all boxes in one GPU call instead of one call per box."""
        torch = self._torch
        box_tensor = torch.as_tensor(box_array, device=self._predictor.device)
        transformed = self._predictor.transform.apply_boxes_torch(box_tensor, image_hw)
        masks, scores, _ = self._predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=True,
        )
        best = scores.argmax(dim=1)
        selected = masks[torch.arange(masks.shape[0], device=masks.device), best]
        return [np.asarray(mask, dtype=bool) for mask in selected.cpu().numpy()]


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
        openclip_device = select_device(str(config.get("openclip_device", self.device)), warning)
        detector_device = select_device(
            str(config.get("groundingdino_device", self.device)), warning
        )
        sam_device = select_device(str(config.get("sam_device", self.device)), warning)
        self.devices = sorted({self.device, openclip_device, detector_device, sam_device})
        budget_gb = float(config.get("gpu_memory_budget_gb", 0.0) or 0.0)
        for item in self.devices:
            apply_gpu_memory_budget(item, budget_gb, warning, info)
        info(
            "Loading inference models "
            f"(OpenCLIP={openclip_device}, GroundingDINO={detector_device}, SAM={sam_device})"
        )
        self.clip = OpenClipAdapter(
            config["openclip_model"],
            config["openclip_checkpoint_path"],
            openclip_device,
            warning,
            int(config.get("text_embedding_batch_size", 64)),
        )
        self.detector = GroundingDinoAdapter(
            config["groundingdino_config_path"],
            config["groundingdino_model"],
            detector_device,
            warning,
        )
        self.segmenter = SamAdapter(
            config["sam_model"], config["sam_model_type"], sam_device, warning
        )
        info("Inference models ready")


class _CudaFallbackProxy:
    """Delegate CUDA APIs except for the GroundingDINO extension check."""

    def __init__(self, cuda):
        self._cuda = cuda

    @staticmethod
    def is_available() -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._cuda, name)


class _TorchFallbackProxy:
    def __init__(self, torch):
        self._torch = torch
        self.cuda = _CudaFallbackProxy(torch.cuda)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def _enable_groundingdino_fallback(
    device: str, warning: Callable[[str], None]
) -> None:
    """Use GroundingDINO's PyTorch deformable-attention kernel without ``_C``.

    CUDA runtime images do not include nvcc, so the optional extension cannot be
    compiled reproducibly there. The upstream implementation already contains a
    differentiable PyTorch kernel; its dispatch condition only needs to be kept
    away from the absent extension.
    """
    if not device.startswith("cuda"):
        return
    global _GROUNDINGDINO_FALLBACK_REPORTED
    try:
        with _log_warnings(warning):
            import groundingdino._C  # noqa: F401

        return
    except (ImportError, OSError):
        pass
    with _log_warnings(warning):
        from groundingdino.models.GroundingDINO import ms_deform_attn

    if not isinstance(ms_deform_attn.torch, _TorchFallbackProxy):
        ms_deform_attn.torch = _TorchFallbackProxy(ms_deform_attn.torch)
    if not _GROUNDINGDINO_FALLBACK_REPORTED:
        _GROUNDINGDINO_FALLBACK_REPORTED = True
        warning(
            "GroundingDINO custom CUDA ops are unavailable; using its portable "
            "PyTorch deformable-attention kernel"
        )
