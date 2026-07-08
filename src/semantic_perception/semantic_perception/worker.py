"""Bounded multi-worker inference orchestration, with no ROS dependencies."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from semantic_perception.inference.crop_embeddings import CropEmbedding, encode_object_crops
from semantic_perception.inference.embedding_cache import load_or_generate
from semantic_perception.inference.geometry import Geometry3D, compute_geometry
from semantic_perception.inference.models import Detection, ModelBundle


@dataclass(frozen=True)
class Frame:
    sequence: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: tuple[float, float, float, float]


@dataclass
class Proposal:
    detection: Detection
    mask: np.ndarray | None
    embeddings: CropEmbedding
    geometry: Geometry3D


@dataclass
class FrameResult:
    frame: Frame
    proposals: list[Proposal]
    error: str = ""


class WorkerPool:
    def __init__(
        self,
        config: dict,
        warning: Callable[[str], None],
        info: Callable[[str], None],
    ):
        self._config = config
        self._warning = warning
        count = int(config["num_worker_threads"])
        size = int(config["frame_queue_size"])
        if count <= 0 or size <= 0:
            raise ValueError("num_worker_threads and frame_queue_size must be positive")
        devices = list(config["devices"])
        if not devices:
            devices = [str(config["device"])]
        self._queues = [queue.Queue(maxsize=size) for _ in range(count)]
        result_size = int(config.get("result_queue_size", max(2, count * 2)))
        if result_size <= 0:
            raise ValueError("result_queue_size must be positive")
        self._results: queue.Queue[FrameResult] = queue.Queue(maxsize=result_size)
        self._stopping = threading.Event()
        self._next_queue = 0
        self._lock = threading.Lock()

        # Loading happens synchronously at startup, never in a frame callback.
        self._models = [
            ModelBundle(config, devices[index % len(devices)], warning, info)
            for index in range(count)
        ]
        # Every worker owns its OpenCLIP model. The first creates the cache and
        # subsequent workers strictly reuse it.
        self.prompts, self.text_embeddings, _ = load_or_generate(
            config["prompt_csv_path"],
            config["class_embedding_cache_path"],
            config["openclip_model"],
            self._models[0].clip.encode_texts,
            info,
        )
        # Text embeddings prepare the HM3D vocabulary for a later semantic
        # module. Detection itself stays class-agnostic and uses a short prompt.
        self._prompt = str(config["groundingdino_prompt"]).strip()
        self._threads = [
            threading.Thread(
                target=self._run,
                args=(index,),
                daemon=True,
                name=f"semantic-inference-{index}",
            )
            for index in range(count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, frame: Frame) -> bool:
        """Non-blocking enqueue; tries each worker before dropping the frame."""
        with self._lock:
            start = self._next_queue
            self._next_queue = (self._next_queue + 1) % len(self._queues)
        for offset in range(len(self._queues)):
            target = self._queues[(start + offset) % len(self._queues)]
            try:
                target.put_nowait(frame)
                return True
            except queue.Full:
                continue
        self._warning(f"Dropped RGB-D frame {frame.sequence}: all worker queues are full")
        return False

    def get_result(self) -> FrameResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stopping.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            self._warning(f"Inference threads did not stop within timeout: {alive}")

    def _run(self, index: int) -> None:
        frames = self._queues[index]
        models = self._models[index]
        while not self._stopping.is_set():
            try:
                frame = frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                result = self._process(frame, models)
            except Exception as exc:
                result = FrameResult(frame, [], str(exc))
                self._warning(f"Frame {frame.sequence} inference failed: {exc}")
            self._put_result(result)
            frames.task_done()

    def _process(self, frame: Frame, models: ModelBundle) -> FrameResult:
        detections = models.detector.detect(
            frame.rgb,
            self._prompt,
            float(self._config["detection_threshold"]),
            float(self._config["text_threshold"]),
        )
        if not detections:
            return FrameResult(frame, [])
        boxes = [detection.box_xyxy for detection in detections]
        masks = models.segmenter.segment(frame.rgb, boxes)
        if len(masks) != len(boxes):
            self._warning("SAM returned the wrong number of masks; treating all masks as invalid")
            masks = [None] * len(boxes)
        embeddings = encode_object_crops(
            frame.rgb,
            boxes,
            masks,
            models.clip.encode_images,
            float(self._config["bbox_embedding_weight"]),
            float(self._config["masked_embedding_weight"]),
            self._warning,
        )
        proposals = []
        for detection, mask, embedding in zip(detections, masks, embeddings):
            geometry = compute_geometry(
                frame.depth_m,
                mask,
                detection.box_xyxy,
                frame.intrinsics,
                int(self._config["min_valid_depth_points"]),
                float(self._config["max_depth_m"]),
            )
            proposals.append(Proposal(detection, mask, embedding, geometry))
        return FrameResult(frame, proposals)

    def _put_result(self, result: FrameResult) -> None:
        try:
            self._results.put_nowait(result)
            return
        except queue.Full:
            pass
        try:
            discarded = self._results.get_nowait()
            self._warning(
                f"Discarded unpublished frame {discarded.frame.sequence}: result queue is full"
            )
        except queue.Empty:
            pass
        self._results.put_nowait(result)
