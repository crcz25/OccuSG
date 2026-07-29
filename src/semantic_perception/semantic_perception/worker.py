"""Bounded multi-worker inference orchestration, with no ROS dependencies."""

from __future__ import annotations

import collections
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

import numpy as np

from semantic_perception.inference.crop_embeddings import CropEmbedding, encode_object_crops
from semantic_perception.inference.embedding_cache import build_label_embedding_cache
from semantic_perception.inference.geometry import Geometry3D, compute_geometry
from semantic_perception.inference.models import (
    Detection,
    ModelBundle,
    bind_thread_to_device,
    cuda_memory_summary,
    non_maximum_suppression,
    release_cuda_memory,
)
from semantic_perception.inference.vocabulary import (
    build_detector_prompt,
    parse_excluded_labels,
    resolve_global_index,
    summarize,
)


@dataclass(frozen=True)
class Frame:
    sequence: int
    stamp_sec: int
    stamp_nanosec: int
    frame_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: tuple[float, float, float, float]
    camera_to_world: np.ndarray | None = None
    received_monotonic: float = 0.0
    # Original optical frame and the timestamp of the transform actually applied.
    source_frame_id: str = ""
    transform_stamp_sec: float = 0.0


@dataclass
class Proposal:
    detection: Detection
    mask: np.ndarray | None
    embeddings: CropEmbedding
    geometry: Geometry3D
    mask_pixels: int = 0


@dataclass
class FrameResult:
    frame: Frame
    proposals: list[Proposal]


class _FrameQueue:
    """Bounded FIFO shared by all workers; overflow evicts the OLDEST frame.

    Keeping the newest frames bounds end-to-end latency under overload, which
    matters more than history completeness for a real-time perception stream.
    A shared queue also load-balances automatically: whichever worker is free
    takes the next frame, so no frame is stuck behind a slow worker.
    """

    def __init__(self, maxsize: int):
        if maxsize <= 0:
            raise ValueError("frame_queue_size must be positive")
        self.maxsize = maxsize
        self._items: collections.deque[Frame] = collections.deque()
        self._condition = threading.Condition()

    def put(self, frame: Frame) -> Frame | None:
        """Append ``frame``; return the evicted oldest frame when full."""
        with self._condition:
            dropped = self._items.popleft() if len(self._items) >= self.maxsize else None
            self._items.append(frame)
            self._condition.notify()
        return dropped

    def get(self, timeout: float) -> Frame | None:
        with self._condition:
            if self._condition.wait_for(lambda: len(self._items) > 0, timeout):
                return self._items.popleft()
            return None

    def clear(self) -> None:
        with self._condition:
            self._items.clear()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


class _Statistics:
    """Thread-safe counters and stage-timing accumulators."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: collections.Counter[str] = collections.Counter()
        self._timing_totals: dict[str, float] = collections.defaultdict(float)
        self._timing_counts: collections.Counter[str] = collections.Counter()

    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def record_timing(self, name: str, seconds: float) -> None:
        with self._lock:
            self._timing_totals[name] += seconds
            self._timing_counts[name] += 1

    def snapshot(self) -> tuple[dict[str, int], dict[str, float], dict[str, int]]:
        with self._lock:
            return dict(self._counts), dict(self._timing_totals), dict(self._timing_counts)


class WorkerPool:
    """One inference worker per device, all pulling from a shared frame queue."""

    def __init__(
        self,
        config: dict,
        warning: Callable[[str], None],
        info: Callable[[str], None],
        error: Callable[[str], None] | None = None,
    ):
        self._config = config
        self._warning = warning
        self._error = error or warning
        devices = [str(device) for device in (config.get("devices") or [])]
        if not devices:
            devices = [str(config["device"])]
        count = int(config["num_worker_threads"])
        if count < 0:
            raise ValueError("num_worker_threads must be 0 (one per device) or positive")
        if count == 0:
            count = len(devices)
        if count > len(devices):
            warning(
                f"num_worker_threads={count} exceeds the {len(devices)} configured "
                "devices; workers will share devices and each loads a full model "
                "copy (more GPU memory for little extra throughput)"
            )
        elif count < len(devices):
            warning(
                f"num_worker_threads={count} uses only the first {count} of the "
                f"{len(devices)} configured devices"
            )
        result_size = int(config.get("result_queue_size", max(2, count * 2)))
        if result_size <= 0:
            raise ValueError("result_queue_size must be positive")
        self._frames = _FrameQueue(int(config["frame_queue_size"]))
        self._results: queue.Queue[FrameResult] = queue.Queue(maxsize=result_size)
        self._results_lock = threading.Lock()
        self._stopping = threading.Event()
        self._stats = _Statistics()
        self._dropped_frames = 0
        self._drop_report_every = max(1, int(config.get("drop_report_every", 30)))
        self._drop_stale_results = bool(config.get("drop_stale_results", True))
        self._latest_result_sequence = -1
        self._last_report_time = time.monotonic()
        self._last_counts: dict[str, int] = {}
        self._last_timing_totals: dict[str, float] = {}
        self._last_timing_counts: dict[str, int] = {}

        # Loading happens synchronously at startup, never in a frame callback.
        self._models = [
            ModelBundle(config, devices[index % len(devices)], warning, info)
            for index in range(count)
        ]
        self._devices = [bundle.device for bundle in self._models]
        info(
            "Worker-to-device assignment: "
            + ", ".join(f"worker {i} -> {d}" for i, d in enumerate(self._devices))
        )
        # Every worker owns its OpenCLIP model. The first creates the cache and
        # subsequent workers strictly reuse it. Text embeddings are computed once
        # here and never re-encoded per detection or per frame.
        self.labels = build_label_embedding_cache(
            config["class_labels_path"],
            config["class_embedding_cache_path"],
            self._models[0].clip.cache_key,
            self._models[0].clip.encode_texts,
            info,
        )
        info(self.labels.summary())
        excluded_labels = parse_excluded_labels(config.get("excluded_class_labels"))
        unknown = self.labels.vocabulary.unknown_labels(excluded_labels)
        if unknown:
            warning(
                f"excluded_class_labels contains {list(unknown)}, which are not in "
                f"{config['class_labels_path']}; they exclude nothing"
            )
        self._prompt = build_detector_prompt(
            self.labels.vocabulary,
            int(config["detector_vocabulary_size"]),
            excluded_labels,
        )
        tokens, token_limit = self._models[0].detector.caption_token_budget(
            self._prompt.caption
        )
        info("Grounding DINO vocabulary: " + summarize(self._prompt)
             + f", {tokens}/{token_limit} text tokens")
        if tokens > token_limit:
            raise ValueError(
                f"The detector caption needs {tokens} text tokens but the model "
                f"accepts {token_limit}; the trailing classes would be truncated "
                f"and undetectable. Lower detector_vocabulary_size "
                f"(currently {len(self._prompt.labels)})."
            )
        self._merge_iou_threshold = float(config["detector_merge_iou_threshold"])
        if not 0.0 <= self._merge_iou_threshold <= 1.0:
            raise ValueError("detector_merge_iou_threshold must be between 0 and 1")
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
        """Accept the newest frame; evict the oldest queued frame when full."""
        self._stats.count("submitted")
        if not any(thread.is_alive() for thread in self._threads):
            self._note_dropped_frame(frame.sequence, "no inference workers are active")
            return False
        dropped = self._frames.put(frame)
        if dropped is not None:
            self._note_dropped_frame(
                dropped.sequence, "input queue overflow; kept the newer frame"
            )
        return True

    def get_result(self) -> FrameResult | None:
        """Pop the next result; must be called from a single consumer thread.

        With ``drop_stale_results`` enabled, results finishing after a newer
        frame was already delivered are discarded so downstream consumers see
        monotonically increasing sequences/timestamps.
        """
        while True:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                return None
            if (
                self._drop_stale_results
                and result.frame.sequence <= self._latest_result_sequence
            ):
                self._stats.count("dropped_stale")
                continue
            self._latest_result_sequence = max(
                self._latest_result_sequence, result.frame.sequence
            )
            return result

    def best_class(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return the closest cached OpenCLIP class for debug visualization."""
        result = self.labels.classify(embedding)
        if result is None:
            return "", 0.0
        global_index, similarity = result
        return self.labels.vocabulary.label_at(global_index) or "", similarity

    def stats_report(self) -> tuple[str | None, str | None]:
        """Return ``(info, warning)`` describing activity since the last call."""
        counts, timing_totals, timing_counts = self._stats.snapshot()
        now = time.monotonic()
        elapsed = max(now - self._last_report_time, 1e-6)
        count_delta = {
            key: counts.get(key, 0) - self._last_counts.get(key, 0)
            for key in set(counts) | set(self._last_counts)
        }
        total_delta = {
            key: timing_totals.get(key, 0.0) - self._last_timing_totals.get(key, 0.0)
            for key in timing_totals
        }
        events_delta = {
            key: timing_counts.get(key, 0) - self._last_timing_counts.get(key, 0)
            for key in timing_counts
        }
        self._last_report_time = now
        self._last_counts = counts
        self._last_timing_totals = timing_totals
        self._last_timing_counts = timing_counts

        submitted = count_delta.get("submitted", 0)
        processed = count_delta.get("completed", 0) + count_delta.get("failed", 0)
        pending = len(self._frames)
        if submitted == 0 and processed == 0 and pending == 0:
            return None, None

        def mean_ms(name: str) -> str:
            events = events_delta.get(name, 0)
            if events <= 0:
                return "-"
            return f"{total_delta.get(name, 0.0) / events * 1000.0:.0f}"

        total = len(self._threads)
        active = sum(1 for thread in self._threads if thread.is_alive())
        input_hz = submitted / elapsed
        processed_hz = processed / elapsed
        dropped_input = count_delta.get("dropped_input", 0)
        info = (
            f"input {input_hz:.1f} Hz, processed {processed_hz:.1f} Hz, "
            f"queue {pending}/{self._frames.maxsize}, "
            f"workers {active}/{total} active, "
            f"latency {mean_ms('latency')} ms "
            f"(detect {mean_ms('detect')} / segment {mean_ms('segment')} / "
            f"embed {mean_ms('embed')} / geometry {mean_ms('geometry')} ms), "
            f"dropped input {dropped_input} / results "
            f"{count_delta.get('dropped_result', 0)} / stale "
            f"{count_delta.get('dropped_stale', 0)}"
        )
        info += (
            f", labels (unmapped {count_delta.get('unmapped_phrase', 0)} / "
            f"merged {count_delta.get('merged_duplicate', 0)} / "
            f"cache miss {count_delta.get('label_cache_miss', 0)})"
        )
        memory = cuda_memory_summary(self._devices)
        if memory:
            info += f"; VRAM {memory}"
        warning = None
        if dropped_input > 0:
            warning = (
                f"Input rate {input_hz:.1f} Hz exceeds processing capacity "
                f"{processed_hz:.1f} Hz; dropped {dropped_input} frames in the "
                f"last {elapsed:.0f} s (oldest first, the newest frames are kept)"
            )
        if active == 0:
            warning = "All inference workers have stopped; every new frame is dropped"
        return info, warning

    def close(self) -> None:
        self._stopping.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            self._warning(f"Inference threads did not stop within timeout: {alive}")
        # Release queued frames/results promptly, then return cached VRAM.
        self._frames.clear()
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
        self._models = []
        release_cuda_memory(self._devices, self._warning)

    def _run(self, index: int) -> None:
        models = self._models[index]
        try:
            bind_thread_to_device(models.device)
        except Exception:
            self._error(
                f"Worker {index} could not bind to {models.device}:\n"
                f"{traceback.format_exc()}"
            )
        while not self._stopping.is_set():
            frame = self._frames.get(timeout=0.1)
            if frame is None:
                continue
            try:
                result = self._process(frame, models)
            except Exception:
                self._stats.count("failed")
                self._error(
                    f"Worker {index} ({models.device}) failed processing frame "
                    f"{frame.sequence}:\n{traceback.format_exc()}"
                )
                raise
            self._stats.count("completed")
            self._put_result(result)

    def all_workers_failed(self) -> bool:
        return not any(thread.is_alive() for thread in self._threads)

    def counters(self) -> dict[str, int]:
        """Return cumulative pipeline counters for diagnostics and tests."""
        counts, _, _ = self._stats.snapshot()
        return dict(counts)

    def _detect_vocabulary(self, rgb: np.ndarray, models: ModelBundle) -> list[Detection]:
        """Detect the whole vocabulary in one pass and label each result.

        The prompt carries a local-to-global index map, so a detector result
        always resolves to exactly one vocabulary entry. Results whose phrase maps
        to no entry are dropped rather than given a fabricated label.
        """
        candidates: list[Detection] = []
        unmapped = 0
        for box, confidence, local_index, phrase in models.detector.detect_with_classes(
            rgb,
            self._prompt.labels,
            self._prompt.caption,
            float(self._config["detection_threshold"]),
            float(self._config["text_threshold"]),
        ):
            global_index = resolve_global_index(
                self._prompt, self.labels.vocabulary, local_index, phrase
            )
            if global_index is None:
                unmapped += 1
                continue
            class_name = self.labels.vocabulary.label_at(global_index)
            if not class_name:
                unmapped += 1
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
        if unmapped:
            self._stats.count("unmapped_phrase", unmapped)
        if len(candidates) < 2:
            return candidates
        # A multi-class caption can return several boxes over the same region
        # under neighbouring labels; keep the highest-confidence one per location.
        kept = non_maximum_suppression(
            np.asarray([d.box_xyxy for d in candidates], dtype=np.float64).reshape(-1, 4),
            np.asarray([d.confidence for d in candidates], dtype=np.float64),
            self._merge_iou_threshold,
        )
        self._stats.count("merged_duplicate", max(0, len(candidates) - len(kept)))
        return [candidates[index] for index in kept]

    def _process(self, frame: Frame, models: ModelBundle) -> FrameResult:
        started = time.monotonic()
        detections = self._detect_vocabulary(frame.rgb, models)
        detected = time.monotonic()
        boxes = [detection.box_xyxy for detection in detections]
        masks = models.segmenter.segment(frame.rgb, boxes)
        if len(masks) != len(boxes):
            self._warning("SAM returned the wrong number of masks; treating all masks as invalid")
            masks = [None] * len(boxes)
        segmented = time.monotonic()
        # The label embedding is looked up, never re-encoded, and a cache miss
        # invalidates the detection instead of contributing a zero vector.
        label_embeddings: list[np.ndarray | None] = []
        for detection in detections:
            vector = self.labels.by_index(detection.class_index)
            if vector is None:
                self._stats.count("label_cache_miss")
            label_embeddings.append(vector)
        embeddings = encode_object_crops(
            frame.rgb,
            boxes,
            masks,
            label_embeddings,
            models.clip.encode_images,
            self._warning,
        )
        embedded = time.monotonic()
        proposals = []
        for detection, mask, embedding in zip(detections, masks, embeddings):
            geometry = compute_geometry(
                frame.depth_m,
                mask,
                detection.box_xyxy,
                frame.intrinsics,
                int(self._config["min_valid_depth_points"]),
                float(self._config["max_depth_m"]),
                frame.camera_to_world,
            )
            proposals.append(
                Proposal(
                    detection,
                    mask,
                    embedding,
                    geometry,
                    mask_pixels=int(np.count_nonzero(mask)) if mask is not None else 0,
                )
            )
        finished = time.monotonic()
        self._stats.record_timing("detect", detected - started)
        self._stats.record_timing("segment", segmented - detected)
        self._stats.record_timing("embed", embedded - segmented)
        self._stats.record_timing("geometry", finished - embedded)
        self._stats.record_timing("total", finished - started)
        if frame.received_monotonic > 0.0:
            self._stats.record_timing("latency", finished - frame.received_monotonic)
        return FrameResult(frame, proposals)

    def _put_result(self, result: FrameResult) -> None:
        # The lock serializes producers so the final put cannot race another
        # worker between the eviction and the insert; the consumer only ever
        # removes items, which can only create space.
        with self._results_lock:
            try:
                self._results.put_nowait(result)
                return
            except queue.Full:
                pass
            try:
                discarded = self._results.get_nowait()
                self._stats.count("dropped_result")
                self._warning(
                    f"Discarded unpublished result for frame "
                    f"{discarded.frame.sequence}: result queue is full"
                )
            except queue.Empty:
                pass
            self._results.put_nowait(result)

    def _note_dropped_frame(self, sequence: int, reason: str) -> None:
        self._stats.count("dropped_input")
        self._dropped_frames += 1
        if self._dropped_frames == 1 or self._dropped_frames % self._drop_report_every == 0:
            self._warning(
                f"Dropped RGB-D frame {sequence}: {reason} "
                f"({self._dropped_frames} total dropped)"
            )
