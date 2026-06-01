"""Bounded detection-message queue and diagnostics for scene-graph ingestion."""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class QueuedDetectionMessage:
    """Detection message captured by the subscription callback."""

    sequence: int
    msg: object
    received_ros_time_sec: float


class DetectionInputQueue:
    """Small thread-safe FIFO used to keep detection callbacks non-blocking."""

    def __init__(self, max_messages: int = 100):
        self.max_messages = max(1, int(max_messages))
        self._queue: deque[QueuedDetectionMessage] = deque()
        self._lock = threading.Lock()
        self._sequence = 0
        self._rejections = Counter()
        self.stats = {
            "messages_received": 0,
            "detections_received": 0,
            "messages_dropped_queue_full": 0,
            "messages_dropped_tf": 0,
            "messages_applied": 0,
            "detections_accepted": 0,
            "detections_rejected": 0,
            "objects_created": 0,
            "objects_updated": 0,
            "last_msg_stamp_sec": None,
            "last_msg_frame_id": "",
            "last_received_ros_time_sec": None,
            "last_object_creation_ros_time_sec": None,
            "last_object_update_ros_time_sec": None,
        }

    def enqueue(self, msg: object, received_ros_time_sec: float) -> QueuedDetectionMessage:
        """Append a detection message, dropping the oldest one if full."""
        with self._lock:
            if len(self._queue) >= self.max_messages:
                self._queue.popleft()
                self.stats["messages_dropped_queue_full"] += 1

            self._sequence += 1
            queued = QueuedDetectionMessage(
                sequence=self._sequence,
                msg=msg,
                received_ros_time_sec=float(received_ros_time_sec),
            )
            self._queue.append(queued)

            detections = getattr(msg, "detections", []) or []
            header = getattr(msg, "header", None)
            stamp = getattr(header, "stamp", None)
            stamp_sec = None
            if stamp is not None:
                stamp_sec = float(getattr(stamp, "sec", 0)) + float(
                    getattr(stamp, "nanosec", 0)
                ) * 1e-9

            self.stats["messages_received"] += 1
            self.stats["detections_received"] += len(detections)
            self.stats["last_msg_stamp_sec"] = stamp_sec
            self.stats["last_msg_frame_id"] = str(getattr(header, "frame_id", "") or "")
            self.stats["last_received_ros_time_sec"] = float(received_ros_time_sec)
            return queued

    def pop_batch(self, max_messages: int) -> List[QueuedDetectionMessage]:
        """Pop up to ``max_messages`` from the front of the queue."""
        max_messages = max(1, int(max_messages))
        batch: List[QueuedDetectionMessage] = []
        with self._lock:
            while self._queue and len(batch) < max_messages:
                batch.append(self._queue.popleft())
        return batch

    def push_front(self, items: Iterable[QueuedDetectionMessage]) -> None:
        """Return unprocessed items to the front in their original order."""
        items = list(items)
        if not items:
            return
        with self._lock:
            for item in reversed(items):
                self._queue.appendleft(item)
            while len(self._queue) > self.max_messages:
                self._queue.pop()
                self.stats["messages_dropped_queue_full"] += 1

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)

    def record_tf_rejection(self, reason: str = "tf_lookup_failed") -> None:
        with self._lock:
            self.stats["messages_dropped_tf"] += 1
            self._rejections[str(reason)] += 1

    def record_apply_result(
        self,
        result_stats: dict,
        applied_ros_time_sec: float,
    ) -> None:
        """Fold object-manager result counters into aggregate diagnostics."""
        with self._lock:
            self.stats["messages_applied"] += 1
            accepted = int(result_stats.get("accepted_detections", 0))
            rejected = int(result_stats.get("rejected_detections", 0))
            created = int(result_stats.get("new_objects", 0))
            updated = int(result_stats.get("updated_objects", 0))
            self.stats["detections_accepted"] += accepted
            self.stats["detections_rejected"] += rejected
            self.stats["objects_created"] += created
            self.stats["objects_updated"] += updated
            if created:
                self.stats["last_object_creation_ros_time_sec"] = float(
                    applied_ros_time_sec
                )
            if updated:
                self.stats["last_object_update_ros_time_sec"] = float(
                    applied_ros_time_sec
                )
            for reason, count in dict(
                result_stats.get("rejected_by_reason", {}) or {}
            ).items():
                self._rejections[str(reason)] += int(count)

    def snapshot(self) -> dict:
        """Return a stable diagnostics dictionary."""
        with self._lock:
            snapshot = dict(self.stats)
            snapshot["pending_messages"] = len(self._queue)
            snapshot["rejected_by_reason"] = dict(self._rejections)
            return snapshot
