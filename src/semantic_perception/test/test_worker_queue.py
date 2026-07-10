import queue
import threading
import time
import types

import numpy as np
import pytest

from semantic_perception.worker import (
    Frame,
    FrameResult,
    WorkerPool,
    _FrameQueue,
    _Statistics,
)


class _FakeThread:
    """Stand-in for a worker thread whose liveness is controlled by the test."""

    def __init__(self, alive: bool = True):
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def make_frame(sequence: int) -> Frame:
    return Frame(
        sequence, 0, 0, "camera", np.empty((1, 1, 3)), np.empty((1, 1)), (1, 1, 0, 0)
    )


def make_pool(
    frame_queue_size: int = 1,
    result_queue_size: int = 8,
    active: list[bool] | None = None,
    drop_report_every: int = 30,
) -> tuple[WorkerPool, list[str]]:
    warnings: list[str] = []
    pool = WorkerPool.__new__(WorkerPool)
    pool._frames = _FrameQueue(frame_queue_size)
    pool._results = queue.Queue(maxsize=result_queue_size)
    pool._results_lock = threading.Lock()
    pool._stats = _Statistics()
    pool._stopping = threading.Event()
    pool._dropped_frames = 0
    pool._drop_report_every = drop_report_every
    pool._drop_stale_results = True
    pool._latest_result_sequence = -1
    pool._warning = warnings.append
    pool._error = warnings.append
    active = [True] if active is None else active
    pool._devices = ["cpu"] * len(active)
    pool._threads = [_FakeThread(alive) for alive in active]
    pool._last_report_time = time.monotonic() - 10.0
    pool._last_counts = {}
    pool._last_timing_totals = {}
    pool._last_timing_counts = {}
    return pool, warnings


def test_frame_queue_is_fifo_and_drops_oldest_on_overflow():
    frames = _FrameQueue(2)
    assert frames.put(make_frame(0)) is None
    assert frames.put(make_frame(1)) is None
    dropped = frames.put(make_frame(2))
    assert dropped is not None and dropped.sequence == 0
    assert frames.get(timeout=0.01).sequence == 1
    assert frames.get(timeout=0.01).sequence == 2
    assert frames.get(timeout=0.01) is None
    assert len(frames) == 0


def test_submit_keeps_newest_frame_and_reports_drop():
    pool, warnings = make_pool(frame_queue_size=1)
    assert pool.submit(make_frame(7))
    assert pool.submit(make_frame(8))
    assert len(pool._frames) == 1
    assert pool._frames.get(timeout=0.01).sequence == 8
    assert "Dropped RGB-D frame 7" in warnings[0]
    assert "kept the newer frame" in warnings[0]
    assert "1 total dropped" in warnings[0]


def test_submit_with_no_active_workers_drops_and_throttles_reports():
    pool, warnings = make_pool(active=[False], drop_report_every=3)
    for sequence in range(5):
        assert not pool.submit(make_frame(sequence))

    assert len(warnings) == 2
    assert "no inference workers are active" in warnings[0]
    assert "1 total dropped" in warnings[0]
    assert "3 total dropped" in warnings[1]


def test_get_result_is_monotonic_and_drops_stale_results():
    pool, _ = make_pool()
    for sequence in (2, 1, 3):
        pool._put_result(FrameResult(make_frame(sequence), []))

    assert pool.get_result().frame.sequence == 2
    assert pool.get_result().frame.sequence == 3
    assert pool.get_result() is None
    assert pool._stats.snapshot()[0]["dropped_stale"] == 1


def test_put_result_discards_oldest_when_result_queue_is_full():
    pool, warnings = make_pool(result_queue_size=1)
    pool._put_result(FrameResult(make_frame(1), []))
    pool._put_result(FrameResult(make_frame(2), []))

    assert "Discarded unpublished result for frame 1" in warnings[0]
    assert pool.get_result().frame.sequence == 2
    assert pool._stats.snapshot()[0]["dropped_result"] == 1


def test_best_class_uses_cached_text_embeddings():
    pool = WorkerPool.__new__(WorkerPool)
    pool.prompts = ["chair", "table"]
    pool.text_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    label, score = pool.best_class(np.array([0.2, 0.8], dtype=np.float32))

    assert label == "table"
    assert score == np.float32(0.8)


def test_worker_run_logs_full_exception_and_terminates_without_recovery():
    """A raw processing failure is logged in full and the worker thread stops;
    nothing classifies, hides, or "recovers" from it."""
    pool, messages = make_pool()
    pool._models = [types.SimpleNamespace(device="cpu")]
    pool._frames.put(make_frame(7))

    def fail(_frame, _models):
        raise RuntimeError("CUDA error: unspecified launch failure")

    pool._process = fail

    with pytest.raises(RuntimeError, match="unspecified launch failure"):
        pool._run(0)

    # The complete original exception (with traceback) was logged as-is.
    assert any(
        "CUDA error: unspecified launch failure" in message and "Traceback" in message
        for message in messages
    )
    # No result was fabricated for the failed frame.
    assert pool.get_result() is None
    assert pool._stats.snapshot()[0]["failed"] == 1


def test_worker_run_processes_multiple_frames_until_stopped():
    pool, _ = make_pool(frame_queue_size=2)
    pool._models = [types.SimpleNamespace(device="cpu")]
    pool._frames.put(make_frame(1))
    pool._frames.put(make_frame(2))

    processed = []

    def succeed(frame, _models):
        processed.append(frame.sequence)
        if len(processed) == 2:
            pool._stopping.set()
        return FrameResult(frame, [])

    pool._process = succeed
    pool._run(0)

    assert processed == [1, 2]
    assert pool.get_result().frame.sequence == 1
    assert pool.get_result().frame.sequence == 2


def test_all_workers_failed_reflects_thread_liveness():
    pool, _ = make_pool(active=[False, False])
    assert pool.all_workers_failed()

    pool, _ = make_pool(active=[False, True])
    assert not pool.all_workers_failed()


def test_stats_report_summarizes_activity_and_warns_on_overload():
    pool, _ = make_pool()
    pool._stats.count("submitted", 20)
    pool._stats.count("completed", 10)
    pool._stats.count("dropped_input", 5)
    for _ in range(10):
        pool._stats.record_timing("detect", 0.050)
        pool._stats.record_timing("latency", 0.200)

    info, warning = pool.stats_report()

    assert info is not None
    assert "processed" in info and "queue 0/1" in info and "workers 1/1 active" in info
    assert "detect 50" in info
    assert "latency 200 ms" in info
    assert warning is not None
    assert "exceeds processing capacity" in warning

    info, warning = pool.stats_report()
    assert info is None and warning is None
