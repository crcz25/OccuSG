import queue
import threading

import numpy as np

from semantic_perception.worker import Frame, WorkerPool


def test_queue_overflow_reports_drop():
    warnings = []
    pool = WorkerPool.__new__(WorkerPool)
    pool._queues = [queue.Queue(maxsize=1)]
    pool._next_queue = 0
    pool._lock = threading.Lock()
    pool._warning = warnings.append
    frame = Frame(7, 0, 0, "camera", np.empty((1, 1, 3)), np.empty((1, 1)), (1, 1, 0, 0))
    assert pool.submit(frame)
    assert not pool.submit(frame)
    assert "Dropped RGB-D frame 7" in warnings[0]


def test_best_class_uses_cached_text_embeddings():
    pool = WorkerPool.__new__(WorkerPool)
    pool.prompts = ["chair", "table"]
    pool.text_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    label, score = pool.best_class(np.array([0.2, 0.8], dtype=np.float32))

    assert label == "table"
    assert score == np.float32(0.8)
