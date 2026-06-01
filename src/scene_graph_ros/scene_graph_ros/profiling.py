"""Lightweight wall-clock runtime profiling helpers."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import socket
import time
from typing import Dict, Iterator, Optional


class ProfilingRecorder:
    """Collect per-stage monotonic wall-clock samples and write JSON summaries."""

    def __init__(
        self,
        *,
        node_name: str,
        package_name: str,
        run_name: str,
        output_path: str,
        enabled: bool = False,
        save_on_shutdown: bool = True,
        discard_first_n: int = 0,
        file_tag: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.node_name = str(node_name)
        self.package_name = str(package_name)
        self.run_name = str(run_name or "profiling_run")
        self.output_path = Path(str(output_path or ".")).expanduser()
        self.enabled = bool(enabled)
        self.save_on_shutdown = bool(save_on_shutdown)
        self.discard_first_n = max(0, int(discard_first_n))
        self.file_tag = str(file_tag or self.node_name)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: Optional[str] = None
        self._samples: Dict[str, list[dict]] = {}
        self._saved = False
        self._metadata = self._machine_metadata()
        if metadata:
            self._metadata.update(metadata)

        if self.enabled and self.save_on_shutdown:
            atexit.register(self.save)
            self._install_signal_handler(signal.SIGINT)
            self._install_signal_handler(signal.SIGTERM)

    def record(
        self,
        stage_name: str,
        elapsed_ms: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record one stage sample in milliseconds."""
        if not self.enabled:
            return
        try:
            elapsed = float(elapsed_ms)
        except (TypeError, ValueError):
            return
        if not math.isfinite(elapsed):
            return

        sample = {"elapsed_ms": elapsed}
        if metadata:
            sample["metadata"] = dict(metadata)
        self._samples.setdefault(str(stage_name), []).append(sample)

    @contextmanager
    def time(self, stage_name: str, metadata: Optional[dict] = None) -> Iterator[None]:
        """Context manager that records elapsed wall-clock time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage_name, (time.perf_counter() - start) * 1000.0, metadata)

    def summary(self) -> dict:
        """Return raw samples and per-stage summary statistics."""
        stages = {}
        for stage_name, samples in sorted(self._samples.items()):
            values = [
                float(sample["elapsed_ms"])
                for sample in samples[self.discard_first_n :]
                if "elapsed_ms" in sample
            ]
            stages[stage_name] = {
                "samples_ms": list(samples),
                "summary": self._summarize_values(values),
            }

        return {
            "run_name": self.run_name,
            "node_name": self.node_name,
            "package_name": self.package_name,
            "discarded_warmup_count": self.discard_first_n,
            "started_at": self.started_at,
            "ended_at": self.ended_at or datetime.now(timezone.utc).isoformat(),
            "metadata": dict(self._metadata),
            "stages": stages,
        }

    def save(self) -> Optional[Path]:
        """Write the profiling JSON file once."""
        if not self.enabled or self._saved:
            return None
        self.ended_at = datetime.now(timezone.utc).isoformat()
        path = self._write_json()
        self._saved = True
        return path

    def save_checkpoint(self) -> Optional[Path]:
        """Write current profiling JSON without marking the recorder final."""
        if not self.enabled:
            return None
        return self._write_json()

    def _write_json(self) -> Path:
        self.output_path.mkdir(parents=True, exist_ok=True)
        path = self.output_path / f"{self.run_name}.{self.file_tag}.json"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n")
        tmp_path.replace(path)
        return path

    @staticmethod
    def _summarize_values(values: list[float]) -> dict:
        if not values:
            return {
                "n": 0,
                "mean_ms": None,
                "std_ms": None,
                "min_ms": None,
                "max_ms": None,
            }
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {
            "n": len(values),
            "mean_ms": mean,
            "std_ms": math.sqrt(max(0.0, variance)),
            "min_ms": min(values),
            "max_ms": max(values),
        }

    @staticmethod
    def _machine_metadata() -> dict:
        cpu_model = platform.processor() or platform.machine()
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

        ram_gb = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = pages * page_size / (1024**3)
        except (AttributeError, OSError, ValueError):
            pass

        return {
            "hostname": socket.gethostname(),
            "cpu_model": cpu_model,
            "logical_cores": os.cpu_count(),
            "ram_gb": ram_gb,
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "date_time": datetime.now(timezone.utc).isoformat(),
        }

    def _install_signal_handler(self, signum: signal.Signals) -> None:
        previous_handler = signal.getsignal(signum)

        def _handler(received_signum, frame):
            self.save()
            if callable(previous_handler):
                previous_handler(received_signum, frame)
            elif previous_handler == signal.SIG_DFL:
                raise SystemExit(128 + int(received_signum))

        try:
            signal.signal(signum, _handler)
        except (OSError, ValueError):
            pass
