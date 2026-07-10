#!/usr/bin/env python3
"""WorkerPool stress test: feed the bundled RGB-D fixture at fixed or ramping rates.

Runs the full GroundingDINO -> MobileSAM -> OpenCLIP pipeline through the
multi-worker pool without ROS, printing the pool's own diagnostics so input
rate, throughput, latency, drops, and VRAM can be observed under overload.
While running it also samples per-GPU utilization, VRAM, temperature, power,
and PCIe link state into a CSV, and snapshots kernel NVIDIA Xid entries before
and after the run (best effort; requires non-interactive sudo for dmesg).

Safe-by-default: 30 s at 5 Hz. To keep a display GPU completely out of reach,
sandbox with CUDA_VISIBLE_DEVICES, e.g.:

    CUDA_VISIBLE_DEVICES=1 python3 stress_test_worker_pool.py --devices cuda:0

Ramp example (equivalent to rosbag rates 0.1 -> ~1.0 for a 9 Hz bag):

    python3 stress_test_worker_pool.py --ramp 1,2,4,6,8,9 --step-seconds 60
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = next(
    (parent for parent in PACKAGE_ROOT.parents if (parent / "models").is_dir()),
    PACKAGE_ROOT.parents[1],
)
sys.path.insert(0, str(PACKAGE_ROOT))

from semantic_perception.worker import Frame, WorkerPool  # noqa: E402

TELEMETRY_FIELDS = (
    "timestamp,index,name,temperature.gpu,power.draw,power.limit,memory.used,"
    "memory.total,utilization.gpu,utilization.memory,pstate,"
    "pcie.link.gen.current,pcie.link.width.current"
)


class GpuTelemetry:
    """Samples nvidia-smi into a CSV once per second on a background thread."""

    def __init__(self, path: Path, interval: float = 1.0):
        self._path = path
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-telemetry")
        self.available = True

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _sample(self) -> list[str]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={TELEMETRY_FIELDS}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def _run(self) -> None:
        try:
            with open(self._path, "w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(TELEMETRY_FIELDS.split(","))
                while not self._stop.is_set():
                    try:
                        for line in self._sample():
                            writer.writerow([cell.strip() for cell in line.split(",")])
                        stream.flush()
                    except Exception as exc:
                        writer.writerow([f"# telemetry error: {exc}"])
                        stream.flush()
                    self._stop.wait(self._interval)
        except OSError:
            self.available = False


def snapshot_xids(label: str) -> list[str]:
    """Best-effort host kernel Xid/PCIe error listing (needs sudo -n dmesg)."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "dmesg", "-T"], capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            print(f"[warn] {label}: cannot read kernel log ({result.stderr.strip()})")
            return []
        lines = [
            line for line in result.stdout.splitlines()
            if any(token in line for token in ("Xid", "fallen off", "AER:", "NVRM"))
        ]
        print(f"[info] {label}: {len(lines)} kernel NVRM/Xid/AER lines")
        for line in lines[-5:]:
            print(f"[kernel] {line}")
        return lines
    except Exception as exc:
        print(f"[warn] {label}: kernel log unavailable: {exc}")
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--workers", type=int, default=0, help="0 = one per device")
    parser.add_argument("--hz", type=float, default=5.0, help="fixed frame submission rate")
    parser.add_argument("--seconds", type=float, default=30.0, help="duration at a fixed rate")
    parser.add_argument(
        "--ramp", type=str, default="",
        help="comma-separated submission rates in Hz, e.g. '1,2,4,6,8'; overrides --hz",
    )
    parser.add_argument(
        "--step-seconds", type=float, default=60.0, help="duration of each ramp step"
    )
    parser.add_argument("--budget-gb", type=float, default=22.0)
    parser.add_argument("--frame-queue-size", type=int, default=4)
    parser.add_argument("--models-dir", type=Path, default=WORKSPACE_ROOT / "models")
    parser.add_argument(
        "--telemetry-csv", type=Path, default=Path("gpu_telemetry.csv"),
        help="per-second GPU utilization/VRAM/temperature/power/PCIe samples",
    )
    args = parser.parse_args()

    if args.ramp:
        steps = [(float(rate), args.step_seconds) for rate in args.ramp.split(",")]
    else:
        steps = [(args.hz, args.seconds)]
    total_seconds = sum(duration for _, duration in steps)

    fixture = PACKAGE_ROOT / "test" / "Dd4bFSTQ8gi_000018"
    rgb = np.asarray(Image.open(f"{fixture}_rgb.png").convert("RGB"), dtype=np.uint8)
    depth = np.asarray(Image.open(f"{fixture}_depth.png")).astype(np.float32) * 0.001
    intrinsics = (540.0, 540.0, 539.5, 359.5)
    models = args.models_dir

    config = {
        "device": "cuda",
        "devices": args.devices,
        "num_worker_threads": args.workers,
        "frame_queue_size": args.frame_queue_size,
        "drop_report_every": 30,
        "result_queue_size": 8,
        "drop_stale_results": True,
        "gpu_memory_budget_gb": args.budget_gb,
        "openclip_model": "ViT-H-14",
        "openclip_checkpoint_path": str(models / "laion2b_s32b_b79k.bin"),
        "text_embedding_batch_size": 64,
        "groundingdino_config_path": str(models / "groundingdino/GroundingDINO_SwinT_OGC.py"),
        "groundingdino_model": str(models / "groundingdino/groundingdino_swint_ogc.pth"),
        "sam_model": str(models / "mobilesam/mobile_sam.pt"),
        "sam_model_type": "vit_t",
        "prompt_csv_path": str(models / "labels/HM3D_CountsOfObjectTypes.csv"),
        "class_embedding_cache_path": str(models / "hm3d_openclip_embedding_cache.bin"),
        "groundingdino_prompt": "object",
        "detection_threshold": 0.35,
        "text_threshold": 0.25,
        "bbox_embedding_weight": 0.5,
        "masked_embedding_weight": 0.5,
        "min_valid_depth_points": 20,
        "max_depth_m": 10.0,
    }

    print(f"[info] pid={os.getpid()} devices={args.devices} "
          f"steps={[(hz, s) for hz, s in steps]} total={total_seconds:.0f} s "
          "(Ctrl+C stops cleanly)")
    xids_before = snapshot_xids("kernel log before test")

    load_start = time.monotonic()
    pool = WorkerPool(
        config,
        lambda m: print(f"[warn] {m}"),
        lambda m: print(f"[info] {m}"),
        lambda m: print(f"[error] {m}"),
    )
    print(f"[info] pool ready in {time.monotonic() - load_start:.1f} s")

    sequence = 0
    delivered = 0
    proposals_seen = 0
    monotonic_ok = True
    last_seq = -1
    aborted = ""

    def drain() -> None:
        nonlocal delivered, proposals_seen, monotonic_ok, last_seq
        while True:
            result = pool.get_result()
            if result is None:
                return
            if result.frame.sequence <= last_seq:
                monotonic_ok = False
            last_seq = result.frame.sequence
            delivered += 1
            proposals_seen += len(result.proposals)

    try:
        with GpuTelemetry(args.telemetry_csv):
            next_report = time.monotonic() + 10.0
            for rate, duration in steps:
                if aborted:
                    break
                period = 1.0 / rate
                print(f"[step] submitting at {rate:g} Hz for {duration:g} s")
                deadline = time.monotonic() + duration
                next_submit = time.monotonic()
                while time.monotonic() < deadline:
                    now = time.monotonic()
                    if now >= next_submit:
                        pool.submit(
                            Frame(sequence, 0, 0, "camera", rgb, depth, intrinsics,
                                  received_monotonic=now)
                        )
                        sequence += 1
                        next_submit += period
                    drain()
                    if pool.all_workers_failed():
                        aborted = "all workers failed"
                        print(f"[FATAL] {aborted}")
                        break
                    if now >= next_report:
                        info, warning = pool.stats_report()
                        if info:
                            print(f"[stats] {info}")
                        if warning:
                            print(f"[stats-warn] {warning}")
                        next_report += 10.0
                    time.sleep(0.002)
            drain_deadline = time.monotonic() + 20.0
            while not aborted and delivered < sequence and time.monotonic() < drain_deadline:
                drain()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("[info] interrupted; shutting down cleanly")
    finally:
        drain()
        info, _ = pool.stats_report()
        if info:
            print(f"[final-stats] {info}")
        close_start = time.monotonic()
        pool.close()
        print(f"[info] pool closed in {time.monotonic() - close_start:.2f} s")

    xids_after = snapshot_xids("kernel log after test")
    new_xids = [line for line in xids_after if line not in set(xids_before)]
    if new_xids:
        print(f"[FATAL] {len(new_xids)} NEW kernel NVRM/Xid/AER entries during the test:")
        for line in new_xids:
            print(f"[kernel] {line}")
        print("[FATAL] This is a driver/hardware-level failure; if a GPU is gone "
              "from host nvidia-smi, reboot the host before retesting.")

    print(
        f"[RESULT] submitted={sequence} delivered={delivered} "
        f"proposals={proposals_seen} monotonic={monotonic_ok} "
        f"aborted='{aborted}' new_kernel_errors={len(new_xids)} "
        f"throughput={delivered / max(total_seconds, 1e-6):.2f} Hz "
        f"telemetry={args.telemetry_csv}"
    )
    if aborted or new_xids:
        sys.exit(1)


if __name__ == "__main__":
    main()
