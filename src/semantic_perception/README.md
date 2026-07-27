# `semantic_perception`

`semantic_perception` provides a standalone and ROS 2 RGB-D inference pipeline:

1. GroundingDINO generates class-agnostic object boxes.
2. MobileSAM (`vit_t`) generates a mask for each box.
3. OpenCLIP encodes bounding-box and masked crops.
4. The fused visual embedding is compared with a cached prompt vocabulary for
   debug labeling.
5. Valid aligned depth is projected into 3D with the supplied camera intrinsics.

The standalone runner can additionally transform camera-frame geometry with a
4x4 camera-to-world pose. The ROS node currently publishes camera-frame geometry
because it does not subscribe to odometry or TF.

## Dependencies: system versus venv

The supported platform is Ubuntu 22.04, ROS 2 Humble, and Python 3.10.

Install these through apt/rosdep:

- ROS packages: `rclpy`, `ament_index_python`, `sensor_msgs`, `vision_msgs`,
  `geometry_msgs`, `std_msgs`, `message_filters`, and `cv_bridge`;
- ROS build tools: `ament_python`, Colcon, and rosdep;
- native OpenCV and Python venv support;
- Git and a compiler toolchain, which GroundingDINO needs while packaging.

Install these only in the Python venv from `requirements.txt`:

- PyTorch 2.7.1 and torchvision 0.22.1 CUDA 12.8 wheels, with RTX 3090/Ampere
  (`sm_86`) and RTX 50-series/Blackwell (`sm_120`) support;
- OpenCLIP 2.32.0;
- pinned GroundingDINO and MobileSAM Git revisions;
- Transformers, timm, NumPy, SciPy, Pillow, OpenCV wheels, supervision, and
  supporting packages.

Do not attempt to install `rclpy` or `cv_bridge` from pip. The venv uses
`--system-site-packages` so it can import the ROS modules installed under
`/opt/ros/humble` while isolating the inference stack.

## Create the environment

The standard location is `$HOME/venv`, also used by the devcontainer:

```bash
cd /path/to/occusg_ws
src/semantic_perception/scripts/create_inference_env.sh
source "$HOME/venv/bin/activate"
python -m pip check
```

Choose another location by setting `VENV_PATH` consistently:

```bash
VENV_PATH="$PWD/.venv" \
  src/semantic_perception/scripts/create_inference_env.sh
source "$PWD/.venv/bin/activate"
```

The environment script installs CUDA-enabled PyTorch wheels. They run in CPU
mode on hosts without an available NVIDIA GPU. GPU execution requires a driver
compatible with CUDA 12.x and the NVIDIA Container Toolkit when running in
Docker.

## Build the package

Always run Colcon through the venv's Python. Merely activating the venv and then
executing `/usr/bin/colcon` can generate `semantic_perception_node` with a system
Python shebang that cannot import Torch.

For the first build, include `semantic_perception_msgs` with
`--packages-up-to`:

```bash
cd /path/to/occusg_ws
source /opt/ros/humble/setup.bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"

python -m colcon build \
  --symlink-install \
  --packages-up-to semantic_perception
source install/setup.bash
```

For later rebuilds when the message package is already installed:

```bash
python -m colcon build \
  --symlink-install \
  --packages-select semantic_perception
source install/setup.bash
```

Verify the interpreter embedded in the ROS executable:

```bash
head -1 install/semantic_perception/lib/semantic_perception/semantic_perception_node
```

It must refer to the venv's Python. If it says `#!/usr/bin/python3`, remove
`build/semantic_perception` and `install/semantic_perception`, then rebuild using
`python -m colcon`.

The workspace-level [`scripts/build_workspace.sh`](../../scripts/build_workspace.sh)
performs the same venv-aware build for all packages.

## Runtime assets

The default configuration expects these files under the workspace `models/`
directory:

```text
models/
  groundingdino/
    GroundingDINO_SwinT_OGC.py
    groundingdino_swint_ogc.pth
  mobilesam/
    mobile_sam.pt
  labels/
    HM3D_CountsOfObjectTypes.csv
  laion2b_s32b_b79k.bin
```

`hm3d_openclip_embedding_cache.bin` is generated automatically in the same
directory. Model files are not required to compile the package, but all five
assets above are required by the default ROS configuration at runtime.

## Standalone validation

Run unit and bundled-fixture tests:

```bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"
PYTHONPATH=src/semantic_perception \
  python -m pytest -q src/semantic_perception/test
```

Run all three models against the synchronized RGB, depth, and pose fixture:

```bash
PYTHONPATH=src/semantic_perception \
  python -m semantic_perception.standalone \
  --output /tmp/semantic_perception_sample.json
```

The smoke command fails unless at least one proposal has a MobileSAM mask,
OpenCLIP embedding, and valid world-frame geometry. The fixture has no separate
calibration file, so the command defaults to a 90-degree horizontal field of
view. Override this with `--horizontal-fov` when necessary.

The fixture pose follows Habitat/OpenGL axes. The default
`--pose-convention habitat` converts it to an optical camera frame. Use
`--pose-convention optical` when the input pose already uses +X right, +Y down,
and +Z forward.

The supplied ViT-H OpenCLIP model stays on CPU by default in the standalone
command so the full model set fits on common 4 GB GPUs. Systems with enough VRAM
can pass `--openclip-device cuda:0`. For CPU-only execution, pass
`--device cpu --openclip-device cpu`.

## ROS node and launch file

The supported package launch file exposes one launch argument, `config`:

```bash
source /opt/ros/humble/setup.bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"
source install/setup.bash

ros2 launch semantic_perception semantic_perception.launch.py \
  config:="$PWD/src/semantic_perception/config/semantic_perception.yaml"
```

The executable is named `semantic_perception_node`. The former executable name
`semantic_perception`, ONNX model parameters, and `use_gpu` parameter are not
part of this package.

### Important ROS parameters

| Parameter | Default | Purpose |
|---|---|---|
| `rgb_topic` | `/camera/color/image_raw` | RGB image input. |
| `depth_topic` | `/camera/depth/image_raw` | Aligned depth input (`16UC1` or `32FC1`). |
| `camera_info_topic` | `/camera/color/camera_info` | Intrinsics matching the aligned RGB-D images. |
| `proposal_topic` | `/semantic_perception/object_proposals` | `ObjectProposal3DArray` output. |
| `publish_debug_image` | `true` in the checked-in YAML | Enable annotated image rendering. |
| `debug_image_topic` | `/semantic_perception/debug_image` | Reliable raw debug-image output. |
| `debug_mask_alpha` | `0.45` | MobileSAM mask opacity from 0 to 1. |
| `openclip_model` | `ViT-H-14` | OpenCLIP architecture. |
| `openclip_checkpoint_path` | `models/laion2b_s32b_b79k.bin` | OpenCLIP checkpoint. |
| `text_embedding_batch_size` | `64` | Prompt-cache generation batch size. |
| `groundingdino_config_path` | `models/groundingdino/GroundingDINO_SwinT_OGC.py` | GroundingDINO configuration. |
| `groundingdino_model` | `models/groundingdino/groundingdino_swint_ogc.pth` | GroundingDINO weights. |
| `groundingdino_prompt` | `object` | Class-agnostic detector prompt. |
| `sam_model` | `models/mobilesam/mobile_sam.pt` | MobileSAM weights. |
| `sam_model_type` | `vit_t` | MobileSAM registry key. |
| `devices` | `['cuda:0', 'cuda:1']` | Worker-device assignment (worker `i` uses `devices[i % len]`). |
| `device` | `cuda` | Fallback when `devices` is empty. |
| `num_worker_threads` | `2` | Independently loaded model bundles; `0` = one per device (recommended). |
| `frame_queue_size` | `4` | Shared bounded input queue; overflow drops the oldest frame, `1` = latest-frame-only. |
| `result_queue_size` | `8` | Bounded result queue. |
| `drop_stale_results` | `true` | Publish results in monotonic frame order, dropping late out-of-order ones. |
| `gpu_memory_budget_gb` | `0.0` (`22.0` in the checked-in YAML) | Per-GPU memory cap for this process; `0` = unlimited. Clamped to currently-free VRAM minus 1 GiB when the GPU is shared (e.g. with a display server). CUDA only; other backends warn and continue. |
| `stats_report_interval_sec` | `10.0` | Runtime diagnostics period (rates, queue depth, stage timings, drops, VRAM); `0` disables. |
| `detection_threshold` | `0.35` | GroundingDINO box threshold. |
| `text_threshold` | `0.25` | GroundingDINO text threshold. |
| `bbox_embedding_weight` | `0.5` | Fused embedding bbox weight. |
| `masked_embedding_weight` | `0.5` | Fused embedding mask weight. |
| `sync_queue_size` | `10` | Approximate-time synchronizer queue. |
| `sync_slop_seconds` | `0.1` | Maximum synchronization offset. |
| `min_valid_depth_points` | `20` | Minimum points for valid 3D geometry. |
| `max_depth_m` | `10.0` | Maximum accepted depth. |

`devices` takes precedence when it is nonempty. On a one-GPU machine, set
`devices: ["cuda:0"]` and `num_worker_threads: 1`. On a 4 GB GPU, the complete
ViT-H/DINO/SAM bundle generally does not fit; use `devices: ["cpu"]`,
`device: "cpu"`, and one worker. There is no `use_gpu` flag.

## Test with the included Matterport-style rosbag

The inspected `2t7WUuJeko7` bag contains:

- `/rgb`: 640x480 `rgb8`;
- `/depth`: 640x480 `32FC1`;
- `/rgb/camera_info`: matching intrinsics.

Start the node before playback. The following command uses CPU and one worker,
which is suitable for a 4 GB GPU host:

```bash
ros2 run semantic_perception semantic_perception_node --ros-args \
  --params-file src/semantic_perception/config/semantic_perception.yaml \
  -p device:=cpu \
  -p devices:="['cpu']" \
  -p num_worker_threads:=1 \
  -p frame_queue_size:=1 \
  -r /camera/color/image_raw:=/rgb \
  -r /camera/depth/image_raw:=/depth \
  -r /camera/color/camera_info:=/rgb/camera_info
```

Wait for `Inference models ready`, then play only the required topics slowly:

```bash
ros2 bag play \
  bags/2t7WUuJeko7/2t7WUuJeko7_0.db3 \
  --storage sqlite3 \
  --rate 0.01 \
  --topics /rgb /depth /rgb/camera_info
```

Topic filtering avoids malformed legacy `/rosout` QoS metadata in this specific
bag. CPU inference is slow, so bounded-queue frame-drop warnings are expected.

Inspect proposals with:

```bash
ros2 topic hz /semantic_perception/object_proposals
ros2 topic echo \
  /semantic_perception/object_proposals \
  semantic_perception_msgs/msg/ObjectProposal3DArray \
  --once --no-arr
```

## Debug image

With `publish_debug_image: true`, the node publishes:

- GroundingDINO boxes and confidence values;
- MobileSAM mask overlays;
- the nearest cached OpenCLIP class label.

The class label is for visualization only. `ObjectProposal3D.class_name` remains
unset because semantic classification is reserved for a later module.

View the reliable debug stream with:

```bash
ros2 run image_view image_view --ros-args \
  -r image:=/semantic_perception/debug_image
```

Set `publish_debug_image: false` to eliminate rendering and image-publication
overhead.

## Prompt embedding cache

When the prompt CSV, OpenCLIP model identity, or cache changes, the node rebuilds
the binary text-embedding cache. A progress bar reports prompt completion:

```text
OpenCLIP text embeddings: 42%|...| 320/761 [01:24<01:57, 3.76prompt/s]
```

Lower `text_embedding_batch_size` if cache generation uses too much RAM or VRAM.
Subsequent runs reuse a matching cache.

## Docker

The devcontainer creates `/home/devuser/venv` from this package's pinned
requirements and places it on `PATH`. The workspace entrypoint invokes the root
venv-aware build script. Inside the container, use the same build, test, and
runtime commands shown above.

The devcontainer uses the CUDA *devel* image, but intentionally does not compile
GroundingDINO's custom `_C` deformable-attention extension. The pinned upstream
extension uses a pre-2.6 PyTorch C++ API; the adapter therefore selects
GroundingDINO's portable PyTorch deformable-attention implementation, which is
compatible with the newer CUDA wheel and does not depend on a GPU during the
image build.

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'` from `ros2 run`

Check the executable's first line:

```bash
head -1 install/semantic_perception/lib/semantic_perception/semantic_perception_node
```

If it points to `/usr/bin/python3`, rebuild with the venv interpreter as described
above. Activating the venv after an incorrect build cannot change an existing
script's shebang.

### Upstream warnings

Current pinned dependencies may emit deprecation warnings from Transformers,
timm, Torch checkpointing, or MobileSAM's timm registry. They are non-fatal when
the node reaches `Inference models ready` and begins publishing results.

### ONNX and `use_gpu`

This package does not use ONNX Runtime, TensorRT, `ONNXRUNTIME_*` CMake flags, or
a `use_gpu` ROS parameter. Use `device`/`devices` for the ROS node and
`--device`/`--openclip-device` for the standalone runner.

### `CUDA error: unspecified launch failure` / a GPU disappears from `nvidia-smi`

This error means the CUDA context is dead; it is frequently the *symptom* of a
driver- or hardware-level event, not an application bug. Diagnose in this order:

1. **Capture the system state immediately** (before and after a test):

   ```bash
   ./src/semantic_perception/scripts/collect_gpu_diagnostics.sh
   ```

2. **Check the host kernel log for the FIRST failure.** Application logs show
   whichever CUDA call happened to observe the dead context first (CLIP,
   GroundingDINO, ...); the kernel log shows the actual origin:

   ```bash
   sudo dmesg -T | grep -iE "NVRM|Xid|fallen off|AER"
   ```

   * `Xid 79: GPU has fallen off the bus` — the GPU dropped off PCIe. This is a
     power/PCIe/hardware event. **Only a host reboot recovers it**; restarting
     the node, recreating the CUDA context, or rebuilding the container cannot.
   * `Xid 154: recovery action ... Node Reboot Required` — the driver has
     poisoned new CUDA initialization process-wide; reboot the host.
   * `Xid 13/31/43` with a process name — an application-level illegal access;
     debug with `CUDA_LAUNCH_BLOCKING=1` and per-stage logs.

3. **Node behaviour on errors.** The node applies no custom CUDA error
   handling: an exception raised anywhere in a worker's pipeline (detect,
   segment, embed, geometry) is logged in full — original exception, message,
   and traceback — via `logger.error(...)`, and that worker's thread then
   terminates; nothing classifies the error, retries it, or substitutes a
   generic message. Each worker owns its models on its own GPU, so the
   remaining workers keep processing independently. When every worker thread
   has stopped, the node shuts down (parameter
   `shutdown_when_all_workers_failed`) so a supervisor can restart it.

### Dual-GPU power stability (RTX 3090)

Two RTX 3090s draw 350 W each sustained and transient-spike far above that for
milliseconds. On marginal PSUs, daisy-chained PCIe power pigtails, or riser
cables this manifests as Xid 79 under rising inference load. Mitigations, in
order of effectiveness:

```bash
# Cap sustained board power (per boot; put in a systemd unit for persistence).
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 280        # 3090 perf loss is small; transient spikes shrink a lot
```

* Use one dedicated PCIe 8-pin cable per connector (no Y-splitters/pigtails).
* Prefer direct slot mounting over riser cables; reseat card and power cables.
* Size the PSU for >= 2x the combined sustained GPU draw (>= 1200 W for two
  3090s plus CPU).
* Re-validate with the stress test below while watching kernel logs.

### Stress testing

`scripts/stress_test_worker_pool.py` drives the full pipeline without ROS,
records per-second GPU telemetry (utilization, VRAM, temperature, power, PCIe
link) to CSV, snapshots kernel Xid entries before/after, and aborts as soon as
all workers fail:

```bash
# Ramp the submission rate: ~rosbag rates 0.1 -> 1.0 for a 9 Hz bag.
python3 src/semantic_perception/scripts/stress_test_worker_pool.py \
    --ramp 1,2,4,6,8,9 --step-seconds 60 --telemetry-csv /tmp/gpu_telemetry.csv
```

Isolation runs for narrowing down a failing component:

```bash
# One GPU at a time
python3 ... --devices cuda:0
python3 ... --devices cuda:1
# Serialize all kernel launches to expose the first failing operation
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 PYTHONFAULTHANDLER=1 python3 ...
```

`CUDA_LAUNCH_BLOCKING=1` is for debugging only — do not leave it enabled in
production. Device-side assertions (`TORCH_USE_CUDA_DSA`) are a *compile-time*
option of PyTorch: setting the environment variable at runtime on a release
wheel does nothing; you would need a source build of PyTorch with
`TORCH_USE_CUDA_DSA=1` to get device-side assertion messages.
