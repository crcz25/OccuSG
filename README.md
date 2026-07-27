# OccuSG

OccuSG is a ROS 2 Humble workspace for occupancy-grounded room segmentation and
hierarchical 3D scene graphs. The intended pipeline combines RGB-D perception,
point-cloud generation, occupancy mapping, room decomposition, and scene-graph
construction.

![OccuSG pipeline overview](img/pipeline.svg)

## Paper and citation

```bibtex
@misc{occusg2026,
      title={Occupancy-Grounded Room Segmentation for Hierarchical 3D Scene Graphs},
      author={Carlos Cueto Zumaya and Iacopo Catalano and Jorge Peña-Queralta and Wallace Moreira Bessa},
      year={2026},
      eprint={2606.13727},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.13727},
}
```

## Current workspace contents

`colcon list` currently discovers these seven packages:

| Package | Build type | Role |
|---|---|---|
| `incremental_dude_msgs` | `ament_cmake` | Region interface messages. |
| `incremental_dude_ros2` | `ament_cmake` | Incremental DuDe room decomposition. |
| `point_cloud_generator` | `ament_python` | Synchronized depth-to-point-cloud projection. |
| `scene_graph_core` | `ament_python` | Scene-graph data structures and algorithms. |
| `scene_graph_ros` | `ament_python` | ROS orchestration, visualization, export, and evaluation utilities. |
| `semantic_perception_msgs` | `ament_cmake` | RGB-D object-proposal interfaces. |
| `semantic_perception` | `ament_python` | GroundingDINO, MobileSAM, and OpenCLIP inference. |

The checked-out `src/mapconversion/` and `src/octomap_mapping/` directories are
empty and are not Colcon packages. The two `scene_graph_ros` pipeline launch
files still reference `mapconversion` and the former ONNX/YOLO semantic node.
They are therefore not supported end-to-end entry points in this checkout. The
standalone semantic pipeline and `semantic_perception.launch.py` are current.

## Dependency model

Use Ubuntu 22.04, Python 3.10, and ROS 2 Humble. Dependencies are intentionally
split between the operating system and one Python virtual environment.

### System and ROS dependencies

Install ROS, compiler, interface, and native C++ dependencies with apt/rosdep.
Do not install ROS packages such as `rclpy` or `cv_bridge` from pip.

Required system components include:

- ROS 2 Humble desktop, `ros-dev-tools`, Colcon, and rosdep;
- a C++17 compiler, CMake, Git, and Python's venv support;
- OpenCV development libraries for `incremental_dude_ros2`;
- CGAL, GMP, and MPFR for the DuDe implementation;
- the ROS packages declared in each `package.xml`.

After configuring the official ROS 2 apt repository, a typical base install is:

```bash
sudo apt-get update
sudo apt-get install -y \
  ros-humble-desktop \
  ros-dev-tools \
  build-essential \
  cmake \
  git \
  python3-venv \
  libcgal-dev \
  libgmp-dev \
  libmpfr-dev \
  libopencv-dev
```

The build script initializes rosdep when necessary and installs dependencies
declared by the package manifests. To do this manually:

```bash
source /opt/ros/humble/setup.bash
sudo rosdep init       # once per machine; skip if already initialized
rosdep update --include-eol-distros
rosdep install --from-paths src --ignore-src -r -y
```

### Python inference environment

Torch, torchvision, OpenCLIP, GroundingDINO, MobileSAM, Transformers, and their
Python image-processing dependencies are pinned in
[`src/semantic_perception/requirements.txt`](src/semantic_perception/requirements.txt).
They must be installed in the venv, not globally.

The default environment location is `$HOME/venv`, matching the devcontainer:

```bash
cd /path/to/occusg_ws
src/semantic_perception/scripts/create_inference_env.sh
source "$HOME/venv/bin/activate"
```

To use another location, set `VENV_PATH` consistently:

```bash
VENV_PATH="$PWD/.venv" \
  src/semantic_perception/scripts/create_inference_env.sh
source "$PWD/.venv/bin/activate"
```

The venv is created with `--system-site-packages`. This is required so the venv
interpreter can import apt-installed ROS modules while keeping the large
inference stack isolated.

## Build the complete workspace

### One-command build

From a clean terminal, after the system/ROS prerequisites above are installed:

```bash
cd /path/to/occusg_ws
./scripts/build_workspace.sh
```

The script:

1. sources `/opt/ros/${ROS_DISTRO:-humble}/setup.bash`;
2. creates `${VENV_PATH:-$HOME/venv}` if it does not exist;
3. verifies that the venv imports both `rclpy` and Torch;
4. installs declared apt/ROS dependencies through rosdep;
5. runs `pip check`;
6. builds every discovered package with the venv's Python.

After it completes, source the workspace in the terminal where ROS commands will
run:

```bash
source install/setup.bash
```

Useful environment overrides are:

```bash
# Use an existing environment at a custom location.
VENV_PATH="$PWD/.venv" ./scripts/build_workspace.sh

# Skip rosdep on repeated/offline builds after system dependencies are installed.
SKIP_ROSDEP=1 ./scripts/build_workspace.sh

# Forward additional arguments to `colcon build`.
SKIP_ROSDEP=1 ./scripts/build_workspace.sh --executor sequential
```

### Equivalent manual build

The critical detail is invoking Colcon through the venv interpreter. Activating
the venv and then running `/usr/bin/colcon` is not equivalent: it can generate
ROS Python entry points with a `#!/usr/bin/python3` shebang, which cannot import
Torch.

```bash
cd /path/to/occusg_ws
source /opt/ros/humble/setup.bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"

rosdep install --from-paths src --ignore-src -r -y
python -m pip check
python -m colcon build \
  --symlink-install \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

source install/setup.bash
```

## Build only `semantic_perception`

On a fresh workspace, build the message dependency and package together:

```bash
cd /path/to/occusg_ws
source /opt/ros/humble/setup.bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"

python -m colcon build \
  --symlink-install \
  --packages-up-to semantic_perception
source install/setup.bash
```

For subsequent code-only rebuilds after `semantic_perception_msgs` is installed:

```bash
python -m colcon build \
  --symlink-install \
  --packages-select semantic_perception
source install/setup.bash
```

Verify that the generated executable uses the venv:

```bash
head -1 install/semantic_perception/lib/semantic_perception/semantic_perception_node
```

The first line must point to `${VENV_PATH:-$HOME/venv}/bin/python`. If an older
build points to `/usr/bin/python3`, remove only
`build/semantic_perception` and `install/semantic_perception`, then rebuild with
`python -m colcon` as shown above.

## Docker/devcontainer

The development image is based on
`nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`. It installs ROS Humble, native
build dependencies, and `/home/devuser/venv` from the same pinned semantic
requirements. The repository is mounted at `/workspace/occusg_ws`.

```bash
docker compose -f .devcontainer/docker-compose-humble.yml up -d --build
docker compose -f .devcontainer/docker-compose-humble.yml exec dev bash
```

The container entrypoint runs the same venv-aware workspace build script. The
compose file reserves NVIDIA GPUs, so Docker GPU use requires a compatible host
driver and the NVIDIA Container Toolkit. The image uses PyTorch 2.7.1 with the
CUDA 12.8 wheel, covering RTX 3090/Ampere (`sm_86`) and RTX 50-series/Blackwell
(`sm_120`) GPUs.

## ONNX Runtime and GPU selection

The current source tree has no ONNX Runtime consumer and no CMake target reads
`ONNXRUNTIME_DIR` or `ONNXRUNTIME_USE_GPU`. Those dependencies and build flags
were removed from the documented and container build process.

The `model_file`, `class_file`, and `use_gpu` keys that remain in
`scene_graph_ros/config/scene_graph_pipeline_params*.yaml` belong to the former
YOLO/ONNX node. The current `semantic_perception` node does not declare or read
them. Do not pass `use_gpu` to the current node.

Current device selection is:

- ROS node: `device` and `devices` parameters in
  `src/semantic_perception/config/semantic_perception.yaml`;
- standalone runner: `--device` and `--openclip-device`;
- Docker GPU exposure: NVIDIA Container Toolkit and Compose device reservation.

## Current launch and runtime entry points

The supported semantic launch command has one launch argument, `config`:

```bash
source install/setup.bash
ros2 launch semantic_perception semantic_perception.launch.py \
  config:=/absolute/path/to/semantic_perception.yaml
```

The model files are runtime assets and are not required to compile the workspace.
See [`src/semantic_perception/README.md`](src/semantic_perception/README.md) for
model names, standalone validation, ROS parameters, rosbag testing, debug images,
and CPU/GPU guidance.

The launch arguments declared in
`scene_graph_pipeline_mp3d_bag.launch.py` and
`scene_graph_pipeline.tbot3.launch.py` still exist syntactically, but those
launches are not valid end-to-end entry points in the current checkout for the
reasons described above.

## Tests

Run the standalone semantic tests inside the inference venv:

```bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"
PYTHONPATH=src/semantic_perception \
  python -m pytest -q src/semantic_perception/test
```

Run Colcon tests for the built workspace with:

```bash
source /opt/ros/humble/setup.bash
source "${VENV_PATH:-$HOME/venv}/bin/activate"
source install/setup.bash
python -m colcon test
python -m colcon test-result --verbose
```

Evaluation and profiling utilities remain under `src/scene_graph_ros/scripts/`.
Use each script's `--help` output for its current standalone command-line API.
