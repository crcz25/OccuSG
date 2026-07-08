#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
VENV_PATH="${VENV_PATH:-${HOME}/venv}"
SKIP_ROSDEP="${SKIP_ROSDEP:-0}"

log() {
  printf '[build_workspace] %s\n' "$*"
}

fail() {
  printf '[build_workspace] ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf '[build_workspace] ERROR: command failed at line %s (exit %s)\n' \
    "${BASH_LINENO[0]}" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

[[ -f "${ROS_SETUP}" ]] || fail "ROS setup not found: ${ROS_SETUP}"
[[ -d "${WORKSPACE_ROOT}/src" ]] || fail "Workspace src directory is missing"

# ROS setup files are not guaranteed to be nounset-safe.
set +u
source "${ROS_SETUP}"
set -u

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  log "creating semantic inference environment at ${VENV_PATH}"
  "${WORKSPACE_ROOT}/src/semantic_perception/scripts/create_inference_env.sh" \
    "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" - <<'PY'
import rclpy
import torch

print(f"[build_workspace] Python: {__import__('sys').executable}")
print(f"[build_workspace] PyTorch: {torch.__version__}")
PY

case "${SKIP_ROSDEP}" in
  0|false|FALSE|off|OFF)
    command -v rosdep >/dev/null 2>&1 || fail "rosdep is not installed"
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
      log "initializing rosdep"
      if (( EUID == 0 )); then
        rosdep init
      else
        command -v sudo >/dev/null 2>&1 || fail "sudo is required to initialize rosdep"
        sudo rosdep init
      fi
    fi
    if [[ ! -d "${HOME}/.ros/rosdep/sources.cache" ]]; then
      log "updating rosdep index"
      rosdep update --include-eol-distros
    fi
    log "installing declared system/ROS dependencies"
    rosdep install --from-paths "${WORKSPACE_ROOT}/src" --ignore-src -r -y
    ;;
  1|true|TRUE|on|ON)
    log "skipping rosdep because SKIP_ROSDEP=${SKIP_ROSDEP}"
    ;;
  *)
    fail "SKIP_ROSDEP must be 0/1, true/false, or on/off"
    ;;
esac

log "checking Python dependency consistency"
"${VENV_PATH}/bin/python" -m pip check

log "building workspace at ${WORKSPACE_ROOT}"
cd "${WORKSPACE_ROOT}"
"${VENV_PATH}/bin/python" -m colcon build \
  --symlink-install \
  --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "$@"

log "build complete"
log "source ${WORKSPACE_ROOT}/install/setup.bash before running ROS commands"
