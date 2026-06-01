#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden via env)
: "${USERNAME:=devuser}"
: "${WORKSPACE_DIR:=/workspace}"
: "${PROJECT_WS:=${WORKSPACE_DIR}/occusg_ws}"

log() { echo "[entrypoint] $*"; }

source_setup() {
  local setup_file="$1"
  if [[ -f "$setup_file" ]]; then
    set +u
    source "$setup_file"
    set -u
  fi
}

ensure_apt_metadata() {
  if command -v apt-get >/dev/null 2>&1 \
      && [[ -z "$(find /var/lib/apt/lists -type f -name '*Packages*' -print -quit 2>/dev/null)" ]]; then
    log "updating apt package metadata for rosdep"
    sudo apt-get update
  fi
}

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"
export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-dummy}"
export GAZEBO_AUDIO_DEVICE="${GAZEBO_AUDIO_DEVICE:-none}"

TB3_SETUP="${WORKSPACE_DIR}/turtlebot3_ws/install/setup.bash"
ROS_SETUP_CMD='source "/opt/ros/${ROS_DISTRO}/setup.bash"'
if [[ -f "$TB3_SETUP" ]]; then
  # Prefer the built TurtleBot3 workspace so ROS resolves the overridden assets.
  source_setup "$TB3_SETUP"
  ROS_SETUP_CMD="${ROS_SETUP_CMD} && source '${TB3_SETUP}'"
fi

if [[ -d "${PROJECT_WS}/src" ]]; then
  log "installing ROS dependencies from ${PROJECT_WS}/src"
  ensure_apt_metadata
  if ! PROJECT_WS="${PROJECT_WS}" bash -lc "${ROS_SETUP_CMD} && cd \"\${PROJECT_WS}\" && rosdep install --from-paths src --ignore-src -r -y"; then
    log "WARN: rosdep failed; continuing so the container stays available"
  fi

  log "building workspace ${PROJECT_WS} with verbose compiler output"
  if ! PROJECT_WS="${PROJECT_WS}" bash -lc "${ROS_SETUP_CMD} && cd \"\${PROJECT_WS}\" && colcon build --symlink-install --event-handlers console_direct+ --cmake-args -DONNXRUNTIME_USE_GPU=${USE_GPU:-ON} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_VERBOSE_MAKEFILE=ON"; then
    log "ERROR: workspace build failed; continuing so the container stays available for debugging"
  fi

  PROJECT_SETUP="${PROJECT_WS}/install/setup.bash"
  if [[ -f "$PROJECT_SETUP" ]]; then
    source_setup "$PROJECT_SETUP"
  fi
else
  log "WARN: missing workspace source directory: ${PROJECT_WS}/src"
fi

log "done. exec: $*"
exec "$@"
