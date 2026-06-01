FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS occusg

ARG USERNAME="devuser"
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKSPACE_DIR="/workspace"
ARG ROS_DISTRO="humble"
ARG USE_GPU="ON"

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    USERNAME=${USERNAME} \
    WORKSPACE_DIR=${WORKSPACE_DIR} \
    ROS_DISTRO=${ROS_DISTRO} \
    USE_GPU=${USE_GPU} \
    VENV_PATH=/opt/venv \
    USE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/cmake/pcl

# Use bash as default shell
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

#### Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    locales tzdata \
    nano wget curl gnupg2 git sudo build-essential software-properties-common ca-certificates ripgrep \
    coreutils bash-completion supervisor \
    lsb-release dirmngr mlocate tree \
    ninja-build ffmpeg=7:* libsm6=2:* libxext6=2:* \
    clang clangd cmake gdb gdbserver x11-apps \
    libpcl-dev libeigen3-dev libopencv-dev libopencv-contrib-dev libcgal-dev libfmt-dev libgoogle-glog-dev libyaml-cpp-dev libboost-all-dev \
    libgl1-mesa-dri libglu1-mesa mesa-utils \
    unzip rsync \
    # python3.9 python3.9-venv python3.9-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#### Set locale
RUN locale-gen en_US.UTF-8 \
    && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata

###  Install ROS Humble
RUN apt-get update \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    ros-$ROS_DISTRO-desktop \
    ros-$ROS_DISTRO-vision-msgs \
    ros-$ROS_DISTRO-vision-msgs-rviz-plugins \
    ros-$ROS_DISTRO-octomap \
    ros-$ROS_DISTRO-octomap-mapping \
    ros-$ROS_DISTRO-octomap-server \
    ros-$ROS_DISTRO-octomap-rviz-plugins \
    ros-$ROS_DISTRO-rosbag2-storage-mcap \
    ros-$ROS_DISTRO-image-view \
    ros-dev-tools \
    python3-dev \
    python3-pip \
    python3-rosdep \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-bloom \
    python3-rosdep \
    python3-argcomplete \
    python3-venv \
    python3-mypy \
    python3-pydocstyle \
    python3-pytest \
    python3-catkin-pkg \
    python3-setuptools \
    fakeroot \
    debhelper \
    dh-python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q -L -O  /tmp/onnxruntime-gpu.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz  \
    && tar -zxvf /tmp/onnxruntime-gpu.tgz -C /opt/ \
    && rm /tmp/onnxruntime-gpu.tgz \
    && echo "/opt/onnxruntime-linux-x64-gpu-1.19.2/lib" >> /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

RUN wget -q -L -O  /tmp/onnxruntime-cpu.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz  \
    && tar -zxvf /tmp/onnxruntime-cpu.tgz -C /opt/ \
    && rm /tmp/onnxruntime-cpu.tgz \
    && echo "/opt/onnxruntime-linux-x64-1.19.2/lib" >> /etc/ld.so.conf.d/onnxruntime.conf \
    && ldconfig

RUN case "${USE_GPU}" in \
        ON|on|1|true|TRUE|True) \
          ln -sfn /opt/onnxruntime-linux-x64-gpu-1.19.2 /opt/onnxruntime-current ;; \
        OFF|off|0|false|FALSE|False) \
          ln -sfn /opt/onnxruntime-linux-x64-1.19.2 /opt/onnxruntime-current ;; \
        *) \
          echo "Invalid USE_GPU='${USE_GPU}'. Expected ON or OFF." >&2; exit 1 ;; \
    esac
ENV PATH=/opt/onnxruntime-current/bin:$PATH

# Install mprocs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Check if "ubuntu" user exists, delete it if it does, then create the desired user
RUN if getent passwd ubuntu > /dev/null 2>&1; then \
    userdel -r ubuntu && \
    echo "Deleted existing ubuntu user"; \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "Created new user $USERNAME"

# Add sudo support for the non-root user
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" > /etc/profile.d/ros2.sh

USER $USERNAME
# WORKDIR $WORKSPACE_DIR
RUN sudo rosdep init && rosdep update --include-eol-distros

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Make cargo available in later Docker layers
ENV PATH="/home/${USERNAME}/.cargo/bin:/home/${USERNAME}/.local/bin:${PATH}"

# Build mprocs from source
RUN git clone https://github.com/pvolok/mprocs.git /home/$USERNAME/mprocs \
 && cd /home/$USERNAME/mprocs \
 && cargo build --release \
 && mkdir -p /home/$USERNAME/.local/bin \
 && install -m 0755 /home/$USERNAME/mprocs/target/release/mprocs /home/$USERNAME/.local/bin/mprocs

USER root
RUN mkdir -p ${WORKSPACE_DIR} \
    && mkdir -p ${WORKSPACE_DIR}/occusg_ws/src \
    && chown -R ${USERNAME}:${USERNAME} ${WORKSPACE_DIR}

USER $USERNAME
WORKDIR ${WORKSPACE_DIR}

# Install ONNX Runtime
RUN source /opt/ros/$ROS_DISTRO/setup.bash \
    && python3 -m venv --system-site-packages --symlinks /home/${USERNAME}/venv \
    && /home/${USERNAME}/venv/bin/python -m pip install --upgrade pip setuptools==70.3.0 wheel certifi

RUN /home/${USERNAME}/venv/bin/pip install --no-cache-dir \
    scikit-learn \
    onnxruntime-gpu \
    pandas \
    openpyxl \
    transforms3d \
    'opencv-contrib-python==4.11.0.86' \
    'opencv-python== 4.11.0.86' \
    'tensorrt==10.12.0.36'
RUN /home/${USERNAME}/venv/bin/pip install --upgrade scipy

RUN /home/${USERNAME}/venv/bin/pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Set up bashrc
RUN echo "alias src_ros='source /opt/ros/$ROS_DISTRO/setup.bash'" >> /home/${USERNAME}/.bashrc
RUN echo "alias mprocs_listener_talker='mprocs -c ${WORKSPACE_DIR}/.mprocs.yaml'" >> /home/${USERNAME}/.bashrc
RUN echo "alias mprocs_tbot='mprocs -c ${WORKSPACE_DIR}/.mprocs.yaml'" >> /home/${USERNAME}/.bashrc
RUN echo "export WORKSPACE_DIR=${WORKSPACE_DIR}" >> /home/${USERNAME}/.bashrc
RUN echo "export CUDA_HOME=/usr/local/cuda" >> /home/${USERNAME}/.bashrc
RUN echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH" >> /home/${USERNAME}/.bashrc
RUN echo "export LD_LIBRARY_PATH=/home/${USERNAME}/venv/lib/python3.10/site-packages/tensorrt_libs:$LD_LIBRARY_PATH" >> /home/${USERNAME}/.bashrc

USER root
ENV SDL_AUDIODRIVER=dummy GAZEBO_AUDIO_DEVICE=none

# Gazebo Classic + TurtleBot3 dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gazebo libgazebo-dev ros-${ROS_DISTRO}-gazebo-* ros-${ROS_DISTRO}-cartographer ros-${ROS_DISTRO}-cartographer-ros ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup ros-${ROS_DISTRO}-turtlebot3-teleop \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install TurtleBot3
USER $USERNAME
WORKDIR ${WORKSPACE_DIR}
RUN mkdir ${WORKSPACE_DIR}/turtlebot3_ws/src -p \
    && cd turtlebot3_ws/src \
    && git clone -b ${ROS_DISTRO} https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git \
    && git clone -b ${ROS_DISTRO} https://github.com/ROBOTIS-GIT/DynamixelSDK.git \
    && git clone -b ${ROS_DISTRO} https://github.com/ROBOTIS-GIT/turtlebot3_msgs.git \
    && git clone -b ${ROS_DISTRO} https://github.com/ROBOTIS-GIT/turtlebot3.git \
    && git clone https://github.com/crcz25/husarion_gz_worlds.git

# Add Realsense-enabled Waffle model
ENV TB3_WS=${WORKSPACE_DIR}/turtlebot3_ws
ENV TB3_SIM=${TB3_WS}/src/turtlebot3_simulations/turtlebot3_gazebo
ENV TB3_DESC=${TB3_WS}/src/turtlebot3/turtlebot3_description/urdf

RUN set -eux; \
    mkdir -p /tmp/realsense_waf; \
    git clone --depth 1 https://github.com/mlherd/ros2_turtlebot3_waffle_intel_realsense /tmp/realsense_waf; \
    # Ensure model dirs exist
    mkdir -p "${TB3_SIM}/models" "${TB3_DESC}"; \
    # Backup original waffle model (optional)
    if [ -d "${TB3_SIM}/models/turtlebot3_waffle" ]; then \
        mv "${TB3_SIM}/models/turtlebot3_waffle" "${TB3_SIM}/models/turtlebot3_waffle_backup"; \
    fi; \
    # Copy Realsense model into simulations package
    cp -r /tmp/realsense_waf/turtlebot3_waffle "${TB3_SIM}/models/"; \
    # Replace waffle URDF
    cp /tmp/realsense_waf/urdf/turtlebot3_waffle.urdf "${TB3_DESC}/turtlebot3_waffle.urdf"; \
    rm -rf /tmp/realsense_waf;

# Make sure Gazebo can find user models
ENV GAZEBO_MODEL_PATH=/home/${USERNAME}/.gazebo/models
ENV GAZEBO_RESOURCE_PATH=/opt/ros/${ROS_DISTRO}/share

# --- Gazebo worlds/models dataset ---
SHELL ["/bin/bash", "-lc"]
WORKDIR ${WORKSPACE_DIR}

ENV HOME=/home/${USERNAME}
ENV GZ_ASSETS=${HOME}/gazebo_assets
ENV GAZEBO_MODEL_DATABASE_URI=""
ENV GAZEBO_PLUGIN_PATH=/opt/ros/${ROS_DISTRO}/lib

# Clone datasets
RUN set -euxo pipefail; \
    git clone --depth=1 https://github.com/mlherd/Dataset-of-Gazebo-Worlds-Models-and-Maps "${GZ_ASSETS}"; \
    git clone --depth=1 https://github.com/osrf/servicesim "${GZ_ASSETS}/servicesim"

# Unpack zipped worlds/assets
RUN set -euxo pipefail; \
    find "${GZ_ASSETS}/worlds" -name "*.zip" -exec bash -c 'unzip -o "$1" -d "$(dirname "$1")"' _ {} \;

# Fix broken mesh/model URIs in source files
RUN set -euxo pipefail; \
    find "${GZ_ASSETS}/worlds" "${GZ_ASSETS}/servicesim" \
      -type f \( -name "*.world" -o -name "*.sdf" \) \
      -exec sed -i -E \
        's#<uri>(file://)?models/([^<]+)</uri>#<uri>model://\2</uri>#g' {} +

# Comment embedded TurtleBot3 waffle includes so launch files control robot spawn
RUN set -euxo pipefail; \
    find "${GZ_ASSETS}/worlds" "${GZ_ASSETS}/servicesim" \
      -type f \( -name "*.world" -o -name "*.sdf" \) \
      -exec sed -z -i -E \
        's#(<include>[[:space:]]*<pose>[^<]*</pose>[[:space:]]*<uri>model://turtlebot3_waffle(_pi)?</uri>[[:space:]]*</include>)#<!-- \1 -->#g' {} +

# Fix known bookstore case-mismatch mesh filenames
RUN set -euxo pipefail; \
    MESH_DIR="${GZ_ASSETS}/worlds/bookstore/models/aws_robomaker_retail_Spotlight_01/meshes"; \
    if [ -d "${MESH_DIR}" ]; then \
      cd "${MESH_DIR}"; \
      ln -sf aws_spotlight_01_collision.DAE aws_Spotlight_01_collision.DAE; \
      ln -sf aws_spotlight_01_visual.DAE    aws_Spotlight_01_visual.DAE; \
    fi

# Helper that builds clean Gazebo env vars from the original dataset structure
RUN set -euxo pipefail; \
    mkdir -p "${HOME}/.local/bin"; \
    cat > "${HOME}/.local/bin/setup_gazebo_assets_env.sh" <<'EOF'
#!/usr/bin/env bash

ROS_DISTRO_NAME="${ROS_DISTRO:-humble}"
HOME_DIR="${HOME:-/home/${USER:-devuser}}"
GZ_ASSETS_DIR="${GZ_ASSETS:-${HOME_DIR}/gazebo_assets}"

# Source ROS/Gazebo if available
[ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ] && source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh
[ -f /usr/share/gazebo-11/setup.sh ] && source /usr/share/gazebo-11/setup.sh

dedup_join_colon() {
  awk 'NF && !seen[$0]++' | paste -sd: -
}

MODEL_ROOTS="$(
  {
    find "${GZ_ASSETS_DIR}/worlds" -type d \( -name models -o -name 'models_part*' \) 2>/dev/null
    [ -d "${GZ_ASSETS_DIR}/servicesim/servicesim_competition/models" ] && \
      printf '%s\n' "${GZ_ASSETS_DIR}/servicesim/servicesim_competition/models"
    [ -d "${GZ_ASSETS_DIR}/robots" ] && \
      printf '%s\n' "${GZ_ASSETS_DIR}/robots"
    [ -d "/usr/share/gazebo-11/models" ] && \
      printf '%s\n' "/usr/share/gazebo-11/models"
  } | dedup_join_colon
)"

RESOURCE_ROOTS="$(
  {
    [ -d "/usr/share/gazebo-11" ] && printf '%s\n' "/usr/share/gazebo-11"
    [ -d "/usr/share/gazebo-11/media" ] && printf '%s\n' "/usr/share/gazebo-11/media"
    find "${GZ_ASSETS_DIR}/worlds" -type d -name media -exec dirname {} \; 2>/dev/null
    [ -d "${GZ_ASSETS_DIR}/servicesim/servicesim_competition" ] && \
      printf '%s\n' "${GZ_ASSETS_DIR}/servicesim/servicesim_competition"
  } | dedup_join_colon
)"

PLUGIN_ROOTS="$(
  {
    [ -d "/opt/ros/${ROS_DISTRO_NAME}/lib" ] && printf '%s\n' "/opt/ros/${ROS_DISTRO_NAME}/lib"
    printf '%s' "${GAZEBO_PLUGIN_PATH:-}" | tr ':' '\n'
  } | dedup_join_colon
)"

export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH="${MODEL_ROOTS}"
export GAZEBO_RESOURCE_PATH="${RESOURCE_ROOTS}"
export GAZEBO_PLUGIN_PATH="${PLUGIN_ROOTS}"
EOF
RUN chmod +x "${HOME}/.local/bin/setup_gazebo_assets_env.sh"

# Source the helper from the user's .bashrc, and rem`1 stale broken exports
RUN set -euxo pipefail; \
    sed -i '/# >>> gazebo_assets >>>/,/# <<< gazebo_assets <<</d' "${HOME}/.bashrc"; \
    sed -i '/GAZEBO_MODEL_PATH/d;/GAZEBO_RESOURCE_PATH/d;/GAZEBO_PLUGIN_PATH/d;/GAZEBO_MODEL_DATABASE_URI/d' "${HOME}/.bashrc"; \
    { \
      echo '# >>> gazebo_assets >>>'; \
      echo 'source "${HOME}/.local/bin/setup_gazebo_assets_env.sh"'; \
      echo '# <<< gazebo_assets <<<'; \
    } >> "${HOME}/.bashrc"

# Sanity checks
RUN set -euxo pipefail; \
    ! grep -RInE '<uri>(file://)?models/' "${GZ_ASSETS}/worlds" "${GZ_ASSETS}/servicesim" 2>/dev/null; \
    test -e "${GZ_ASSETS}/worlds/bookstore/models/aws_robomaker_retail_Spotlight_01/meshes/aws_Spotlight_01_collision.DAE"; \
    test -e "${GZ_ASSETS}/worlds/bookstore/models/aws_robomaker_retail_Spotlight_01/meshes/aws_Spotlight_01_visual.DAE"

#
RUN set -euxo pipefail; \
    { \
      echo '# >>> sim_config >>>'; \
      echo 'export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"'; \
      echo 'export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"'; \
      echo 'export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"'; \
      echo 'source /usr/share/gazebo/setup.sh'; \
      echo '# <<< sim_config <<<'; \
    } >> "${HOME}/.bashrc"

RUN bash -lc "source /opt/ros/$ROS_DISTRO/setup.bash && cd ${WORKSPACE_DIR}/turtlebot3_ws && colcon build --symlink-install"

# Add entrypoint
USER root
COPY --chown=${USERNAME}:${USERNAME} .devcontainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && updatedb

USER $USERNAME
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
