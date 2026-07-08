#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-${VENV_PATH:-${HOME}/venv}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade \
  pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 packaging==26.2

# Install CUDA PyTorch first. GroundingDINO imports torch from its build script.
"${VENV_PATH}/bin/python" -m pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Hide GPUs only while building GroundingDINO. CUDA runtime images intentionally
# have no nvcc; the package pipeline uses GroundingDINO's portable torch kernel.
CUDA_VISIBLE_DEVICES="" "${VENV_PATH}/bin/python" -m pip install \
  --no-build-isolation --no-cache-dir -r "${PACKAGE_DIR}/requirements.txt"

"${VENV_PATH}/bin/python" -m pip check
echo "Inference environment ready: ${VENV_PATH}"
echo "Activate it with: source ${VENV_PATH}/bin/activate"
