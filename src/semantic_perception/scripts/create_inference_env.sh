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
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Hide GPUs while building GroundingDINO so the install never depends on one
# being present. When nvcc is available (CUDA *devel* image or host toolkit),
# TORCH_CUDA_ARCH_LIST makes GroundingDINO compile its fast _C CUDA extension
# even without a visible GPU (8.6 = RTX 3090; +PTX covers newer GPUs). Without
# nvcc the install still succeeds and inference uses the slower torch fallback.
if command -v nvcc >/dev/null 2>&1; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
fi
CUDA_VISIBLE_DEVICES="" "${VENV_PATH}/bin/python" -m pip install \
  --no-build-isolation --no-cache-dir -r "${PACKAGE_DIR}/requirements.txt"

if command -v nvcc >/dev/null 2>&1; then
  "${VENV_PATH}/bin/python" -c "import torch; import groundingdino._C" \
    || echo "WARNING: groundingdino._C did not compile; inference will use the slower torch fallback" >&2
fi

"${VENV_PATH}/bin/python" -m pip check
echo "Inference environment ready: ${VENV_PATH}"
echo "Activate it with: source ${VENV_PATH}/bin/activate"
