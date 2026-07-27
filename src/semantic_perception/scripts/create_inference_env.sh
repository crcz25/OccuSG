#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${1:-${VENV_PATH:-${HOME}/venv}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv --system-site-packages "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade \
  pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 packaging==26.2

# Install CUDA PyTorch first. GroundingDINO imports torch from its build script.
# cu128 is required for Blackwell (sm_120), while retaining RTX 3090 (sm_86).
"${VENV_PATH}/bin/python" -m pip install --no-cache-dir \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

# GroundingDINO's pinned extension uses a pre-2.6 PyTorch C++ API. Do not
# request a CUDA extension build; the node uses GroundingDINO's portable
# deformable-attention implementation when _C is absent.
env -u TORCH_CUDA_ARCH_LIST CUDA_VISIBLE_DEVICES="" \
  "${VENV_PATH}/bin/python" -m pip install \
  --no-build-isolation --no-cache-dir -r "${PACKAGE_DIR}/requirements.txt"

"${VENV_PATH}/bin/python" -c "import torch; from groundingdino.util.inference import Model; assert torch.version.cuda == '12.8', torch.version.cuda; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda} installed')"

"${VENV_PATH}/bin/python" -m pip check
echo "Inference environment ready: ${VENV_PATH}"
echo "Activate it with: source ${VENV_PATH}/bin/activate"
