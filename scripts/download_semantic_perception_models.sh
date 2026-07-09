#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"

models_dir="$workspace_root/models"
groundingdino_dir="$models_dir/groundingdino"
mobilesam_dir="$models_dir/mobilesam"
labels_dir="$models_dir/labels"
clip_dir="$models_dir/clip"

mkdir -p "$groundingdino_dir" "$mobilesam_dir" "$labels_dir" "$clip_dir"

wget -O "$groundingdino_dir/groundingdino_swint_ogc.pth" \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

wget -O "$groundingdino_dir/GroundingDINO_SwinT_OGC.py" \
  "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"

wget -O "$mobilesam_dir/mobile_sam.pt" \
  "https://github.com/ChaoningZhang/MobileSAM/raw/refs/heads/master/weights/mobile_sam.pt"

wget -O "$labels_dir/HM3D_CountsOfObjectTypes.csv" \
  "https://raw.githubusercontent.com/crcz25/HOV-SG/main/hovsg/labels/HM3D_CountsOfObjectTypes.csv"

wget -O "$clip_dir/H_laion2b_s32b_b79k.bin" \
  "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin?download=true"

wget -O "$clip_dir/L_laion2b_s32b_b82k.bin" \
  "  https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K/resolve/main/open_clip_pytorch_model.bin"

cp "$groundingdino_dir/groundingdino_swint_ogc.pth" "$models_dir/groundingdino_swint_ogc.pth"
cp "$groundingdino_dir/GroundingDINO_SwinT_OGC.py" "$models_dir/GroundingDINO_SwinT_OGC.py"
cp "$mobilesam_dir/mobile_sam.pt" "$models_dir/mobile_sam.pt"
cp "$labels_dir/HM3D_CountsOfObjectTypes.csv" "$models_dir/HM3D_CountsOfObjectTypes.csv"

printf 'Downloaded semantic_perception model files into %s\n' "$models_dir"
