#!/usr/bin/env bash
set -euo pipefail

MINE_DIR="/home/crcz/repos/results/MINE"
HYDRA_DIR="/home/crcz/repos/results/HYDRA"
GT_DIR="/home/crcz/repos/results/GT"

MINE_SCRIPT="/home/crcz/repos/3dsg/src/scene_graph_ros/scripts/evaluate_mp3d_regions.py"
HYDRA_SCRIPT="/home/crcz/repos/3dsg/src/scene_graph_ros/scripts/evaluate_hydra_mp3d_regions.py"

MINE_OUT="/home/crcz/repos/results/mine_evals"
HYDRA_OUT="/home/crcz/repos/results/hydra_evals"
LOG_DIR="/home/crcz/repos/results/logs"
LOG_FILE="$LOG_DIR/region_evals_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$MINE_OUT"
mkdir -p "$HYDRA_OUT"
mkdir -p "$LOG_DIR"

# Log all output to a timestamped file as well as the console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Log file: $LOG_FILE"
echo "Started at: $(date -Iseconds)"

echo "========================================"
echo "Running MINE region evaluations"
echo "Input directory: $MINE_DIR"
echo "Output directory: $MINE_OUT"
echo "========================================"

for scan_path in "$MINE_DIR"/*; do
    if [[ ! -d "$scan_path" ]]; then
        continue
    fi

    scan_id="$(basename "$scan_path")"

    echo ""
    echo "----------------------------------------"
    echo "Evaluating MINE scan: $scan_id"
    echo "----------------------------------------"

    python3 "$MINE_SCRIPT" \
        --scan_id "$scan_id" \
        --dsg_dir "$MINE_DIR" \
        --mp3d_root "$GT_DIR" \
        --output_dir "$MINE_OUT" \
        --auto_align
done

echo ""
echo "========================================"
echo "Running HYDRA region evaluations"
echo "Input directory: $HYDRA_DIR"
echo "Output directory: $HYDRA_OUT"
echo "========================================"

for scan_path in "$HYDRA_DIR"/*; do
    if [[ ! -d "$scan_path" ]]; then
        continue
    fi

    scan_id="$(basename "$scan_path")"

    echo ""
    echo "----------------------------------------"
    echo "Evaluating HYDRA scan: $scan_id"
    echo "----------------------------------------"

    python3 "$HYDRA_SCRIPT" \
        --scan_id "$scan_id" \
        --hydra_dir "$HYDRA_DIR" \
        --mp3d_root "$GT_DIR" \
        --output_dir "$HYDRA_OUT" \
        --auto_align
done

echo ""
echo "All evaluations completed."
