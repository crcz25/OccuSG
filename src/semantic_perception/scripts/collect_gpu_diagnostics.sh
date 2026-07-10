#!/usr/bin/env bash
# Capture a timestamped snapshot of GPU, driver, kernel, thermal, power, PCIe,
# and container state for diagnosing fatal CUDA failures (e.g. Xid 79 / "GPU
# has fallen off the bus"). Run it BEFORE a stress test and again IMMEDIATELY
# after a failure, then diff the two snapshot directories.
#
# Usage:
#   ./collect_gpu_diagnostics.sh [output_root]
#
# Root is not required; anything that needs elevated access (kernel log,
# lspci) is attempted through non-interactive sudo and skipped with a note
# when unavailable. Inside a container the kernel log still reflects the HOST
# kernel, so Xid entries are visible when sudo is permitted.

set -u
ROOT="${1:-gpu_diagnostics}"
OUT="${ROOT}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"
echo "Writing diagnostics to ${OUT}"

capture() { # capture <filename> <command...>
    local file="${OUT}/$1"
    shift
    {
        echo "# \$ $*"
        echo "# captured $(date -Is)"
        "$@"
        echo "# exit status: $?"
    } >"${file}" 2>&1
}

# --- GPU and driver state -------------------------------------------------
capture nvidia-smi.txt nvidia-smi
capture nvidia-smi-q.txt nvidia-smi -q
capture nvidia-smi-topo.txt nvidia-smi topo -m
capture nvidia-smi-csv.txt nvidia-smi \
    --query-gpu=timestamp,index,name,uuid,serial,temperature.gpu,power.draw,power.limit,enforced.power.limit,memory.used,memory.total,utilization.gpu,utilization.memory,pstate,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,fan.speed \
    --format=csv
# Short utilization/power sample (10 s) to see live behaviour.
capture nvidia-smi-dmon.txt timeout 10 nvidia-smi dmon -c 10
capture nvidia-smi-pmon.txt timeout 10 nvidia-smi pmon -c 10

# --- Kernel and driver diagnostics (host-level; needs sudo) ----------------
if sudo -n true 2>/dev/null; then
    capture dmesg-nvidia.txt bash -c \
        "sudo -n dmesg -T | grep -iE 'NVRM|Xid|GPU|PCIe|fallen off|AER|thermal|over.?current|power' || true"
    capture dmesg-full-tail.txt bash -c "sudo -n dmesg -T | tail -n 500"
    command -v lspci >/dev/null 2>&1 &&
        capture lspci-nvidia.txt bash -c \
            "sudo -n lspci -d 10de: -vv 2>/dev/null || sudo -n lspci -vv"
else
    echo "sudo unavailable: kernel log (dmesg) and lspci skipped" \
        >"${OUT}/dmesg-nvidia.txt"
fi
command -v journalctl >/dev/null 2>&1 && capture journalctl-nvidia.txt bash -c \
    "journalctl -k -b --no-pager 2>/dev/null | grep -iE 'NVRM|Xid|GPU|PCIe|fallen off|AER' || echo 'no journal access or no entries'"

# --- Thermal / power sensors ------------------------------------------------
command -v sensors >/dev/null 2>&1 && capture sensors.txt sensors
[ -d /sys/class/thermal ] && capture thermal-zones.txt bash -c \
    'for z in /sys/class/thermal/thermal_zone*/; do
         [ -f "$z/temp" ] && echo "$(cat "$z/type" 2>/dev/null): $(cat "$z/temp")"
     done'

# --- Container / runtime versions -------------------------------------------
capture os-release.txt cat /etc/os-release
capture kernel.txt uname -a
command -v docker >/dev/null 2>&1 && capture docker-info.txt docker info
command -v nvidia-container-cli >/dev/null 2>&1 &&
    capture nvidia-container-cli.txt nvidia-container-cli info
capture nvidia-devices.txt bash -c 'ls -l /dev/nvidia* 2>&1'

# --- PyTorch stack -----------------------------------------------------------
PYTHON="${PYTHON:-/home/devuser/venv/bin/python}"
[ -x "${PYTHON}" ] || PYTHON=python3
capture torch-env.txt "${PYTHON}" -c '
import torch
print("torch", torch.__version__)
print("built for CUDA", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} {p.total_memory/2**30:.1f} GiB sm_{p.major}{p.minor}")
'

# --- Application log tail (pass SEM_PERCEPTION_LOG to include it) -----------
if [ -n "${SEM_PERCEPTION_LOG:-}" ] && [ -f "${SEM_PERCEPTION_LOG}" ]; then
    tail -n 300 "${SEM_PERCEPTION_LOG}" >"${OUT}/application-log-tail.txt"
fi

echo "Done. Key check: grep -i xid '${OUT}/dmesg-nvidia.txt'"
grep -i "xid" "${OUT}/dmesg-nvidia.txt" 2>/dev/null | tail -n 5 || true
