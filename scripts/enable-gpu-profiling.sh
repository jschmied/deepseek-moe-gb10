#!/bin/bash
# Lift NVIDIA's perf-counter restriction so nsys/ncu can read GPU metrics unprivileged.
#
# WHY: nsys 2025.3.2 on this box HAS the right metric set for this chip (`gb10b`) but refuses with
#   GPU Metrics: None of the installed GPUs are supported:
#       Blackwell GB20B | NVIDIA GB10 - Insufficient privilege, see ERR_NVGPUCTRPERM
# and there is no fallback: tegrastats is not installed (no EMC counter), ncu is not installed, and
# `nvidia-smi dmon`'s mem% is STUBBED on this iGPU -- measured 2026-09-14, a saturating 8192^3 matmul
# drives sm% to 96 while mem% stays 0 and fb/bar1 report "-".
#
# So this is the only route to a memory-system number on this machine, and the open question needs
# one: overlapping expert copies with MoE kernels slows the FFN by 32% at ~6% of the theoretical
# stream rate, and we cannot tell whether that is L2 pollution, controller latency, or launch
# contention without a profile.
#
# SCOPE: this lifts exactly one restriction -- who may read GPU performance counters. It grants no
# other privilege. Preferred over passwordless sudo, which would widen every command in the session.
#
# Run with: sudo bash scripts/enable-gpu-profiling.sh     then REBOOT.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
CONF=/etc/modprobe.d/nvidia-profiling.conf
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' > "$CONF"
chmod 644 "$CONF"
echo "wrote $CONF:"; cat "$CONF"
if command -v update-initramfs >/dev/null; then update-initramfs -u; fi
echo
echo "Done. REBOOT, then verify as the normal user with:"
echo "    nsys profile --gpu-metrics-devices=help"
echo "It should list the GB10 instead of 'Insufficient privilege'."
