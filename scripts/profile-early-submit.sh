#!/bin/bash
# AFTER the reboot: the profile that answers the open question. Three arms, GPU metrics on.
# Needs scripts/enable-gpu-profiling.sh applied and a reboot; no sudo at run time.
set -u
R=$HOME/git/deepseek-v41-flash-spark
O=$HOME/ds41-queue/logs
for mode in off chunk0 all; do
  echo "=== EARLY_SUBMIT=$mode ==="
  cd "$R" && ./stop.sh >/dev/null 2>&1; sleep 4
  find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  DSV41_PROMPT_CACHE=0 DSV41_EARLY_SUBMIT=$mode DSV41_LM_PHASES=1 \
  DSV41_CB3_CACHE=$HOME/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a DSV41_HEAD_FMT=fp8 \
    nsys profile --gpu-metrics-set=gb10b --gpu-metrics-devices=all \
      --trace=cuda,osrt --sample=none -o "$O/nsys-early-$mode" --force-overwrite=true \
      ./start.sh --no-wait >/dev/null 2>&1 &
  for i in $(seq 1 40); do curl -sf -m 3 http://127.0.0.1:8001/health >/dev/null 2>&1 && break; sleep 15; done
  $HOME/vllm-venv-main-dflash2/bin/python "$HOME/git/deepseek-moe-gb10/tools/longctx_profile.py" 9000 16 2>&1 | grep -E "^ +9000"
  cd "$R" && ./stop.sh >/dev/null 2>&1; sleep 6
done
echo "reports in $O/nsys-early-*.nsys-rep -- open the GPU Memory Bandwidth row against the"
echo "attention / H2D / FFN phases. The question: does mem BW rise 55 -> 70 -> 90% as FFN time"
echo "rises, which would settle the contention mechanism."
