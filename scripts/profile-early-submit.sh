#!/bin/bash
# The profile that answers the open question: why does overlapping expert copies with the MoE
# kernels cost 32 % of FFN time at ~6 % of this box's measured 240 GB/s stream rate?
#
# Needs scripts/enable-gpu-profiling.sh applied AND a reboot (it is a module load parameter).
# Verify first:  nsys profile --gpu-metrics-devices=help   must list the GB10.
#
# Two things the first version of this script got wrong, both fatal, both silent:
#   * `nsys profile ./start.sh` profiles a shell that exits in a second. start.sh launches the
#     server with nohup and returns, so the traced process tree is empty and no report is written.
#     The profiler has to be injected where the interpreter actually starts -- start.sh resolves
#     $PYTHON with `command -v` and nohups it, so a shim on $PYTHON is the seam.
#   * the metric set is gb20b, not gb10b. `--gpu-metrics-devices=help` on this box prints
#     "Blackwell GB20B | NVIDIA GB10"; gb10b is a different chip and the run fails.
#
# Structure: `nsys launch` puts the server in a paused session, `nsys start` opens the collection
# window at the request and `nsys stop` closes it and writes the report. So the trace covers the
# prefill only, not the ~80 GB warm start, and the report is finalised by the profiler rather than
# by stop.sh's SIGKILL escalation, which would truncate it.
set -u
R=$HOME/git/deepseek-v41-flash-spark
O=$HOME/ds41-queue/logs
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
REALPY=$(grep -E '^PYTHON=' "$R/.env" | cut -d= -f2-)
PORT=$(grep -E '^PORT=' "$R/.env" | cut -d= -f2-)
SESSION=ds41prof

# the seam: start.sh nohups "$PYTHON" server/app.py, so this is what gets traced.
cat > "$T/nsys-python" <<EOF
#!/bin/bash
exec nsys launch --session-new=$SESSION -t cuda,osrt -n true "$REALPY" "\$@"
EOF
chmod +x "$T/nsys-python"

for mode in off chunk0 all; do
  echo "=== EARLY_SUBMIT=$mode ==="
  cd "$R" && ./stop.sh >/dev/null 2>&1; sleep 4
  find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  nsys sessions list 2>/dev/null | grep -q "$SESSION" && nsys shutdown --session=$SESSION >/dev/null 2>&1

  PYTHON=$T/nsys-python \
  DSV41_PROMPT_CACHE=0 DSV41_EARLY_SUBMIT=$mode DSV41_LM_PHASES=1 \
  DSV41_CB3_CACHE=$HOME/dsv41-cb3/experts-cb3-s3.bin DSV41_DENSE_FP4=attn,wo_a DSV41_HEAD_FMT=fp8 \
    ./start.sh >"$O/nsys-start-$mode.log" 2>&1
  if ! curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  server did not come up -- see $O/nsys-start-$mode.log"; tail -5 "$O/nsys-start-$mode.log"; continue
  fi

  # collection window: the request, and nothing else.
  nsys start --session=$SESSION --gpu-metrics-set=gb20b --gpu-metrics-devices=all \
       --sample=none --cpuctxsw=none -o "$O/nsys-early-$mode" -f true >/dev/null 2>&1 \
    || { echo "  nsys start FAILED for $mode"; cd "$R" && ./stop.sh >/dev/null 2>&1; continue; }
  $HOME/vllm-venv-main-dflash2/bin/python "$HOME/git/deepseek-moe-gb10/tools/longctx_profile.py" 9000 16 2>&1 \
    | grep -E "^ +9000"
  nsys stop --session=$SESSION 2>&1 | tail -2

  cd "$R" && ./stop.sh >/dev/null 2>&1; sleep 6
  ls -la "$O/nsys-early-$mode.nsys-rep" 2>&1 | tail -1
done
echo
echo "reports: $O/nsys-early-{off,chunk0,all}.nsys-rep"
echo
echo "NOTE: there is no memory-bandwidth counter on this box by ANY route -- the gb20b GPU metric"
echo "set has none (clocks, copy engines, GR/SM/Tensor active, warps in flight), --soc-metrics"
echo "answers \"not supported on this system\", nvidia-smi dmon mem% is stubbed on this iGPU, and"
echo "ncu is not installed. The original phrasing of the question cannot be measured directly."
echo
echo "What the available counters DO separate, measured during the MoE kernels:"
echo "  SM Issue % falls, Compute Warps in Flight flat, kernel duration up"
echo "      -> warps resident but stalled: a memory-system stall (L2 or interference)."
echo "  SMs Active drops with gaps, kernel duration unchanged"
echo "      -> launch/scheduling contention, not memory at all."
echo "  Sync/Async Copy Engine Active % at its ceiling while the FFN inflates"
echo "      -> the copy engines themselves are the constraint."
echo "Bandwidth saturation stays implausible on arithmetic alone (~6% of 240 GB/s), and the"
echo "read-early/copy-late A/B settles read-side vs copy-side without any counter at all."
