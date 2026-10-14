#!/bin/bash
# .auto/nano_bench.sh — autoresearch harness for the Jetson Nano Q1_0 (true
# 1-bit) 8B engine (branch bonsai-jetson-nano). Syncs sources, builds, runs
# the fixed prompt, prints METRIC lines.
#
# Metrics
#   decode_ms_per_tok  PRIMARY  marginal per-token forward time (gen steps)
#   prefill_ms_per_tok secondary prompt-token forward time (same kernel)
#   peak_rss_mb        secondary peak RSS of the engine process
#   build_s            secondary wall time of the whole build on the Nano
#
# Fairness notes (do not tune): the 44s one-time split-repack + arena upload
# happens once per process; decode/prefill are measured with in-binary
# per-step timers (BQ1_TIME), so they are the steady-state forward cost and
# exclude that startup. Peak RSS is measured with VmHWM polled from the
# process (no /usr/bin/time on the Nano image) and includes the mmap'd GGUF
# as resident, which is the number that matters for OOM on this board.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
# Prompt A (fixed, never tuned): chatml four-seasons, 20 prompt ids.
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
GEN=${GEN:-24}

mkdir -p "$REPO/.auto"

scp -q -o ConnectTimeout=25 \
  "$SRC/fwd8.c" "$SRC/pq1_expf_neon.h" "$SRC/pq1_swar_sm53.cu" "$SRC/pq1_wrap.h" "$SRC/pq1_host.c" \
  "$SRC/pq1_fwd8_glue.c" "$SRC/pq1_fwd8_shim.h" "$SRC/pq1_parity.c" \
  "$NANO:$R/" || { echo "SCP FAILED"; exit 1; }

BUILD=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH \
  CUDAINC=/usr/local/cuda-10.2/targets/aarch64-linux/include && set -o pipefail && \
  rm -f fwd8_q1 pq1_parity fwd8_q1.o pq1_glue.o pq1_swar_sm53.o && \
  t0=\$(date +%s) && \
  nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c pq1_swar_sm53.cu -o pq1_swar_sm53.o 2>&1 | head -n5 && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA  -c pq1_host.c -o pq1_host.o 2>&1 | head -n3 && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -DBQ1_TIME  -include $R/pq1_fwd8_shim.h -c fwd8.c -o fwd8_q1.o 2>&1 | grep -e ' error' | head -n3; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA  -I\$CUDAINC -c pq1_fwd8_glue.c -o pq1_glue.o 2>&1 | grep -e ' error' | head -n3 || true; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA -I\$CUDAINC -c pq1_parity.c -o pq1_parity.o 2>&1 | grep -e ' error' | head -n3 || true; \
  g++ fwd8_q1.o pq1_host.o pq1_glue.o pq1_swar_sm53.o -o fwd8_q1 -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -n3 && \
  g++ pq1_host.o pq1_parity.o pq1_swar_sm53.o -o pq1_parity -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -n3 && \
  echo BUILT && echo \$((\$(date +%s)-t0))" 2>&1)
echo "$BUILD" | grep -q BUILT || { echo "BUILD FAILED"; echo "$BUILD" | head -n 20; exit 1; }
BUILD_S=$(echo "$BUILD" | tail -n1)

# GPU clock witness (contention on this board changes results; never tuned).
CLK=$(ssh -o ConnectTimeout=25 "$NANO" "cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null || echo 0" 2>/dev/null)

ssh -o ConnectTimeout=25 "$NANO" "cd $R && ${PQ1_EXTRA:-} OMP_NUM_THREADS=4 BQ1_TRACE=0 PQ1_STATS=1 timeout 1200 /usr/bin/time -v ./fwd8_q1 $MODEL $IDS --gen $GEN > run.log 2> time.log; echo RC=\$?" > "$REPO/.auto/last_nano_run.txt" 2>&1
RC=$(grep -o 'RC=[0-9-]*' "$REPO/.auto/last_nano_run.txt" | head -n1 | cut -d= -f2)
HWM=$(ssh -o ConnectTimeout=25 "$NANO" "awk '/Maximum resident set size/{print \$NF}' $R/time.log" 2>/dev/null)
ssh -o ConnectTimeout=25 "$NANO" "cat $R/run.log" > "$REPO/.auto/last_run_log.txt" 2>&1
ssh -o ConnectTimeout=25 "$NANO" "cat $R/time.log" > "$REPO/.auto/last_nano_time.txt" 2>&1

[ "$RC" = "0" ] || { echo "RUN FAILED rc=$RC"; tail -n 5 "$REPO/.auto/last_run_log.txt"; exit 1; }

# Per-step times -> prefill (prompt steps) vs decode (generated steps).
read -r PRE DEC NSTEP STARTUP <<<"$(python3 - "$REPO/.auto/last_run_log.txt" <<'PY'
import re, sys
rows = []
for ln in open(sys.argv[1], errors="replace"):
    m = re.match(r"BQ1TIME step=(\d+) ms=([0-9.]+)", ln)
    if m:
        rows.append((int(m.group(1)), float(m.group(2))))
n0 = 20
# step 0 carries the one-time split-repack + arena upload; excluded from
# prefill so the metric is steady-state forward cost (reported separately).
pre = [ms for s, ms in rows if 0 < s < n0]
dec = [ms for s, ms in rows if s >= n0]
p = sum(pre) / len(pre) if pre else 0.0
d = sum(dec) / len(dec) if dec else 0.0
startup = next((ms for s, ms in rows if s == 0), 0.0)
print(f"{p:.3f} {d:.3f} {len(rows)} {startup:.1f}")
PY
)"

echo "METRIC decode_ms_per_tok=$DEC"
echo "METRIC prefill_ms_per_tok=$PRE"
# Same measurements in tokens/s (1000/ms per token). Reported next to the ms
# values so neither is derived from a different run.
echo "METRIC decode_tps=$(python3 -c "print(f'{1000.0/float(\"$DEC\"):.4f}')")"
echo "METRIC prefill_tps=$(python3 -c "print(f'{1000.0/float(\"$PRE\"):.4f}')")"
echo "METRIC peak_rss_mb=$(( ${HWM:-0} / 1024 ))"
GPU=$(grep -o 'PQ1GPU.*' "$REPO/.auto/last_nano_time.txt" 2>/dev/null | tail -n1)
gv() { echo "$GPU" | sed -n "s/.*$1=\([0-9.]*\).*/\1/p"; }
echo "METRIC gpu_used_mb=$(gv used_mb)"
echo "METRIC gpu_arena_mb=$(gv arena_mb)"
echo "METRIC gpu_peak_mb=$(gv peak_mb)"
echo "METRIC gpu_free_mb=$(gv free_mb)"
echo "METRIC gpu_total_mb=$(gv total_mb)"
echo "METRIC build_s=$BUILD_S"
echo "METRIC startup_s=$(python3 -c "print(f'{$STARTUP/1000:.1f}')")"
echo "METRIC steps=$NSTEP gpu_clk_hz=$CLK"
# Context for the log (not metrics)
tail -n 3 "$REPO/.auto/last_run_log.txt" | sed 's/^/  /'
