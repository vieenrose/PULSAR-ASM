#!/bin/bash
# K2-0.9B Jetson Nano GPU (CUDA) decode benchmark: EN prompt 21 ids + 200 greedy gen,
# threads 1 (GPU policy: threads-1 only). Builds k2_gexp on the Nano from local
# /tmp/k2_cuda.c (+ fixed k2_cuda.S). Interleaved A/B vs CPU binary k2_base.
# Primary metric: decode tok_s GPU. Gate: 216-step stream bit-identical to golden.
set -u
SRC=${SRC:-/tmp/k2_cuda.c}
PEER=picard@picard-desktop
EN="64018 2985 200 21127 542 1426 265 30066 4318 394 316 1662 37398 15 64019 64018 612 10102 200 64029 200"
GOLDEN=/home/user/PULSAR-ASM/.auto/k2nano_golden.txt
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

# deploy sources
timeout 60 ssh -o BatchMode=yes $PEER 'cat > /tmp/k2_exp.c' < "$SRC" || { echo "GATE FAIL: deploy c"; exit 3; }
timeout 60 ssh -o BatchMode=yes $PEER 'cat > /tmp/k2_exp.S' < /tmp/k2_cuda.S || { echo "GATE FAIL: deploy S"; exit 3; }
ML=$(md5sum "$SRC" | cut -d" " -f1)
MR=$(timeout 30 ssh -o BatchMode=yes $PEER 'md5sum /tmp/k2_exp.c | cut -d" " -f1')
[ "$ML" = "$MR" ] || { echo "GATE FAIL: md5 c"; exit 3; }
ML=$(md5sum /tmp/k2_cuda.S | cut -d" " -f1)
MR=$(timeout 30 ssh -o BatchMode=yes $PEER 'md5sum /tmp/k2_exp.S | cut -d" " -f1')
[ "$ML" = "$MR" ] || { echo "GATE FAIL: md5 S"; exit 3; }

# build exp (gcc link: dlopen libcuda, no -static)
timeout 180 ssh -o BatchMode=yes $PEER \
  'cd /tmp && as -march=armv8-a k2_exp.S -o k2_exp.o && gcc -O2 -c k2_exp.c -o k2_expc.o && gcc -nostartfiles -o k2_gexp k2_exp.o k2_expc.o -ldl' \
  || { echo "GATE FAIL: build"; exit 3; }

runleg_gpu() { # warm(gen0) + timed(gen200); $1 unused (kept for call-site compat)
  local out
  out=$(timeout 500 ssh -o BatchMode=yes $PEER \
    "cd /tmp && ./k2_gexp k2h_09_q4.blob $EN --gen 0 --threads 1 --batch 8 > /dev/null 2>&1; T1=\$(date +%s%N); ./k2_gexp k2h_09_q4.blob $EN --gen 200 --threads 1 --batch 8 > /tmp/gbench_out.txt 2>&1; rc=\$?; T2=\$(date +%s%N); echo \"\$T1 \$T2 \$rc\"")
  read -r B1 B2 RC <<< "$out"
  if [ -z "$B1" ] || [ "$RC" != "0" ]; then
    echo "GATE FAIL: gpu leg (rc=$RC)"; echo "out: [$out]" | head -c 400; echo
    timeout 30 ssh -o BatchMode=yes $PEER 'ls -la /tmp/k2_gexp; head -3 /tmp/gbench_out.txt' 2>&1 | head -6
    exit 3; fi
}
runleg_cpu() {
  local out
  out=$(timeout 500 ssh -o BatchMode=yes $PEER \
    "cd /tmp && T1=\$(date +%s%N); ./k2_base k2h_09_q4.blob $EN --gen 200 --threads 1 --batch 8 > /dev/null 2>&1; rc=\$?; T2=\$(date +%s%N); echo \"\$T1 \$T2 \$rc\"")
  read -r B1 B2 RC <<< "$out"
  [ -n "$B1" ] && [ "$RC" = "0" ] || { echo "GATE FAIL: cpu leg ($RC)"; exit 3; }
}

echo "cooling 60s"; sleep 60
runleg_gpu k2_gexp; AG1=$B1; AG2=$B2
runleg_cpu;        AC1=$B1; AC2=$B2
runleg_cpu;        BC1=$B1; BC2=$B2
runleg_gpu k2_gexp; BG1=$B1; BG2=$B2

calc() { echo "scale=0; ($2 - $1) / 216000000" | bc; }
GPU_MS=$(( ( $(calc $AG1 $AG2) + $(calc $BG1 $BG2) ) / 2 ))
CPU_MS=$(( ( $(calc $AC1 $AC2) + $(calc $BC1 $BC2) ) / 2 ))
python3 -c "
g=$GPU_MS; c=$CPU_MS
tg=1000.0/g; tc=1000.0/c
print(f'METRIC decode_ms={g:.1f}')
print(f'METRIC decode_tok_s={tg:.2f}')
print(f'METRIC base_ms={c:.1f}')
print(f'METRIC base_tok_s={tc:.2f}')
print(f'METRIC ratio={g/c:.4f}')
"

# gate: exp stream vs golden
timeout 60 ssh -o BatchMode=yes $PEER 'grep "^step " /tmp/gbench_out.txt' > $TMP/steps.txt
STEPS=$(wc -l < $TMP/steps.txt)
if [ "$STEPS" != "216" ]; then echo "GATE FAIL: steps=$STEPS want 216"; exit 4; fi
if ! diff -q $TMP/steps.txt $GOLDEN > /dev/null; then echo "GATE FAIL: stream != golden"; exit 4; fi
echo "GATE OK: 216 steps bit-identical vs golden"
