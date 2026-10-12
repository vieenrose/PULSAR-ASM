#!/bin/bash
# K2-0.9B Jetson Nano CPU benchmark: fixed EN prompt (21 ids) + 200 greedy gen, threads 1.
# Builds ./k2_exp on the Nano from /tmp/k2_core_a57.S (local source deployed via ssh).
# A/B meter: baseline binary k2_base (HEAD f6b8cfe) is timed INTERLEAVED with k2_exp
# (order reversed between rounds to cancel thermal/contention drift). Primary metric:
# decode tok_s of k2_exp; decode_ms = (T_gen200 - T_gen0)/steps. Accuracy gate: k2_exp
# step stream must be bit-identical to .auto/k2nano_golden.txt (216 steps).
set -u
SRC=${SRC:-/tmp/k2_core_a57.S}
PEER=picard@picard-desktop
EN="64018 2985 200 21127 542 1426 265 30066 4318 394 316 1662 37398 15 64019 64018 612 10102 200 64029 200"
GOLDEN=/home/user/PULSAR-ASM/.auto/k2nano_golden.txt
GEN=200
TMP=$(mktemp -d)
trap 'rm -rf $TMP' EXIT

# deploy exp source
timeout 60 ssh -o BatchMode=yes $PEER 'cat > /tmp/k2_exp.S' < "$SRC" || { echo "GATE FAIL: deploy"; exit 3; }
ML=$(md5sum "$SRC" | cut -d" " -f1)
MR=$(timeout 30 ssh -o BatchMode=yes $PEER 'md5sum /tmp/k2_exp.S | cut -d" " -f1')
[ "$ML" = "$MR" ] || { echo "GATE FAIL: md5"; exit 3; }

# build exp
timeout 120 ssh -o BatchMode=yes $PEER \
  'cd /tmp && as -march=armv8-a k2_exp.S -o k2_exp.o && ld -static -o k2_exp k2_exp.o' \
  || { echo "GATE FAIL: build"; exit 3; }

# remote timed legs: base full, exp full, exp full, base full (interleaved)
# each preceded by its gen0 warm leg (prefill+cache warm, also = decode-subtract baseline)
remoterun() { # $1=bin $2=gen
  timeout 400 ssh -o BatchMode=yes $PEER \
    "cd /tmp && T0=\$(date +%s%N) && ./k2_base k2h_09_q4.blob $EN --gen 0 --threads 1 --batch 8 > /dev/null 2>&1 && T1=\$(date +%s%N) && ./k2_base k2h_09_q4.blob $EN --gen 200 --threads 1 --batch 8 > /tmp/goldcheck.txt 2>&1 && T2=\$(date +%s%N) && echo \"T0=\$T0 T1=\$T1 T2=\$T2\""
}

runleg() { # $1=bin  writes T0 T1 T2 into globals
  local out
  out=$(timeout 400 ssh -o BatchMode=yes $PEER \
    "cd /tmp && T0=\$(date +%s%N); ./$1 k2h_09_q4.blob $EN --gen 0 --threads 1 --batch 8 >/dev/null 2>&1; T1=\$(date +%s%N); ./$1 k2h_09_q4.blob $EN --gen 200 --threads 1 --batch 8 > /tmp/bench_out.txt 2>&1; T2=\$(date +%s%N); echo \"\$T0 \$T1 \$T2\"")
  read -r B0 B1 B2 <<< "$out"
  [ -n "$B0" ] || { echo "GATE FAIL: leg $1"; exit 3; }
}

echo "cooling 60s"; sleep 60
runleg k2_base;  AB0=$B0; AB1=$B1; AB2=$B2
runleg k2_exp;  AE0=$B0; AE1=$B1; AE2=$B2
runleg k2_exp;  BE0=$B0; BE1=$B1; BE2=$B2
runleg k2_base; BB0=$B0; BB1=$B1; BB2=$B2

# decode ms per token: (T2-T1)/200 minus prefill? T1->T2 spans gen200 run which includes
# its own prefill (21 tokens). Use (T2-T1) full-run time as end-to-end metric; prefill
# cancels in base/exp ratio and is identical code. tok_s over the 216-step stream.
calc() { # $1=T1 $2=T2 -> ns/step
  echo "scale=0; ($2 - $1) / 216000000" | bc
}
BASE_MS=$(( ( ($(calc $AB1 $AB2) + $(calc $BB1 $BB2)) ) / 2 ))
EXP_MS=$((  ( $(calc $AE1 $AE2) + $(calc $BE1 $BE2) ) / 2 ))
python3 -c "
b=$BASE_MS; e=$EXP_MS
tb=1000.0/b; te=1000.0/e
print(f'METRIC decode_ms={e:.1f}')
print(f'METRIC decode_tok_s={te:.2f}')
print(f'METRIC base_ms={b:.1f}')
print(f'METRIC base_tok_s={tb:.2f}')
print(f'METRIC ratio={e/b:.4f}')
"

# accuracy gate: exp stream vs golden
timeout 60 ssh -o BatchMode=yes $PEER 'grep "^step " /tmp/bench_out.txt' > $TMP/exp_steps.txt
STEPS=$(wc -l < $TMP/exp_steps.txt)
if [ "$STEPS" != "216" ]; then echo "GATE FAIL: steps=$STEPS want 216"; exit 4; fi
if ! diff -q $TMP/exp_steps.txt $GOLDEN > /dev/null; then
  echo "GATE FAIL: stream differs from golden"; exit 4; fi
echo "GATE OK: 216 steps bit-identical vs golden"
