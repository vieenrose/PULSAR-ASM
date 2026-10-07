#!/bin/bash
# Benchmark: sync tree sources to Spark, build, time prefill + decode.
# Emits METRIC lines. Saves step logs to .auto/last_*.txt for checks.sh.
set -euo pipefail
SPARK=luigi@spark
RDIR='~/bonsai2/exp'   # quoted: ~ must expand on the Spark (/home/luigi), not here
LOCAL=/home/user/PULSAR-ASM
A_IDS="248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068"
MODEL=../models/Ternary-Bonsai-2-27B-PTQ1_0.gguf

# 1. sync sources that the loop may change (dir first: redirection can't mkdir)
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" "mkdir -p $RDIR"
for f in fwd.c tq_gemv_neon.c tq_xgemv.c fwht.S; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" "cat > $RDIR/$f" < "$LOCAL/pulsar_arm/bonsai2/$f"
done

# 2. build + benchmark in one ssh call (server-side timing excludes ssh latency)
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" <<EOF > "$LOCAL/.auto/last_build.log" 2>&1
set -euo pipefail
mkdir -p $RDIR && cd $RDIR
LB=\$HOME/bonsai2/llama.cpp/build/bin
gcc -O2 -fopenmp -march=armv8.2-a+dotprod+fp16 -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc -O2 -fopenmp -march=armv8.2-a+dotprod+fp16 -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd_exp fwd.o fwd_neon_mt.o fwht.S -L\$LB -lggml-base -lggml-cpu -Wl,-rpath,\$LB -lm
echo BUILD_OK
export OMP_NUM_THREADS=20
S=\$(date +%s%N)
taskset -c 0-19 ./fwd_exp $MODEL $A_IDS --gen 0 > prefill_steps.txt 2>&1
E=\$(date +%s%N)
echo "PREFILL_NS=\$((E-S)) NSTEPS=\$(grep -c '^step' prefill_steps.txt)"
S=\$(date +%s%N)
taskset -c 0-19 ./fwd_exp $MODEL $A_IDS --gen 20 > decode_steps.txt 2>&1
E=\$(date +%s%N)
echo "DECODE_NS=\$((E-S)) NSTEPS=\$(grep -c '^step' decode_steps.txt)"
cat /proc/loadavg
EOF
grep -q BUILD_OK "$LOCAL/.auto/last_build.log" || { tail -5 "$LOCAL/.auto/last_build.log"; exit 1; }

# 3. fetch step logs
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" "cat $RDIR/prefill_steps.txt" > "$LOCAL/.auto/last_prefill.txt"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" "cat $RDIR/decode_steps.txt" > "$LOCAL/.auto/last_decode.txt"

# 4. parse + validate step counts (21 prompt, 41 total; EOS in window = invalid run)
NP=$(grep -c '^step' "$LOCAL/.auto/last_prefill.txt")
ND=$(grep -c '^step' "$LOCAL/.auto/last_decode.txt")
[ "$NP" -eq 21 ] || { echo "PREFILL STEPS $NP != 21"; exit 1; }
[ "$ND" -eq 41 ] || { echo "DECODE STEPS $ND != 41"; exit 1; }
PNS=$(grep -o 'PREFILL_NS=[0-9]*' "$LOCAL/.auto/last_build.log" | cut -d= -f2)
DNS=$(grep -o 'DECODE_NS=[0-9]*' "$LOCAL/.auto/last_build.log" | cut -d= -f2)
LOAD=$(tail -1 "$LOCAL/.auto/last_build.log" | cut -d' ' -f1)

PREFILL_MS=$(python3 -c "print($PNS/21/1e6)")
DECODE_MS=$(python3 -c "print(($DNS-$PNS)/20/1e6)")
echo "METRIC prefill_ms=$PREFILL_MS"
echo "METRIC decode_ms=$DECODE_MS"
echo "METRIC load1=$LOAD"
echo "prefill 21 fwd: ${PREFILL_MS} ms/fwd; decode 20 gen: ${DECODE_MS} ms/tok; load ${LOAD}"
