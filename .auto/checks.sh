#!/bin/bash
# Accuracy gate: greedy streams must equal frozen goldens EXACTLY.
# - Prompt A: reuse measure.sh's saved decode log (41 steps).
# - Prompt B (held-out): run the just-built loop binary on the Spark.
# - If kernel files changed vs HEAD, re-run the NEON bit-exact validation.
set -euo pipefail
SPARK=luigi@spark
LOCAL=/home/user/PULSAR-ASM
B_IDS="248045 846 198 7734 264 6185 36974 883 279 9117 13 248046 198 248045 74455 198 248068"
MODEL=../models/Ternary-Bonsai-2-27B-PTQ1_0.gguf

tops() { grep '^step' "$1" | sed 's/.*top=//;s/ .*//'; }

# A: 41 steps from the measured run
A_N=$(tops "$LOCAL/.auto/last_decode.txt" | wc -l)
[ "$A_N" -eq 41 ] || { echo "CHECKS: A has $A_N tops, want 41"; exit 1; }
if ! diff <(tops "$LOCAL/.auto/last_decode.txt") "$LOCAL/.auto/golden_A.txt" > /dev/null; then
    echo "CHECKS FAILED: prompt-A stream differs from golden"
    diff <(tops "$LOCAL/.auto/last_decode.txt") "$LOCAL/.auto/golden_A.txt" | head -5
    exit 1
fi
echo "checks: prompt A exact (41/41)"

# B: held-out prompt, fresh run of the same binary
ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" \
    "cd ~/bonsai2/exp && OMP_NUM_THREADS=20 taskset -c 0-19 ./fwd_exp $MODEL $B_IDS --gen 10 2>/dev/null | grep '^step'" \
    > "$LOCAL/.auto/last_B.txt"
B_N=$(grep -c '^step' "$LOCAL/.auto/last_B.txt")
[ "$B_N" -eq 27 ] || { echo "CHECKS: B has $B_N steps, want 27"; exit 1; }
if ! diff <(tops "$LOCAL/.auto/last_B.txt") "$LOCAL/.auto/golden_B.txt" > /dev/null; then
    echo "CHECKS FAILED: held-out prompt-B stream differs from golden"
    diff <(tops "$LOCAL/.auto/last_B.txt") "$LOCAL/.auto/golden_B.txt" | head -5
    exit 1
fi
echo "checks: held-out prompt B exact (27/27)"

# Kernel regression: if GEMV/FWHT sources changed, bit-exactness must hold
if git -C "$LOCAL" status --porcelain pulsar_arm/bonsai2/tq_gemv_neon.c pulsar_arm/bonsai2/fwht.S pulsar_arm/bonsai2/tq_xgemv.c | grep -q .; then
    echo "checks: kernel files touched - running bit-exact validation"
    OUT=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$SPARK" \
        'cd ~/bonsai2/exp && LB=~/bonsai2/llama.cpp/build/bin && gcc -O2 -march=armv8.2-a+dotprod+fp16 -DTQ_XGEMV_LIB -o tqv tq_gemv_neon.c tq_xgemv.c fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm && taskset -c 3 ./tqv ../models/Ternary-Bonsai-2-27B-PTQ1_0.gguf 11120992 594124800 64 2>&1 | head -2')
    echo "$OUT"
    if echo "$OUT" | grep -q BIT-EXACT; then echo "checks: kernel bit-exact"; else
        MR=$(echo "$OUT" | grep -o "maxrel [0-9.e+-]*" | head -1 | cut -d" " -f2)
        python3 -c "import sys; sys.exit(0 if float('$MR') < 1e-6 else 1)" \
            || { echo "CHECKS FAILED: kernel diverged (maxrel $MR)"; exit 1; }
        echo "checks: kernel fp32-close (maxrel $MR, reassociation allowed)"
    fi
else
    echo "checks: kernels untouched - bit-exact check skipped"
fi
echo "ALL CHECKS PASSED"
