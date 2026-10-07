#!/bin/bash
# Benchmark: sync tree sources to Spark, build CURRENT + HEAD baseline,
# time prefill + decode for both (A/B in one invocation shares conditions).
# Emits METRIC lines. Saves step logs to .auto/last_*.txt for checks.sh.
set -euo pipefail
SPARK=luigi@spark
RDIR='~/bonsai2/exp'   # quoted: ~ must expand on the Spark (/home/luigi), not here
LOCAL=/home/user/PULSAR-ASM
A_IDS="248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068"
MODEL=../models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"

# 1. sync CURRENT sources + bench script
$SSH "$SPARK" "mkdir -p $RDIR $RDIR/base"
for f in fwd.c tq_gemv_neon.c tq_xgemv.c fwht.S; do
    $SSH "$SPARK" "cat > $RDIR/$f" < "$LOCAL/pulsar_arm/bonsai2/$f"
    git -C "$LOCAL" show "HEAD:pulsar_arm/bonsai2/$f" | $SSH "$SPARK" "cat > $RDIR/base/$f"
done
$SSH "$SPARK" "cat > $RDIR/bench_remote.sh" < "$LOCAL/.auto/bench_remote.sh"

# 2. build + benchmark on the Spark (server-side timing excludes ssh latency)
export OMP_NT_EXP=6  # standard threads (run #268 verdict)
export OMP_NT_BASE=6  # baseline tracks standard
export PRE_EXP="taskset -c 0-19"  # standard launcher
export PRE_BASE="taskset -c 0-19"  # standard launcher
# shellcheck disable=SC2086
$SSH "$SPARK" "PRE_EXP='$PRE_EXP' PRE_BASE='$PRE_BASE' OMP_NT_EXP=$OMP_NT_EXP OMP_NT_BASE=$OMP_NT_BASE bash $RDIR/bench_remote.sh $MODEL $A_IDS" > "$LOCAL/.auto/last_build.log" 2>&1
grep -q BUILD_OK "$LOCAL/.auto/last_build.log" || { tail -5 "$LOCAL/.auto/last_build.log"; exit 1; }

# 3. fetch CURRENT-tree step logs (accuracy gate runs on the candidate)
$SSH "$SPARK" "cat $RDIR/exp_pre.txt" > "$LOCAL/.auto/last_prefill.txt"
$SSH "$SPARK" "cat $RDIR/exp_dec.txt" > "$LOCAL/.auto/last_decode.txt"

# 4. parse + validate step counts (21 prompt, 41 total; EOS in window = invalid run)
NP=$(grep -c '^step' "$LOCAL/.auto/last_prefill.txt")
ND=$(grep -c '^step' "$LOCAL/.auto/last_decode.txt")
[ "$NP" -eq 21 ] || { echo "PREFILL STEPS $NP != 21"; exit 1; }
[ "$ND" -eq 41 ] || { echo "DECODE STEPS $ND != 41"; exit 1; }
get() { grep -o "$1=[0-9]*" "$LOCAL/.auto/last_build.log" | head -1 | cut -d= -f2; }
PNS_A=$(get PREFILL_NS_A); PNS_B=$(get PREFILL_NS_B)
DNS_A=$(get DECODE_NS_A); DNS_B=$(get DECODE_NS_B)
NPS_A=$(grep -o 'NSTEPS_A=[0-9]*' "$LOCAL/.auto/last_build.log" | head -1 | cut -d= -f2)
NDS_A=$(grep -o 'NSTEPS_A=[0-9]*' "$LOCAL/.auto/last_build.log" | tail -1 | cut -d= -f2)
[ "$NPS_A" -eq 21 ] || { echo "BASE/PRE STEPS $NPS_A"; exit 1; }
[ "$NDS_A" -eq 41 ] || { echo "EXP/DEC STEPS $NDS_A"; exit 1; }
NPS_B=$(grep -o 'NSTEPS_B=[0-9]*' "$LOCAL/.auto/last_build.log" | head -1 | cut -d= -f2)
NDS_B=$(grep -o 'NSTEPS_B=[0-9]*' "$LOCAL/.auto/last_build.log" | tail -1 | cut -d= -f2)
[ "$NPS_B" -eq 21 ] || { echo "BASE/PRE STEPS $NPS_B"; exit 1; }
[ "$NDS_B" -eq 41 ] || { echo "BASE/DEC STEPS $NDS_B"; exit 1; }
LOAD=$(tail -1 "$LOCAL/.auto/last_build.log" | cut -d' ' -f1)

PREFILL_MS=$(python3 -c "print($PNS_A/21/1e6)")
DECODE_MS=$(python3 -c "print(($DNS_A-$PNS_A)/20/1e6)")
DECODE_BASE_MS=$(python3 -c "print(($DNS_B-$PNS_B)/20/1e6)")
RATIO=$(python3 -c "print(($DNS_A-$PNS_A)/($DNS_B-$PNS_B))")
echo "METRIC prefill_ms=$PREFILL_MS"
echo "METRIC decode_ms=$DECODE_MS"
echo "METRIC decode_base_ms=$DECODE_BASE_MS"
echo "METRIC decode_ratio=$RATIO"
echo "METRIC load1=$LOAD"
echo "decode current ${DECODE_MS} vs base ${DECODE_BASE_MS} (ratio ${RATIO}), load ${LOAD}"
