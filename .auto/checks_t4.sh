#!/bin/bash
# Accuracy gate for the Ternary-Bonsai-4B engine (fwd8_t4 = the pq1 engine
# built with -DPQ1_TERN -DBONSAI_4B, see build_t4.sh). Mirrors checks.sh:
#
#  1. pq1t_parity: the ternary Harley-Seal kernel at every engine shape
#     (qkv 6144x2560, o 2560x4096, gate+up 19456x2560, down 2560x9728, tied head
#     151669x2560, plus ragged-N shapes) BIT-EXACT vs the fork-faithful host
#     reference (ggml_vec_dot_pq2_0_q8_K order), for both PQ1_HST arms.
#  2. rms_q8k at n=2560 (the 4B hidden size) bit-exact vs the serial chain.
#  3. greedy streams of prompts A (44 steps) and B (26 steps) with the exact
#     f16 cache equal golden_t4_{A,B}.txt, which the PrismML fork produced
#     (llama-cli -fa on, f16 cache) on the same checkpoint: the 4B goldens are
#     the reference's output, not the engine's own.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
R=${R:-/home/pesi/t4}
MODEL=/home/pesi/t4b.gguf
A_IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
B_IDS="151644 872 198 7985 264 6386 38242 911 279 17951 624 151645 198 151644 77091 198"
fail() { echo "CHECKS_T4 FAILED: $*"; exit 1; }

bash "$REPO/.auto/build_t4.sh" || fail "build failed - refusing to gate unbuilt sources"

for H in 0 1; do
  P=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && PQ1_HST=$H ./pq1t_parity 2>&1" 2>/dev/null)
  echo "$P" | grep -q 'PQ1T PARITY OK' || { echo "$P" | grep -v BIT-EXACT | head -n 5; fail "pq1t_parity PQ1_HST=$H"; }
done
echo "checks_t4: pq1t kernel BIT-EXACT at all 7 shapes, both PQ1_HST arms"

# the Q1 parity binary carries the rms kernel; build_q1 keeps it current
ssh -o ConnectTimeout=25 "$NANO" "cd /home/pesi/exp && ./pq1_parity rms 2560 2>&1 | grep -q 'RMS PARITY: PASS'" \
  || fail "rms_q8k n=2560 not bit-exact (rebuild /home/pesi/exp with build_q1.sh?)"
echo "checks_t4: rms_q8k n=2560 bit-exact"

for P in A B; do
  if [ $P = A ]; then IDS="$A_IDS"; G=24; else IDS="$B_IDS"; G=10; fi
  T=$(ssh -o ConnectTimeout=25 "$NANO" \
    "cd $R && PQ1_KVTQ=0 OMP_NUM_THREADS=4 timeout 900 ./fwd8_t4 $MODEL --gen $G $IDS 2>/dev/null | grep '^step'" 2>/dev/null |
    sed 's/.*top=//;s/ .*//' | grep -E '^[0-9]+$')
  GF="$REPO/.auto/golden_t4_$P.txt"
  if ! diff <(echo "$T") "$GF" > /dev/null; then
    diff <(echo "$T") "$GF" | head -n 5
    fail "prompt $P stream differs from fork golden_t4_$P"
  fi
  echo "checks_t4: prompt $P greedy stream equals the fork ($(wc -l < "$GF")/$(wc -l < "$GF"))"
done
exit 0
