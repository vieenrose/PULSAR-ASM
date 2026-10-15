#!/bin/bash
# Accuracy gate for the Nano Q1_0 (true 1-bit) engine (branch bonsai-jetson-nano).
# Runs automatically after every passing benchmark. Two independent checks:
#
#  1. Kernel bit-exactness: pq2_parity must report BIT-EXACT at K=4096 and
#     K=12288 vs the fork-faithful host reference (which mirrors
#     pq2_gemv.c / ggml_vec_dot_pq2_0_q8_K's accumulation order).
#  2. End-to-end greedy streams: prompt A (measured) and prompt B
#     (held-out, different content) must reproduce the frozen goldens
#     token-for-token. Any numerics change that alters the argmax path
#     fails here, so speed cannot be bought with accuracy.
#
# Nothing here is tuned to the benchmark prompts' content; the goldens were
# frozen from the pre-optimization HEAD and are the engine's own output.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
R=/home/pesi/exp
MODEL=/home/pesi/b8.gguf
A_IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
B_IDS="151644 872 198 7985 264 6386 38242 911 279 17951 624 151645 198 151644 77091 198"

fail() { echo "CHECKS FAILED: $*"; exit 1; }

# --- 1. kernel bit-exactness -------------------------------------------------
P=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && ./pq1_parity 256 4096 2>&1 | tail -n 2; ./pq1_parity 512 12288 2>&1 | tail -n 2" 2>/dev/null)
echo "$P" | grep -q "256/256 bit-exact" || fail "K=4096 parity: $(echo "$P" | head -n1)"
echo "$P" | grep -q "512/512 bit-exact" || fail "K=12288 parity: $(echo "$P" | tail -n1)"
echo "checks: pq1 kernel BIT-EXACT at K=4096 (256/256) and K=12288 (512/512)"

# --- 2. greedy stream goldens ------------------------------------------------
ssh -o ConnectTimeout=25 "$NANO" \
  "cd $R && PQ1_KVTQ=0 OMP_NUM_THREADS=4 timeout 900 ./fwd8_q1 $MODEL $B_IDS --gen 10 2>/dev/null | grep '^step'" \
  > "$REPO/.auto/last_B.txt"
A_TOPS=$(grep '^step' "$REPO/.auto/last_run_log.txt" | sed 's/.*top=//;s/ .*//' | grep -E '^[0-9]+$')
B_TOPS=$(sed 's/.*top=//;s/ .*//' "$REPO/.auto/last_B.txt" | grep -E '^[0-9]+$')

GA=$(wc -l < "$REPO/.auto/golden_A.txt")
GB=$(wc -l < "$REPO/.auto/golden_B.txt")
[ "$(echo "$A_TOPS" | grep -c .)" -eq "$GA" ] || fail "prompt A steps $(echo "$A_TOPS" | grep -c .) != $GA (run aborted?)"
if ! diff <(echo "$A_TOPS") "$REPO/.auto/golden_A.txt" > /dev/null; then
  echo "$A_TOPS" > "$REPO/.auto/last_A_tops.txt"
  diff "$REPO/.auto/last_A_tops.txt" "$REPO/.auto/golden_A.txt" | head -n 5
  fail "prompt A greedy stream differs from golden"
fi
echo "checks: prompt A greedy stream exact ($GA/$GA)"

[ "$(echo "$B_TOPS" | grep -c .)" -eq "$GB" ] || fail "held-out prompt B steps $(echo "$B_TOPS" | grep -c .) != $GB"
if ! diff <(echo "$B_TOPS") "$REPO/.auto/golden_B.txt" > /dev/null; then
  echo "$B_TOPS" > "$REPO/.auto/last_B_tops.txt"
  diff "$REPO/.auto/last_B_tops.txt" "$REPO/.auto/golden_B.txt" | head -n 5
  fail "held-out prompt B greedy stream differs from golden"
fi
echo "checks: held-out prompt B greedy stream exact ($GB/$GB)"
exit 0
