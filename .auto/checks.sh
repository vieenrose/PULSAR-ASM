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
MODEL=/home/pesi/b8q1.gguf
A_IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
B_IDS="151644 872 198 7985 264 6386 38242 911 279 17951 624 151645 198 151644 77091 198"

fail() { echo "CHECKS FAILED: $*"; exit 1; }

# --- 0. build from the working tree (never gate a stale binary) --------------
# Run #36 caught this: checks.sh used to run whatever fwd8_q1 / pq1_parity the
# last build left on the Nano, so a kernel-TU-only change measured through
# ab_kernel.sh (which builds its own objects) got a VACUOUS green gate. Proof:
# a deliberately broken kernel (inner loop kk<4) passed all three gates on the
# stale binaries and failed parity 0/256 (maxrel 1.78e+04) once rebuilt.
R=/home/pesi/exp
bash "$REPO/.auto/build_q1.sh" || fail "build failed - refusing to gate unbuilt sources"

# --- 1. kernel bit-exactness -------------------------------------------------
# mode "all" checks EVERY device kernel in the TU against the host reference
# (64-row SWAR, 128-row r128, T-MAC tm, Harley-Seal hs = the engine default
# since loop-261014-2216 row 12). Run #45 made r128 the
# default, so the gate must cover it: a gate that only checks the kernel the
# engine no longer runs is vacuous. Non-zero exit = any kernel not bit-exact.
if ! ssh -o ConnectTimeout=25 "$NANO" "cd $R && ./pq1_parity 256 4096 all >/dev/null 2>&1 && ./pq1_parity 512 12288 all >/dev/null 2>&1"; then
  P=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && ./pq1_parity 256 4096 all 2>&1 | tail -n 3; ./pq1_parity 512 12288 all 2>&1 | tail -n 3" 2>/dev/null)
  fail "pq1 parity (all kernels): $(echo "$P" | grep -E 'MISMATCH|failed' | head -n 2)"
fi
P=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && ./pq1_parity 256 4096 all 2>&1; ./pq1_parity 512 12288 all 2>&1" 2>/dev/null)
[ "$(echo "$P" | grep -c '256/256 bit-exact')" -ge 5 ] || fail "K=4096: not all kernels bit-exact"
[ "$(echo "$P" | grep -c '512/512 bit-exact')" -ge 5 ] || fail "K=12288: not all kernels bit-exact"
echo "checks: pq1 kernels (swar, r128, r256, tm, hs) BIT-EXACT at K=4096 (256/256) and K=12288 (512/512)"

# --- 2. greedy stream goldens ------------------------------------------------
ssh -o ConnectTimeout=25 "$NANO" \
  "cd $R && PQ1_KVTQ=0 OMP_NUM_THREADS=4 timeout 900 ./fwd8_q1 $MODEL $B_IDS --gen 10 2>/dev/null | grep '^step'" \
  > "$REPO/.auto/last_B.txt"
A_TOPS=$(grep '^step' "$REPO/.auto/last_run_log.txt" | sed 's/.*top=//;s/ .*//' | grep -E '^[0-9]+$')
B_TOPS=$(sed 's/.*top=//;s/ .*//' "$REPO/.auto/last_B.txt" | grep -E '^[0-9]+$')

GA=$(wc -l < "$REPO/.auto/golden_q1_A.txt")
GB=$(wc -l < "$REPO/.auto/golden_q1_B.txt")
[ "$(echo "$A_TOPS" | grep -c .)" -eq "$GA" ] || fail "prompt A steps $(echo "$A_TOPS" | grep -c .) != $GA (run aborted?)"
if ! diff <(echo "$A_TOPS") "$REPO/.auto/golden_q1_A.txt" > /dev/null; then
  echo "$A_TOPS" > "$REPO/.auto/last_q1_A_tops.txt"
  diff "$REPO/.auto/last_q1_A_tops.txt" "$REPO/.auto/golden_q1_A.txt" | head -n 5
  fail "prompt A greedy stream differs from golden"
fi
echo "checks: prompt A greedy stream exact ($GA/$GA)"

[ "$(echo "$B_TOPS" | grep -c .)" -eq "$GB" ] || fail "held-out prompt B steps $(echo "$B_TOPS" | grep -c .) != $GB"
if ! diff <(echo "$B_TOPS") "$REPO/.auto/golden_q1_B.txt" > /dev/null; then
  echo "$B_TOPS" > "$REPO/.auto/last_q1_B_tops.txt"
  diff "$REPO/.auto/last_q1_B_tops.txt" "$REPO/.auto/golden_q1_B.txt" | head -n 5
  fail "held-out prompt B greedy stream differs from golden"
fi
echo "checks: held-out prompt B greedy stream exact ($GB/$GB)"

# --- 3. the shipped default: TurboQuant 4-bit K/V cache ------------------------
# Every run above pins PQ1_KVTQ=0 (exact f16 cache, the bit-exact reference).
# The engine default is PQ1_KVTQ=4, which is lossy by design; its greedy
# streams are frozen in golden_q1_{A,B}_tq4.txt (made at the commit that made
# it the default). A regeneration needs a recorded reason in the commit.
for P in A B; do
  if [ $P = A ]; then IDS_P="$A_IDS"; G_P=24; else IDS_P="$B_IDS"; G_P=10; fi
  T_TOPS=$(ssh -o ConnectTimeout=25 "$NANO" \
    "cd $R && OMP_NUM_THREADS=4 timeout 900 ./fwd8_q1 $MODEL $IDS_P --gen $G_P 2>/dev/null | grep '^step'" 2>/dev/null |
    sed 's/.*top=//;s/ .*//' | grep -E '^[0-9]+$')
  GT="$REPO/.auto/golden_q1_${P}_tq4.txt"
  if ! diff <(echo "$T_TOPS") "$GT" > /dev/null; then
    diff <(echo "$T_TOPS") "$GT" | head -n 5
    fail "default (PQ1_KVTQ=4) prompt $P stream differs from golden_q1_${P}_tq4"
  fi
  echo "checks: default TQ4 prompt $P stream matches its golden ($(wc -l < "$GT")/$(wc -l < "$GT"))"
done

# --- 4. time-sliced sessions ---------------------------------------------------
# Four sequences (A B A B, prompts separated by --next) stepped round-robin
# through one engine, each with its own K/V cache and graphs: every stream must
# equal its solo golden, under the exact f16 cache and the TQ4 default (B runs
# 24 tokens here, its golden covers the first 26 steps).
for KV in 0 4; do
  OUT=$(ssh -o ConnectTimeout=25 "$NANO" \
    "cd $R && PQ1_KVTQ=$KV OMP_NUM_THREADS=4 timeout 900 ./fwd8_q1 $MODEL --gen 24 $A_IDS --next $B_IDS --next $A_IDS --next $B_IDS 2>/dev/null | grep '^step'" 2>/dev/null)
  SUF=""; [ $KV = 4 ] && SUF="_tq4"
  for k in 0 1 2 3; do
    if [ $((k % 2)) = 0 ]; then GS="$REPO/.auto/golden_q1_A$SUF.txt"; else GS="$REPO/.auto/golden_q1_B$SUF.txt"; fi
    T=$(echo "$OUT" | grep " sess=$k\$" | sed 's/.*top=//;s/ .*//' | head -n "$(wc -l < "$GS")")
    diff <(echo "$T") "$GS" > /dev/null || fail "session $k of 4 (PQ1_KVTQ=$KV) differs from its solo golden"
  done
  echo "checks: 4 time-sliced sessions (PQ1_KVTQ=$KV) each equal their solo goldens"
done
exit 0
