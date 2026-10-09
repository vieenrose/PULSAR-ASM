#!/bin/bash
# K2-FT phone benchmark: fixed workload, timed, bit-exactness gated.
# Workload: 680-id FT meeting prompt + 400 greedy gen, --threads 4 --batch 8,
# taskset f0 (big cores), Galaxy Note 10+. Binary must already be deployed
# as /data/local/tmp/k2_core with k2h_ft_q4.blob + ft_ids680.txt present.
# Meter is tokens/sec (higher is better); ms kept for reference.
set -u
OUT=${1:-/tmp/k2b}
IDS=$(cat /tmp/ft_ids680.txt)
GOLDEN=/home/user/PULSAR-ASM/.auto/k2ft_ref.txt

bigtemp() {
  adb shell 'for z in 11 12 13 14; do cat /sys/class/thermal/thermal_zone$z/temp; done' 2>/dev/null \
    | sort -n | tail -1
}

echo "thesis: fixed 680-id FT prompt + 400 gen, greedy t4b8"
T0=$(bigtemp); echo "temp_pre_mC=$T0"

# prefill probe first: times prefill AND warms page cache / clocks
P0=$(date +%s%N)
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_ft_q4.blob $IDS --gen 0 --threads 4 --batch 8" > $OUT.pre.txt 2>&1
P1=$(date +%s%N)
PRE_MS=$(( (P1 - P0) / 1000000 ))

# timed full workload (warm)
S0=$(date +%s%N)
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_ft_q4.blob $IDS --gen 400 --threads 4 --batch 8" > $OUT.full.txt 2>&1
S1=$(date +%s%N)
TOTAL_MS=$(( (S1 - S0) / 1000000 ))
STEPS=$(grep -c ^step $OUT.full.txt || echo 0)
echo "total_ms=$TOTAL_MS steps=$STEPS (expect 1080)"

if [ "$STEPS" != "1080" ]; then echo "GATE FAIL: step count"; exit 4; fi
if ! diff <(grep ^step $OUT.full.txt) <(grep ^step $GOLDEN) > /dev/null; then
  echo "GATE FAIL: steps differ from golden"; exit 4
fi
echo "GATE OK: 1080 steps bit-identical"

TOK_S=$(python3 -c "print(f'{(1080 / ($TOTAL_MS / 1000)):.2f}')")
PRE_TS=$(python3 -c "print(f'{(680 / ($PRE_MS / 1000)):.2f}')")
DEC_TS=$(python3 -c "print(f'{(400 / (($TOTAL_MS - $PRE_MS) / 1000)):.2f}')")
DEC_MS=$(python3 -c "print(f'{(($TOTAL_MS - $PRE_MS) / 400):.1f}')")
T1=$(bigtemp); echo "temp_post_mC=$T1"
echo "METER: ${TOK_S} tok/s overall | ${PRE_TS} tok/s prefill | ${DEC_TS} tok/s decode"
echo "METRIC tok_s=$TOK_S"
echo "METRIC prefill_tok_s=$PRE_TS"
echo "METRIC decode_tok_s=$DEC_TS"
echo "METRIC total_ms=$TOTAL_MS"
echo "METRIC prefill_ms=$PRE_MS"
echo "METRIC decode_ms_per_tok=$DEC_MS"
echo "METRIC temp_mC=$T1"
echo "METRIC steps=$STEPS"
