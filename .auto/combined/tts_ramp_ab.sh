#!/bin/bash
# bash tts_ramp_ab.sh [gap_s] [rounds]  (on the board: /home/picard/primetts)
#
# Paired, order-balanced A/B of the arrival clock ramp (pt_v21_nano --serve-fifo, PT_V21_RAMP_MS).
# The metric never sees this path: in the harness the TTS request follows the ASR turn with no gap,
# so the GPU is at max and serve_ramp() returns without launching anything. The deployment case is
# a gap (whatever produces the reply text), and a request arriving on a parked clock costs 2-3x
# (0.48 s at 921.6 MHz vs 1.0-1.7 s at 76.8 MHz). So the A/B has to CREATE the gap: N requests
# separated by GAP seconds, wall timed from the request write to PT_DONE -- which includes the ramp
# itself, so a ramp that does not pay for itself shows up as a loss.
# Arm 1 = shipped default (ramp on), arm 0 = PT_V21_RAMP_MS=0. Order 1,0,1,0,... cancels drift.
cd /home/picard/primetts || exit 1
G=${1:-6}; R=${2:-2}
CASE=${CASE:-tests/golden_v21/T81}
for rep in $(seq 1 $R); do
  for arm in 1 0; do
    I=/tmp/ra.pt.in; L=/tmp/ra.pt.log
    pkill -x pt_v21_nano 2>/dev/null; sleep 1; rm -f $I $L; mkfifo $I; : > $L
    out=""
    if [ "$arm" = 0 ]; then export PT_V21_RAMP_MS=0; else unset PT_V21_RAMP_MS; fi
    ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $I > $L 2>&1 & PS=$!
    i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
    for k in $(seq 0 2); do
      f=$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq 2>/dev/null)
      ( timeout 90 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "PT_DONE $k " ) & W=$!
      sleep $G
      t0=$(date +%s.%N)
      echo "$CASE /tmp/ra.npy" > $I
      wait $W
      t1=$(date +%s.%N)
      out="$out $(echo "$t1 - $t0" | bc | cut -c1-5)@$((f/1000000))"
    done
    echo "RAMP=$arm rep=$rep gap=$G walls:$out"
    grep "clock parked" $L | sed 's/^/   /' | tail -3
    echo Q > $I 2>/dev/null; sleep 0.5; kill $PS 2>/dev/null
    pkill -f "tail -n +1 -f $L" 2>/dev/null; rm -f $I
  done
done
