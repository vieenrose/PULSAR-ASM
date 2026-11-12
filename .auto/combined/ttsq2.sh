#!/bin/bash
# bash ttsq2.sh <gap_s> <nreq>: fresh resident TTS, N requests, blocking wait (no poll loop).
cd /home/picard/primetts || exit 1
G=${1:-0}; N=${2:-8}
I=/tmp/tq.pt.in; L=/tmp/tq.pt.log
pkill -x pt_v21_nano; sleep 1; rm -f $I $L; mkfifo $I; : > $L
./build/pt_v21_nano --container build/v21.ptk --serve-fifo $I > $L 2>&1 & PS=$!
i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
out=""
for k in $(seq 0 $((N-1))); do
  f=$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq 2>/dev/null || cat /sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq 2>/dev/null)
  ( tail -n +1 -f $L 2>/dev/null | grep -m1 -q "PT_DONE $k " ) & W=$!
  sleep 0.2
  t0=$(date +%s.%N)
  echo "tests/golden_v21/T81 /tmp/tq.npy" > $I
  wait $W
  t1=$(date +%s.%N)
  out="$out $(echo "$t1 - $t0" | bc | cut -c1-5)@$((f/1000000))"
  sleep $G
done
echo "gap=$G:$out"
echo Q > $I; sleep 0.5; kill $PS 2>/dev/null; pkill -f "tail -n +1 -f $L"; rm -f $I
