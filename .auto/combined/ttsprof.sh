#!/bin/bash
# bash ttsprof.sh [env...]: fresh resident TTS with the given env, 6 back-to-back requests; wall, user/sys
# CPU of the service per request, then the log of the last request.
cd /home/picard/primetts || exit 1
I=/tmp/tp.pt.in; L=/tmp/tp.pt.log
pkill -x pt_v21_nano; sleep 1; rm -f $I $L; mkfifo $I; : > $L
env "$@" ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $I > $L 2>&1 & PS=$!
i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
cpu() { awk '{print $14, $15}' /proc/$PS/stat; }
for k in 0 1 2 3 4 5; do
  ( timeout 60 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "PT_DONE $k " ) & W=$!
  sleep 0.05; c0=$(cpu); n0=$(wc -l < $L); t0=$(date +%s.%N)
  echo "tests/golden_v21/T81 /tmp/tp.npy" > $I
  wait $W
  t1=$(date +%s.%N); c1=$(cpu)
  echo "req $k wall=$(echo "$t1 - $t0" | bc | cut -c1-5) user_ms=$(( ($(echo $c1 | cut -d' ' -f1) - $(echo $c0 | cut -d' ' -f1)) * 10 )) sys_ms=$(( ($(echo $c1 | cut -d' ' -f2) - $(echo $c0 | cut -d' ' -f2)) * 10 )) threads=$(ls /proc/$PS/task | wc -l)"
done
echo "--- last request log:"; tail -n +$((n0+1)) $L | head -60
echo Q > $I; sleep 0.5; kill $PS 2>/dev/null; rm -f $I
