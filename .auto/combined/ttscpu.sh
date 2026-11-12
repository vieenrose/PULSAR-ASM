#!/bin/bash
# bash ttscpu.sh <prefix cmd...>: resident TTS started under the prefix (e.g. "taskset -c 3"), 8 back-to-back
# requests; prints walls and CPU freq samples (kHz) of all cores taken mid-request.
cd /home/picard/primetts || exit 1
I=/tmp/tc.pt.in; L=/tmp/tc.pt.log
pkill -x pt_v21_nano; sleep 1; rm -f $I $L; mkfifo $I; : > $L
"$@" ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $I > $L 2>&1 & PS=$!
i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
out=""; fr=""
for k in 0 1 2 3 4 5 6 7; do
  ( timeout 60 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "PT_DONE $k " ) & W=$!
  sleep 0.05; t0=$(date +%s.%N)
  echo "tests/golden_v21/T81 /tmp/tc.npy" > $I
  sleep 0.25; f=$(cat /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_cur_freq | tr '\n' '/'); c=$(awk '{print $39}' /proc/$PS/stat)
  wait $W; t1=$(date +%s.%N)
  out="$out $(echo "$t1 - $t0" | bc | cut -c1-5)"; [ $k -ge 4 ] && fr="$fr [$f cpu$c]"
done
echo "[$*] walls:$out"; echo "   freq mid-request:$fr"
echo Q > $I; sleep 0.5; kill $PS 2>/dev/null; rm -f $I
