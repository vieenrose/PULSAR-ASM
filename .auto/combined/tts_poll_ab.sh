#!/usr/bin/env bash
# .auto/combined/tts_poll_ab.sh -- A/B of the resident TTS request reader (prime-tts main_v21.c).
#   new  default: one persistent O_RDWR fifo fd, poll() blocks until a request arrives
#   old  PT_V21_FIFO_POLL=1: the 20 ms polling reader that reopens the fifo on every EOF
# One build, the service is rebooted per arm (ABBA by default), N back-to-back T81 requests per
# boot so the GPU clock stays at max (#277), wall = request write -> PT_DONE line, measured the same
# way for both arms (1 ms poll of the log). Every waveform is byte-compared with a one-shot reference
# made before the first boot: the reader change must not alter a single sample.
set -u
cd /home/picard/primetts
N=${N:-8}; ORDER=${ORDER:-"new old old new"}; CASE=tests/golden_v21/T81
IN=/tmp/tpa.in; LOG=/tmp/tpa.log
./build/pt_v21_nano --container build/v21.ptk --case $CASE --out /tmp/tpa.ref.npy > /tmp/tpa.ref.log 2>&1 \
  || { echo "REF_FAIL"; exit 1; }
for arm in $ORDER; do
  E=""; [ "$arm" = old ] && E="PT_V21_FIFO_POLL=1"
  pkill -f "pt_v21_nano .*--serve-fifo $IN" 2>/dev/null; sleep 1
  rm -f $IN; mkfifo $IN; : > $LOG
  env $E ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $IN > $LOG 2>&1 & S=$!
  i=0; while [ $i -lt 200 ]; do grep -q "serve" $LOG 2>/dev/null && break; sleep 0.1; i=$((i+1)); done
  sleep 2
  line="$arm:"; bad=0
  for k in $(seq 0 $N); do
    rm -f /tmp/tpa.out.npy
    c=$(grep -c PT_DONE $LOG)
    t0=$(date +%s.%N)
    timeout 30 sh -c "echo '$CASE /tmp/tpa.out.npy' > $IN"
    # 1 ms completion poll (identical in both arms, <= 1 ms of bias). Not `tail -f | grep -m`:
    # tail only notices grep has exited on its NEXT write, so with an idle log it sat out the
    # full timeout (every request read 60 s in the first version).
    w=0; while [ "$(grep -c PT_DONE $LOG)" -le "$c" ] && [ $w -lt 30000 ]; do sleep 0.001; w=$((w+1)); done
    t1=$(date +%s.%N)
    cmp -s /tmp/tpa.out.npy /tmp/tpa.ref.npy || bad=$((bad+1))
    [ $k = 0 ] && continue          # first request of a boot: first touch + clock ramp, untimed
    line="$line $(echo "$t1 $t0" | awk '{printf "%.4f", $1-$2}')"
  done
  echo "$line bytes_mismatch=$bad"
  timeout 5 sh -c "echo Q > $IN" 2>/dev/null; sleep 1; kill $S 2>/dev/null; wait $S 2>/dev/null
done
rm -f $IN
