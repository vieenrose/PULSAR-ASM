#!/bin/bash
# bash stall.sh [reps]: resident arm only; 6 short clips then the 60 s clip. On a stall (no CLIP DONE
# within LFM_SRV_WAIT) capture the backbone's process state before tearing down.
cd /home/picard/lfm3 || exit 1
REPS=${1:-3}
S=/tmp/st.srv.in; SL=/tmp/st.srv.log; E=/tmp/st.enc.in; EL=/tmp/st.enc.log
for r in $(seq 1 $REPS); do
  pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; sleep 2
  rm -f /tmp/st.*; mkfifo $S $E; : > $SL; : > $EL
  ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 & EP=$!
  ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run ${PFRUN:-32} > $SL 2>&1 & SP=$!
  i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
  # TTS=1: the PrimeTTS service resident as well (what the combined harness runs since PT_TURN=1),
  # one request after the warm-up and one right after the long turn.
  PI=/tmp/st.pt.in; PL=/tmp/st.pt.log; PP=
  pt_req() {
    local n; n=$(grep -c PT_DONE $PL); ( timeout 90 tail -n +1 -f $PL 2>/dev/null | grep -m1 -q "^PT_DONE $n " ) & local w=$!
    local a; a=$(date +%s.%N); echo "/home/picard/primetts/tests/golden_v21/T81 /tmp/st.tts.npy" > $PI; wait $w
    echo "$1 tts=$(echo "$(date +%s.%N) - $a" | bc | cut -c1-5)s $(grep "^PT_DONE $n " $PL | cut -d' ' -f3-) MemAvailable=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
  }
  if [ "${TTS:-0}" = 1 ]; then
    pkill -x pt_v21_nano; rm -f $PI; mkfifo $PI; : > $PL
    ( cd /home/picard/primetts && exec ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $PI ) > $PL 2>&1 & PP=$!
    i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $PL && break; sleep 0.5; i=$((i+1)); done
  fi
  for w in /tmp/clip_zh.wav /tmp/clip_zh.wav /tmp/clip_en.wav /home/picard/lfm3/cv_zh5.wav /home/picard/lfm3/cv_zh9.wav /home/picard/lfm3/cv_en5.wav /home/picard/lfm3/cv_en9.wav; do
    timeout 300 ./lfm3_asr . $w 96 --out-ids /tmp/st.s.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL >/dev/null 2>&1 || echo "rep $r SHORT_FAIL $w"
  done
  [ -n "$PP" ] && pt_req "rep $r after short turns:"
  # BALLOON_MB: hold that much touched anonymous memory during the 60 s turn (memory-pressure probe).
  BP=
  if [ "${BALLOON_MB:-0}" -gt 0 ]; then
    python3 -c "
import sys, time
b = bytearray(int(sys.argv[1]) << 20)
for i in range(0, len(b), 4096): b[i] = 1
print('BALLOON_READY', flush=True); time.sleep(600)" $BALLOON_MB > /tmp/st.balloon 2>&1 & BP=$!
    i=0; while [ $i -lt 60 ]; do grep -q BALLOON_READY /tmp/st.balloon && break; sleep 0.5; i=$((i+1)); done
    echo "balloon ${BALLOON_MB} MB held, MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
  fi
  t0=$(date +%s.%N)
  ( LFM_SRV_WAIT=${LFM_SRV_WAIT:-200} timeout 400 ./lfm3_asr . /home/picard/lfm3/gate_60s_zh_16k_mono.wav 96 --out-ids /tmp/st.b.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/st.b.log 2>&1; rc=$?
    # the TTS request goes out with no gap after the turn (a 1 s poll here lets the GPU clock start to park)
    if [ -n "$PP" ] && [ $rc = 0 ]; then pt_req "rep $r right after the 60 s turn:" > /tmp/st.ttsafter; pt_req "rep $r second request:" >> /tmp/st.ttsafter; fi
    echo $rc > /tmp/st.rc ) &
  # sample MemAvailable + srv log size while it runs
  lo=99999; while [ ! -f /tmp/st.rc ]; do
    m=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ $m -lt $lo ] && lo=$m
    el=$(echo "$(date +%s.%N) - $t0" | bc | cut -d. -f1)
    if [ "${el:-0}" -ge 150 ] && [ ! -f /tmp/st.cap ]; then
      { echo "--- capture at ${el}s"; ps -o pid,stat,pcpu,rss,vsz,etime,comm -p $SP,$EP; grep -E "State|VmRSS|VmSwap|Threads" /proc/$SP/status
        top -H -b -n 1 -p $SP | sed -n 7,14p; echo "srv log lines: $(wc -l < $SL)"; tail -3 $SL | cut -c1-160
        for t in /proc/$SP/task/*; do echo "$(cat $t/comm) $(awk '{print $3}' $t/stat) wchan=$(cat $t/wchan 2>/dev/null)"; done | sort | uniq -c
        grep -E "MemFree|MemAvailable|SwapFree|SwapTotal|NvMap|Shmem:" /proc/meminfo; cat /sys/devices/57000000.gpu/load 2>/dev/null; } > /tmp/st.cap 2>&1
    fi
    sleep 1
  done
  el=$(echo "$(date +%s.%N) - $t0" | bc)
  [ -n "$BP" ] && kill $BP 2>/dev/null
  echo "rep $r: rc=$(cat /tmp/st.rc) wall=${el}s ids=$(( $(stat -c%s /tmp/st.b.i32 2>/dev/null || echo 0) / 4 )) md5=$(md5sum < /tmp/st.b.i32 2>/dev/null | cut -c1-12) memlow=${lo}MB $(grep -oE 'encoder reply  \(\+[0-9.]+ s\)' /tmp/st.b.log) $(grep -oE 'prompt 1396 ids in chunks of [0-9]+, [0-9]+ ms' $SL)"
  [ -f /tmp/st.ttsafter ] && cat /tmp/st.ttsafter
  [ -n "$PP" ] && { timeout 5 sh -c "echo Q > $PI"; kill $PP 2>/dev/null; rm -f $PI; }
  [ -f /tmp/st.cap ] && [ "$(cat /tmp/st.rc)" != 0 ] && { cat /tmp/st.cap; break; }
done
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; rm -f $S $E
