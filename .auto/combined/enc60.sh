#!/bin/bash
# bash enc60.sh <0|1>: served 60 s encoder request after one small one; 1 = backbone resident too.
cd /home/picard/lfm3 || exit 1
E=/tmp/e60.enc.in; EL=/tmp/e60.enc.log; S=/tmp/e60.srv.in; SL=/tmp/e60.srv.log
pkill -x lfm3_enc_gpu; pkill -x lfm3_nanode; sleep 2
rm -f /tmp/e60.*; mkfifo $E $S; : > $EL; : > $SL
LFM_ENC_TIME=1 ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!; SP=
i=0; while [ $i -lt 120 ]; do grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
if [ "$1" = 1 ]; then
  ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 & SP=$!
  i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && break; sleep 0.5; i=$((i+1)); done
  timeout 300 ./lfm3_asr . /tmp/clip_zh.wav 8 --out-ids /tmp/e60.w.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL >/dev/null 2>&1
else
  timeout 300 ./lfm3_asr . /tmp/clip_zh.wav 8 --out-ids /tmp/e60.w.i32 --enc-in $E --enc-log $EL >/dev/null 2>&1
fi
echo "MemAvailable before big: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
n0=$(wc -l < $EL)
# stop after the encoder stage: tiny gen budget; without backbone the one-shot backbone would load, so kill at reply
( timeout 400 ./lfm3_asr . /home/picard/lfm3/gate_60s_zh_16k_mono.wav 1 --out-ids /tmp/e60.b.i32 ${SP:+--serve-in $S --serve-log $SL} --enc-in $E --enc-log $EL > /tmp/e60.b.log 2>&1 ) &
CP=$!
i=0; while [ $i -lt 700 ]; do grep -q "STAGE 6" /tmp/e60.b.log && break; sleep 0.1; i=$((i+1)); done
grep -E "STAGE 5" /tmp/e60.b.log
pkill -P $CP; kill $CP 2>/dev/null
echo "MemAvailable after big: $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB"
tail -n +$((n0+1)) $EL | grep -iE "wall|ENCODER total|ENC_DONE" | head -40
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; pkill -x lfm3_asr; rm -f $E $S
