#!/bin/bash
# bash guard.sh: three services resident; long clip with the guard forced to trip (floor from $1),
# then a short turn to show the encoder service is still serving.
cd /home/picard/lfm3 || exit 1
S=/tmp/gd.srv.in; SL=/tmp/gd.srv.log; E=/tmp/gd.enc.in; EL=/tmp/gd.enc.log; PI=/tmp/gd.pt.in; PL=/tmp/gd.pt.log
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; pkill -x pt_v21_nano; sleep 2
rm -f /tmp/gd.*; mkfifo $S $E $PI; : > $SL; : > $EL; : > $PL
LFM_ENC_MEM=1 LFM_ENC_MEMFLOOR_MB=$1 ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 & EP=$!
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 & SP=$!
( cd /home/picard/primetts && exec ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $PI ) > $PL 2>&1 & PP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && grep -q "ENC_AWAIT 0" $EL && grep -q "PT_AWAIT 0" $PL && break; sleep 0.5; i=$((i+1)); done
t() { local a=$(date +%s.%N); LFM_ENC_MEMFLOOR_MB=$1 timeout 400 ./lfm3_asr . $2 96 --out-ids /tmp/gd.$3.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/gd.$3.log 2>&1; local rc=$?
      echo "$3: rc=$rc wall=$(echo "$(date +%s.%N) - $a" | bc | cut -c1-6)s ids=$(( $(stat -c%s /tmp/gd.$3.i32 2>/dev/null || echo 0) / 4 )) md5=$(md5sum < /tmp/gd.$3.i32 2>/dev/null | cut -c1-12) MemAvailable=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"; }
t $1 /tmp/clip_zh.wav warm; t $1 /tmp/clip_zh.wav short1
t $1 /home/picard/lfm3/gate_60s_zh_16k_mono.wav long
grep -hE "MEMGUARD|lfm3_asr:" /tmp/gd.long.log | cut -c1-230 | tail -3
t $1 /tmp/clip_en.wav short2
echo "--- encoder service log:"; grep -E "MEM preflight|MEMGUARD|ENC_DONE|retiring" $EL | cut -c1-200 | tail -8
echo "encoder service alive: $(kill -0 $EP 2>/dev/null && echo yes || echo no)"
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; pkill -x pt_v21_nano; rm -f $S $E $PI
