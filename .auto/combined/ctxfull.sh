#!/bin/bash
# bash ctxfull.sh: resident backbone, a 4090-id prompt asking for 20 generated ids (crosses the
# 4096-position context), then a normal turn.
cd /home/picard/lfm3 || exit 1
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; pkill -x pt_v21_nano; sleep 2
S=/tmp/cf.srv.in; SL=/tmp/cf.srv.log; rm -f $S; mkfifo $S; : > $SL
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 & SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && break; sleep 0.5; i=$((i+1)); done
python3 -c "print('-'); print(','.join(['124894']*4090)); print(-1); print(20); print(-1)" > /tmp/cf.spec
n=$(grep -c "DONE steps=" $SL); a=$(date +%s); echo /tmp/cf.spec > $S
i=0; while [ $i -lt 900 ]; do [ "$(grep -c "DONE steps=" $SL)" -gt "$n" ] && break; kill -0 $SP 2>/dev/null || break; sleep 1; i=$((i+1)); done
echo "long spec: $(( $(date +%s) - a )) s"; grep -E "CONTEXT|DONE steps=|ERROR|bad_alloc" $SL | cut -c1-140
timeout 400 ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/cf.zh.i32 --serve-in $S --serve-log $SL > /tmp/cf.zh.log 2>&1
echo "next turn rc=$? $(cmp -s /tmp/cf.zh.i32 /tmp/gen_golden_zh.i32 && echo GOLDEN_OK || echo GOLDEN_FAIL) alive=$(kill -0 $SP 2>/dev/null && echo yes || echo no) MemAvailable=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
timeout 5 sh -c "echo Q > $S"; sleep 1; kill $SP 2>/dev/null; rm -f $S
