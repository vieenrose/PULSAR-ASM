#!/bin/bash
# Per-shape GEMM table of the resident audio encoder (LFM_ENC_TIME=3): which of the ~100
# launches of one ASR pass are anomalously slow. Not a benchmark: a diagnostic.
cd /home/picard/lfm3 || exit 1
S=/tmp/et.srv.in; SL=/tmp/et.srv.log; E=/tmp/et.enc.in; EL=/tmp/et.enc.log
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
rm -f $S $E; mkfifo $S $E; : > $SL; : > $EL
LFM_ENC_TIME=3 ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 &
SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
grep -q "CLIP_AWAIT 0" $SL || { echo BOOT_FAIL; tail -3 $SL; exit 1; }
for c in zh en zh; do
  timeout 300 ./lfm3_asr . /tmp/clip_$c.wav 96 --out-ids /tmp/et.$c.i32 \
      --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/et.$c.log 2>&1 || echo FAIL_$c
done
echo "=== per-shape table (last pass) ==="
grep -E "ENC (GEMM|SUM|STAGE)|shape|ms " $EL | tail -70
timeout 5 sh -c "echo Q > $S" 2>/dev/null; timeout 5 sh -c "echo Q > $E" 2>/dev/null
sleep 1; kill $SP $EP 2>/dev/null; wait 2>/dev/null; rm -f $S $E
