#!/bin/bash
# Run from the engine dir on the board: bash encstage.sh <n_small_before_big>
# Localize the all-zero stage: boot the service WITHOUT --only proj_out so every stage's dump
# runs, then reproduce "small served pass, then the 60 s one" (encmix.sh showed the backbone is
# not needed). The first stage whose dump reads nonzero=0 is where the history bites.
cd /home/picard/lfm3 || exit 1
NSMALL=${1:-1}
E=/tmp/est.enc.in; EL=/tmp/est.enc.log
pkill -x lfm3_enc_gpu 2>/dev/null; sleep 1
rm -f /tmp/est.*; mkfifo $E; : > $EL
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E > $EL 2>&1 &
EP=$!; i=0; while [ $i -lt 120 ]; do grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
for k in $(seq 1 $NSMALL); do
  timeout 900 ./lfm3_asr . /tmp/clip_zh.wav 8 --out-ids /tmp/est.s.i32 --enc-in $E --enc-log $EL >/dev/null 2>&1 || echo FAIL_SMALL_$k
done
timeout 1200 ./lfm3_asr . /home/picard/lfm3/gate_60s_zh_16k_mono.wav 8 --out-ids /tmp/est.b.i32 --enc-in $E --enc-log $EL > /tmp/est.b.log 2>&1
echo "big ids=$(( $(stat -c%s /tmp/est.b.i32 2>/dev/null || echo 0) / 4 ))"
echo "=== first all-zero dumps in the whole session:"; grep -n "nonzero=0" $EL | head -8
echo "=== last pass (from its Tm line) stage dumps:"
awk '/Tm=11003/,0' $EL | grep -E "dump|PLAUS|ENC_DONE" | head -40
timeout 5 sh -c "echo Q > $E" 2>/dev/null; sleep 1; kill $EP 2>/dev/null; wait 2>/dev/null; rm -f $E
