#!/bin/bash
# Localize the resident-encoder failure on a 60 s clip (found by generalize.sh): the served
# request writes proj_out = all zeros, while the same clip through the one-shot path yields 96
# ids. Two suspects, both introduced by the resident architecture, both testable without
# changing code: (a) the grow-only device buffer pool of #263 (a slot that has to grow may leave
# a caller holding the previous pointer -> the read buffer is the zeroed fresh allocation), and
# (b) plain memory pressure from the co-resident backbone (2.15 GB arena) versus the encoder's
# 60 s working set, which is ~20x the harness clips'.
# ARM = env for the encoder service. Safe by construction: only ever one backbone service plus
# one encoder service, never a cold backbone next to a live encoder.
cd /home/picard/lfm3 || exit 1
WAV="${WAV:-/home/picard/lfm3/gate_60s_zh_16k_mono.wav}"
ARM="${ARM:-LFM_ENC_POOL=1}"
S=/tmp/gp.srv.in; SL=/tmp/gp.srv.log; E=/tmp/gp.enc.in; EL=/tmp/gp.enc.log
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
rm -f /tmp/gp.*; mkfifo $S $E; : > $SL; : > $EL
env $ARM ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 128 > $SL 2>&1 &
SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
grep -q "CLIP_AWAIT 0" $SL || { echo BOOT_FAIL; tail -3 $SL; kill $SP $EP 2>/dev/null; exit 1; }
timeout 900 ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/gp.warm.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL >/dev/null 2>&1
timeout 1200 ./lfm3_asr . "$WAV" 96 --out-ids /tmp/gp.big.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/gp.big.log 2>&1
rc=$?
echo "ARM=$ARM rc=$rc ids=$(( $(stat -c%s /tmp/gp.big.i32 2>/dev/null || echo 0) / 4 ))"
grep -oE 'client total [0-9.]+' /tmp/gp.big.log | head -1
awk '/Tm=11003/,0' $EL | head -6
grep -icE "PLAUSIBILITY FAIL" $EL | sed 's/^/plaus_fail=/'
timeout 5 sh -c "echo Q > $S" 2>/dev/null; timeout 5 sh -c "echo Q > $E" 2>/dev/null
sleep 1; kill $SP $EP 2>/dev/null; wait 2>/dev/null; rm -f $S $E
