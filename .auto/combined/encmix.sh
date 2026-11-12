#!/bin/bash
# Run from the engine dir on the board: bash encmix.sh <enc_only|both>
# Which condition breaks a served 60 s request: the service having served SMALLER requests first
# (pool slots that must grow), or the resident backbone's 2.15 GB arena? Uses the real client
# (lfm3_asr), not a hand-rolled fifo writer -- an earlier hand-rolled poll loop produced a
# false "hang" by racing its own writer.
#   MODE=enc_only  : encoder service only, backbone runs one-shot per turn (cold, 28 s each)
#   MODE=both      : encoder + backbone services (the shipping configuration)
cd /home/picard/lfm3 || exit 1
MODE=${1:-both}
SMALLS="/tmp/clip_zh.wav /tmp/clip_en.wav /home/picard/lfm3/cv_zh5.wav /home/picard/lfm3/cv_zh9.wav /home/picard/lfm3/cv_en5.wav /home/picard/lfm3/cv_en9.wav"
BIGW=/home/picard/lfm3/gate_60s_zh_16k_mono.wav
E=/tmp/em.enc.in; EL=/tmp/em.enc.log; S=/tmp/em.srv.in; SL=/tmp/em.srv.log
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
rm -f /tmp/em.*; mkfifo $E; : > $EL
EXTRA=""
if [ "$MODE" = both ]; then
  mkfifo $S; : > $SL
  ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 128 > $SL 2>&1 &
  SP=$!; i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && break; sleep 0.5; i=$((i+1)); done
  EXTRA="--serve-in $S --serve-log $SL"
fi
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!; i=0; while [ $i -lt 120 ]; do grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
for w in $SMALLS; do
  timeout 900 ./lfm3_asr . "$w" 96 --out-ids /tmp/em.s.i32 $EXTRA --enc-in $E --enc-log $EL >/dev/null 2>&1 || echo "FAIL_SMALL $w"
done
timeout 1200 ./lfm3_asr . "$BIGW" 96 --out-ids /tmp/em.big.i32 $EXTRA --enc-in $E --enc-log $EL > /tmp/em.big.log 2>&1
echo "MODE=$MODE big_rc=$? ids=$(( $(stat -c%s /tmp/em.big.i32 2>/dev/null || echo 0) / 4 ))"
grep -E "falling back|did not answer|failed:" /tmp/em.big.log | head -3
awk '/Tm=11003/,0' $EL | grep -E "nonzero|PLAUS|ENC_DONE|rc=" | tail -4
timeout 5 sh -c "echo Q > $E" 2>/dev/null; [ "$MODE" = both ] && timeout 5 sh -c "echo Q > $S" 2>/dev/null
sleep 1; kill $EP 2>/dev/null; [ -n "$SP" ] && kill $SP 2>/dev/null; wait 2>/dev/null; rm -f $E $S
