#!/usr/bin/env bash
# .auto/combined/ab_split.sh -- CPU row share of the prefill (LFM_CPUSPLIT), same build, ABBA arms
# (run #299; #313 adds arm 2 = split on with the twin x-tile reuse off). Each arm boots its own services, 1 warm-up turn per clip, then
# REPS x {zh, en}; prints the backbone's PF4-RUN prefill ms, the turn wall and the ids md5.
set -u
cd /home/picard/lfm3
REPS=${REPS:-3}; ORDER=${ORDER:-"0 1 1 0"}
SRV_IN=/tmp/as.srv.in; SRV_LOG=/tmp/as.srv.log; ENC_IN=/tmp/as.enc.in; ENC_LOG=/tmp/as.enc.log
pf() { grep "PF4-RUN: prompt" $SRV_LOG | tail -1 | grep -oE "[0-9]+ ms \(" | grep -oE "[0-9]+"; }
for A in $ORDER; do
  pkill -f "[l]fm3_nanode .*--clip-fifo"; pkill -f "[l]fm3_enc_gpu .*--serve-fifo"; pkill -f "[p]t_v21_nano .*--serve-fifo"; sleep 1
  rm -f $SRV_IN $ENC_IN; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
  SP=1; XC=1; case $A in 0) SP=0;; 2) SP=1; XC=0;; esac   # arm 2 = split on, twin reuse off
  LFM_CPUSPLIT=$SP LFM_CPUSPLIT_XCACHE=$XC ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
  i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG && break; kill -0 $SRV || break; sleep 0.5; i=$((i+1)); done
  ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
  i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG && break; sleep 0.5; i=$((i+1)); done
  turn() { ./lfm3_asr . /tmp/clip_$1.wav 96 --out-ids /tmp/as.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
             --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/as.log 2>&1; echo "rc=$?"; }
  turn zh >/dev/null; turn en >/dev/null
  for r in $(seq 1 $REPS); do for c in zh en; do rc=$(turn $c)
    echo "arm=$A $c $rc wall=$(grep -o 'client total [0-9.]*' /tmp/as.log | awk '{print $3}') prefill=$(pf) md5=$(md5sum /tmp/as.i32 | cut -c1-8)"
  done; done
  timeout 5 sh -c "echo Q > $SRV_IN"; wait $SRV; grep CPUSPLIT $SRV_LOG | tail -1
  kill $ENC 2>/dev/null; wait $ENC 2>/dev/null
done
rm -f $SRV_IN $ENC_IN
