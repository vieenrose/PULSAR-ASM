#!/usr/bin/env bash
# .auto/combined/ab_poll.sh -- A/B of the hand-off poll intervals on the resident ASR turn.
#
# One build; the arm is the environment the services AND the client run under, because two of the
# switches are read at service start:
#   new  default: backbone fifo read blocks, encoder service polls 2 ms, client waits poll 2 ms
#   old  LFM_FIFO_POLL=1 LFM_ENC_POLL_US=20000 LFM_POLL_US=20000 (the pre-change timings; the
#        client's encoder-reply wait was 10 ms before, so 'old' is slightly pessimistic there)
# Each boot serves one untimed warm-up turn, then N zh + en turns. Order defaults to new old old
# new (ABBA) so boot order and board drift cancel. ids are md5'd per turn: the change must be exact.
set -u
cd /home/picard/lfm3
N=${N:-3}; ORDER=${ORDER:-"new old old new"}
OLD="LFM_FIFO_POLL=1 LFM_ENC_POLL_US=20000 LFM_POLL_US=20000"
SRV_IN=/tmp/abp.srv.in; SRV_LOG=/tmp/abp.srv.log; ENC_IN=/tmp/abp.enc.in; ENC_LOG=/tmp/abp.enc.log
for arm in $ORDER; do
  E=""; [ "$arm" = old ] && E="$OLD"
  pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
  sleep 1
  i=0; while [ $i -lt 60 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
  rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
  env $E ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
  i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  env $E ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
  i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  [ $i -ge 240 ] && { echo "BOOT_FAIL $arm"; exit 1; }
  run() { env $E ./lfm3_asr . "$1" 96 --out-ids /tmp/abp.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
            --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/abp.log 2>&1; echo $?; }
  run /tmp/clip_zh.wav > /dev/null
  line="$arm:"
  for t in $(seq 1 $N); do
    for c in zh en; do
      rc=$(run /tmp/clip_$c.wav)
      w=$(grep -o "client total [0-9.]*" /tmp/abp.log | awk '{print $3}')
      line="$line $c=${w}(rc$rc,$(md5sum /tmp/abp.i32 | cut -c1-8))"
    done
  done
  echo "$line"
  for f in $SRV_IN $ENC_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
  sleep 1; kill $SRV $ENC 2>/dev/null; wait $SRV $ENC 2>/dev/null
done
rm -f $SRV_IN $ENC_IN
