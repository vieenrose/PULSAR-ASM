#!/bin/bash
# bash asr_gap_ab.sh [gap_s] [rounds]  (on the board: /home/picard/lfm3)
#
# Does an ASR turn that starts after the user has been silent pay the parked-clock penalty? The
# harness cannot see it: it runs the timed turns back-to-back after an untimed warm-up turn, so the
# GPU is at max when the turn starts. In a conversation the GPU is idle while the user speaks, and
# nvhost_podgov parks it within ~2 s of idle (the TTS showed 0.48 -> 1.0-1.7 s in that regime,
# run #277/#285). Protocol: boot the two ASR services, one untimed warm-up turn, then for each
# round alternate {sleep GAP, timed turn} and {timed turn immediately}, so both arms share one
# service session and the order cancels drift. Prints the turn wall and the GPU clock read just
# before the request.
cd /home/picard/lfm3 || exit 1
G=${1:-10}; R=${2:-2}
SRV_IN=/tmp/ag.srv.in; SRV_LOG=/tmp/ag.srv.log; ENC_IN=/tmp/ag.enc.in; ENC_LOG=/tmp/ag.enc.log
pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
i=0; while [ $i -lt 60 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
[ $i -ge 240 ] && { echo "BOOT_FAIL"; exit 1; }
turn() {  # $1 = label, $2 = pre-sleep
  [ "$2" != 0 ] && sleep $2
  f=$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq 2>/dev/null)
  a=$(date +%s.%N)
  timeout 400 ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/ag.i32 \
    --serve-in $SRV_IN --serve-log $SRV_LOG --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/ag.log 2>&1
  rc=$?; b=$(date +%s.%N)
  g=$(cmp -s /tmp/ag.i32 /tmp/gen_golden_zh.i32 && echo GOLDEN || echo MISMATCH)
  echo "TURN $1 rc=$rc ids=$g wall=$(python3 -c "print('%.3f'%($b-$a))") clk=$((f/1000000))MHz"
}
turn warmup 0
for r in $(seq 1 $R); do
  turn "r$r-gap" $G
  turn "r$r-now" 0
done
echo "ENC_RAMP_LINES $(grep -c 'clock parked' $ENC_LOG 2>/dev/null) BB_RAMP_LINES $(grep -c 'clock parked' $SRV_LOG 2>/dev/null)"
for f in $SRV_IN $ENC_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
sleep 1; kill $SRV $ENC 2>/dev/null; rm -f $SRV_IN $ENC_IN
