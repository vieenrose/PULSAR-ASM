#!/bin/bash
# bash soak.sh [NTURNS]  (on the board: /home/picard/lfm3)
#
# Long-conversation soak of the three resident services. The harness measures TWO turns, so it
# cannot see anything that drifts with turn count: a per-turn leak, a pool that grows, a service
# that degrades, or per-request state that breaks on turn N. (Two real defects were of exactly
# that class: the encoder's first-request-sized static (#273) and the grow-only pool that pinned
# the board (#275).) This runs N alternating zh/en turns through the resident path -- the
# configuration the harness measures -- and records per turn: ASR wall, TTS wall, MemAvailable,
# the three services' VmRSS, and requires the transcript to equal its frozen golden EVERY turn.
cd /home/picard/lfm3 || exit 1
N=${1:-24}
PTD=/home/picard/primetts
pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
pkill -x pt_v21_nano 2>/dev/null
i=0; while [ $i -lt 60 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
SRV_IN=/tmp/sk.srv.in; SRV_LOG=/tmp/sk.srv.log
ENC_IN=/tmp/sk.enc.in; ENC_LOG=/tmp/sk.enc.log
PT_IN=/tmp/sk.pt.in;   PT_LOG=/tmp/sk.pt.log
rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG $PT_IN $PT_LOG
mkfifo $SRV_IN $ENC_IN $PT_IN; : > $SRV_LOG; : > $ENC_LOG; : > $PT_LOG
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
( cd $PTD && exec ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $PT_IN ) > $PT_LOG 2>&1 & PT=$!
i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $PT_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
[ $i -ge 120 ] && { echo "SOAK_BOOT_FAIL (see $SRV_LOG $ENC_LOG $PT_LOG)"; exit 1; }
echo "BOOT mem=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB"
# One untimed turn, as the harness does: it absorbs the first-request paths and the clock ramp so
# the measured turns start from the state a deployed service is in on the user's second question.
./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/sk.w.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
  --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/sk.w.log 2>&1 || echo "WARMUP_FAIL"
srvrss() { p=$(pgrep -f "$1" | head -1); if [ -n "$p" ]; then awk '/VmRSS/{print int($2/1024)}' /proc/$p/status; else echo -1; fi; }
for t in $(seq 1 $N); do
  if [ $((t % 2)) = 1 ]; then lang=zh; else lang=en; fi
  rm -f /tmp/sk.i32
  a=$(date +%s.%N)
  timeout 400 ./lfm3_asr . /tmp/clip_$lang.wav 96 --out-ids /tmp/sk.i32 \
    --serve-in $SRV_IN --serve-log $SRV_LOG --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/sk.log 2>&1
  rc=$?
  b=$(date +%s.%N)
  npt=$(grep -c PT_DONE $PT_LOG 2>/dev/null); [ -n "$npt" ] || npt=0
  ( timeout 60 tail -n +1 -f $PT_LOG 2>/dev/null | grep -m1 -q "^PT_DONE $npt " ) & w=$!
  ta=$(date +%s.%N)
  timeout 60 sh -c "echo '$PTD/tests/golden_v21/T81 /tmp/sk.tts.npy' > $PT_IN"
  wait $w
  tb=$(date +%s.%N)
  ids=MISSING; [ -s /tmp/sk.i32 ] && { cmp -s /tmp/sk.i32 /tmp/gen_golden_$lang.i32 && ids=GOLDEN || ids=MISMATCH; }
  echo "TURN $t $lang rc=$rc asr=$(python3 -c "print('%.3f'%($b-$a))") tts=$(python3 -c "print('%.3f'%($tb-$ta))") ids=$ids sndr=$(grep "^PT_DONE $npt " $PT_LOG | tail -1 | grep -o 'sndr=[0-9.]*' | cut -d= -f2) mem=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) rss_bb=$(srvrss 'lfm3_nanode.*--clip-fifo') rss_enc=$(srvrss 'lfm3_enc_gpu.*--serve-fifo') rss_tts=$(srvrss 'pt_v21_nano')"
done
echo "ENC_REQS $(grep -c ENC_DONE $ENC_LOG) PT_REQS $(grep -c PT_DONE $PT_LOG)"
for f in $SRV_IN $ENC_IN $PT_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
sleep 1; kill $SRV $ENC $PT 2>/dev/null; rm -f $SRV_IN $ENC_IN $PT_IN
