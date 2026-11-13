#!/usr/bin/env bash
# .auto/combined/ab_free.sh -- order-balanced A/B of the client's exit-path teardown.
#
# Two client binaries built from the same tree, identical except for one line (ltok_free(&tok) at
# the end of main): lfm3_asr.free frees the ~128k-entry vocabulary on the way out, lfm3_asr.nofree
# lets the kernel reclaim it. Both carry the same stage probes, so the only difference is the free.
# Both run against ONE pair of resident services in one session, alternating arms and reversing the
# order every round, and the generated ids are md5-compared per clip: the change must be exact.
set -u
cd /home/picard/lfm3
T=${T:-3}
# ARMS picks the two binaries to compare (default: the teardown A/B). Reverse the order on odd
# rounds so no arm is systematically first after the boot.
ARMS=${ARMS:-"free nofree"}
SRV_IN=/tmp/abf.srv.in; SRV_LOG=/tmp/abf.srv.log; ENC_IN=/tmp/abf.enc.in; ENC_LOG=/tmp/abf.enc.log
pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
i=0; while [ $i -lt 60 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
[ $i -ge 240 ] && { echo "BOOT_FAIL"; exit 1; }

turn() { # $1=arm $2=clip $3=tag
  local o=/tmp/abf.$3.i32; rm -f $o
  timeout 400 ./lfm3_asr.$1 . "$2" 96 --out-ids $o \
    --serve-in $SRV_IN --serve-log $SRV_LOG --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/abf.$3.log 2>&1
  local rc=$? w=$(grep -o "client total [0-9.]*" /tmp/abf.$3.log | awk '{print $3}')
  echo "  $1 $(basename $2) rc=$rc wall=${w}s md5=$(md5sum $o 2>/dev/null | cut -c1-8)"
}

./lfm3_asr.nofree . /tmp/clip_zh.wav 96 --out-ids /tmp/abf.w.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
  --enc-in $ENC_IN --enc-log $ENC_LOG > /dev/null 2>&1          # untimed: past the first-request paths
for r in $(seq 1 $T); do
  if [ $((r % 2)) = 1 ]; then A="$ARMS"; else A="$(echo $ARMS | tr ' ' '\n' | tac | tr '\n' ' ')"; fi
  for arm in $A; do turn $arm /tmp/clip_zh.wav "zh$r"; done
  for arm in $A; do turn $arm /tmp/clip_en.wav "en$r"; done
done
for f in $SRV_IN $ENC_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
sleep 1; kill $SRV $ENC 2>/dev/null; rm -f $SRV_IN $ENC_IN
