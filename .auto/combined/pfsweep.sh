#!/bin/bash
# Sweep --prefill-run T (the batched-prefill chunk size) in the resident ASR service.
# One service boot per arm, cold blob, untimed warm-up turn (builds the prefix cache),
# then timed turns. Prints one line per arm: prefill ms per turn, turn walls, ids md5.
# usage: pfsweep.sh <tag> <PFRUN> [long]
cd /home/picard/lfm3 || exit 1
TAG=$1; PF=$2; LONG=$3
S=/tmp/pf.$TAG.srv.in; SL=/tmp/pf.$TAG.srv.log
E=/tmp/pf.$TAG.enc.in; EL=/tmp/pf.$TAG.enc.log
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
rm -f $S $E; mkfifo $S $E; : > $SL; : > $EL
python3 -c "
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-enc.lfm3q8','tok.bin','mel.bin']:
    try:
        fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError: pass
" 2>/dev/null
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run $PF > $SL 2>&1 &
SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL 2>/dev/null && grep -q "ENC_AWAIT 0" $EL 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
grep -q "CLIP_AWAIT 0" $SL 2>/dev/null || { echo "PF$TAG BOOT_FAIL"; tail -3 $SL; exit 1; }

run() {  # run <clip> <name> -> "wall ids_md5 prefill_ms"
  local clip=$1 nm=$2 npf0 npl0
  npf0=$(grep -c 'PF4-RUN: prompt' $SL)
  local t0=$(date +%s.%N)
  timeout 900 ./lfm3_asr . "$clip" 96 --out-ids /tmp/pf.$TAG.$nm.i32 \
      --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/pf.$TAG.$nm.log 2>&1
  local rc=$?
  local t1=$(date +%s.%N)
  local pf=$(awk -v n=$npf0 '/PF4-RUN: prompt/{c++; if(c>n) print}' $SL | sed 's/.*chunks of [0-9]*, //;s/ ms.*//' | tr '\n' '/')
  echo "$(python3 -c "print('%.3f'%($t1-$t0))")) $(md5sum < /tmp/pf.$TAG.$nm.i32 | cut -c1-8) pf=$pf rc=$rc"
}

run /tmp/clip_zh.wav w1 > /dev/null          # untimed warm-up: prefix cache + clock ramp
OUT=""
for c in zh en zh en; do OUT="$OUT $c=$(run /tmp/clip_$c.wav $c)"; done
[ "$LONG" = long ] && OUT="$OUT L60=$(run gate_60s_16k_mono.wav l60)"
echo "PF$TAG PFRUN=$PF$OUT"
timeout 5 sh -c "echo Q > $S" 2>/dev/null; timeout 5 sh -c "echo Q > $E" 2>/dev/null
sleep 1; kill $SP $EP 2>/dev/null; wait 2>/dev/null; rm -f $S $E
