#!/bin/bash
# Same-service-session A/B of the prefill GEMV occupancy arm: LFM_PF_TLGS=9 (shipping,
# 2 blocks/SM at 64 regs / 32 KB shared) vs 93 (q8_partial_TLg3<8>: 3 blocks/SM, 32 regs,
# 16 KB shared, STACK 144). One boot per arm, cold blob, untimed warm-up, 3 timed turns.
cd /home/picard/lfm3 || exit 1
BIN=${BIN:-./lfm3_nanode_nb}
arm() {  # arm <tag> <tlgs>
  local tag=$1 tg=$2
  local S=/tmp/t3.$tag.srv.in SL=/tmp/t3.$tag.srv.log E=/tmp/t3.$tag.enc.in EL=/tmp/t3.$tag.enc.log
  pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
  rm -f $S $E; mkfifo $S $E; : > $SL; : > $EL
  python3 -c "
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-enc.lfm3q8','tok.bin','mel.bin']:
    try:
        fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError: pass
" 2>/dev/null
  LFM_ENC_TIME=0 ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
  local EP=$!
  env LFM_PF_TLGS=${TLGS:-9} $2 $BIN lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 &
  local SP=$!
  local i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL 2>/dev/null && grep -q "ENC_AWAIT 0" $EL 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  grep -q "CLIP_AWAIT 0" $SL 2>/dev/null || { echo "$tag BOOT_FAIL"; tail -3 $SL; return; }
  ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/t3.$tag.w.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL >/dev/null 2>&1
  local O=""
  for c in zh en zh; do
    local n0=$(grep -c 'PF4-RUN: prompt' $SL)
    local t0=$(date +%s.%N)
    timeout 300 ./lfm3_asr . /tmp/clip_$c.wav 96 --out-ids /tmp/t3.$tag.$c.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/t3.$tag.$c.log 2>&1
    local t1=$(date +%s.%N)
    local pf=$(awk -v n=$n0 '/PF4-RUN: prompt/{c++; if(c>n) print}' $SL | sed 's/.*chunks of [0-9]*, //;s/ ms.*//' | tr '\n' '/')
    O="$O ${c}:$(python3 -c "print('%.3f'%($t1-$t0))"))/$pf"
  done
  echo "ARM $tag TLGS=$tg $O ids $(md5sum < /tmp/t3.$tag.zh.i32 | cut -c1-8)"
  timeout 5 sh -c "echo Q > $S" 2>/dev/null; timeout 5 sh -c "echo Q > $E" 2>/dev/null
  sleep 1; kill $SP $EP 2>/dev/null; wait 2>/dev/null; rm -f $S $E
}
for p in 1 2; do
  if [ $p = 1 ]; then arm A$((p)) "$ARM_A"; arm B$((p)) "$ARM_B"; else arm B$((p)) "$ARM_B"; arm A$((p)) "$ARM_A"; fi
done
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; true
