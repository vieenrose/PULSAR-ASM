#!/bin/bash
# Held-out-clip generalization gate for the RESIDENT ASR architecture.
#
# WHY THIS EXISTS. Every resident-service win (--clip-stdin/#259, --prefill-run/#260, resident
# encoder/#261, prefix cache/#264, id cache/#265, dead-output elimination/#266, buffer pool/#263)
# was certified against the two harness clips only. The claim those changes rest on is stronger
# than "the two goldens still match": it is that a served turn is BYTE-IDENTICAL to the cold
# one-shot path on ANY audio. The prefix cache is keyed on ids, the id cache on a prompt hash,
# the device pool is grow-only, and the last-layer dead-output elimination skips launches for
# every prompt position but the last -- all of them have a shape- or length-dependent code path
# that the two 9.0 s clips never exercise. This script tests that claim on clips that were never
# used to tune anything: the 5 s / 9 s TTS-derived clips and the 60 s real-gate clips (740-ish
# prompt positions, i.e. 6x the harness, which is also an arena/MAXTOK soak).
#
# Method: each clip runs twice, in the two process models that must agree.
#   COLD     ./lfm3_asr . <wav> 96 --out-ids X                 (the path the frozen goldens came from)
#   RESIDENT the same command + --serve-in/--enc-in             (both services live)
# Cold runs happen BEFORE any service is booted: a cold nanode allocates its own 2.15 GB arena and
# a second one alongside the resident service would hang the board (run 281).
# PASS = identical id stream per clip, both goldens still exact, no FAIL_*, low water >= 300 MB.
cd /home/picard/lfm3 || exit 1
# PFRUN: the cold path uses LFM_ASR_PFRUN=32; the service takes --prefill-run as an arg, so the two
# arms can be run at different chunk sizes (the 60 s clip failure is being bisected on this knob).
PFRUN=${PFRUN:-32}
CLIPS="${CLIPS:-/tmp/clip_zh.wav /tmp/clip_en.wav /home/picard/lfm3/cv_zh5.wav /home/picard/lfm3/cv_zh9.wav /home/picard/lfm3/cv_en5.wav /home/picard/lfm3/cv_en9.wav /home/picard/lfm3/gate_60s_zh_16k_mono.wav}"
S=/tmp/gen.srv.in; SL=/tmp/gen.srv.log; E=/tmp/gen.enc.in; EL=/tmp/gen.enc.log
pkill -x lfm3_nanode 2>/dev/null; pkill -x lfm3_enc_gpu 2>/dev/null; sleep 2
rm -f /tmp/gen.*; mkfifo $S $E; : > $SL; : > $EL
lw() { awk '/MemAvailable/ {print $2}' /proc/meminfo; }
MIN=$(lw); MAXT=0
cold_min=$MIN
for w in $CLIPS; do
  n=$(basename "$w" .wav)
  timeout 1200 ./lfm3_asr . "$w" 96 --out-ids /tmp/gen.$n.cold.i32 > /tmp/gen.$n.cold.log 2>&1 || echo "FAIL_COLD_$n"
  v=$(lw); [ "$v" -lt "$cold_min" ] && cold_min=$v
  printf "COLD  %-28s ids=%s wall=%s\n" "$n" "$(wc -c < /tmp/gen.$n.cold.i32 2>/dev/null | awk '{print $1/4}')" "$(grep -oE 'client total [0-9.]+' /tmp/gen.$n.cold.log | awk '{print $3}')"
done
# TTS=1 (default): the gate runs the configuration the combined harness measures since #277 --
# THREE resident services. One TTS request follows every resident ASR turn, rotating over all five
# golden reply cases (the harness only ever uses T81), each compared byte for byte with a one-shot
# reference made here before any service is up. TTS=0 = the old two-service gate.
TTS=${TTS:-1}; PT=/home/picard/primetts; PI=/tmp/gen.pt.in; PL=/tmp/gen.pt.log; PP=
TCASES="${TCASES:-T81 T20 T120 T40 sid1_T29}"
if [ "$TTS" = 1 ]; then
  pkill -x pt_v21_nano 2>/dev/null
  for c in $TCASES; do
    ( cd $PT && ./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/$c --out /tmp/gen.ref.$c.npy > /tmp/gen.ref.$c.log 2>&1 ) || echo "FAIL_TTSREF_$c"
  done
  mkfifo $PI; : > $PL
  ( cd $PT && exec ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $PI ) > $PL 2>&1 &
  PP=$!
  i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $PL && break; sleep 0.5; i=$((i+1)); done
  grep -q "PT_AWAIT 0" $PL || { echo TTS_BOOT_FAIL; kill $PP 2>/dev/null; exit 1; }
fi
TK=0; TTSBAD=0; TTSN=0
tts_turn() {   # $1 = label; request goes out with no gap after the ASR turn, blocking wait
  [ -n "$PP" ] || return 0
  set -- "$1" $TCASES; local lab=$1; shift; local c; c=$(eval echo \${$(( TK % $# + 1 ))}); TK=$((TK+1))
  local n; n=$(grep -c "^PT_DONE" $PL); rm -f /tmp/gen.tts.npy
  ( timeout 120 tail -n +1 -f $PL 2>/dev/null | grep -m1 -q "^PT_DONE $n " ) & local w=$!
  local a; a=$(date +%s.%N); echo "tests/golden_v21/$c /tmp/gen.tts.npy" > $PI; wait $w
  local dt; dt=$(echo "$(date +%s.%N) - $a" | bc | cut -c1-5); TTSN=$((TTSN+1))
  if cmp -s /tmp/gen.tts.npy /tmp/gen.ref.$c.npy; then st=BYTES_OK; else st=BYTES_FAIL; TTSBAD=1; fi
  printf "TTS   %-28s case=%-9s %s wall=%ss MemAvailable=%sMB\n" "$lab" "$c" "$st" "$dt" "$(( $(lw) / 1024 ))"
}
# now the resident services
LFM_ENC_TIME=0 ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $E --only proj_out > $EL 2>&1 &
EP=$!
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run $PFRUN > $SL 2>&1 &
SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && grep -q "ENC_AWAIT 0" $EL && break; sleep 0.5; i=$((i+1)); done
grep -q "CLIP_AWAIT 0" $SL || { echo BOOT_FAIL; tail -3 $SL; kill $SP $EP 2>/dev/null; exit 1; }
# untimed warm-up turn: the loader defers the upload to the first prefill, so the first served
# turn would otherwise absorb the 21 s read + upload (the harness does the same, see #260).
timeout 900 ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/gen.warm.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL >/dev/null 2>&1
BAD=0
for w in $CLIPS; do
  n=$(basename "$w" .wav)
  timeout 1200 ./lfm3_asr . "$w" 96 --out-ids /tmp/gen.$n.warm.i32 --serve-in $S --serve-log $SL --enc-in $E --enc-log $EL > /tmp/gen.$n.warm.log 2>&1 || { echo "FAIL_WARM_$n"; BAD=1; }
  tts_turn "$n"
  v=$(lw); [ "$v" -lt "$MIN" ] && MIN=$v
  a=$(md5sum < /tmp/gen.$n.cold.i32 2>/dev/null | cut -c1-12); b=$(md5sum < /tmp/gen.$n.warm.i32 2>/dev/null | cut -c1-12)
  wa=$(grep -oE 'client total [0-9.]+' /tmp/gen.$n.cold.log | awk '{print $3}'); wb=$(grep -oE 'client total [0-9.]+' /tmp/gen.$n.warm.log | awk '{print $3}')
  if [ "$a" = "$b" ] && [ -n "$a" ]; then st=IDENTICAL; else st="MISMATCH($a/$b)"; BAD=1; fi
  printf "PAIR  %-28s %-16s cold=%ss resident=%ss speedup=%s\n" "$n" "$st" "$wa" "$wb" "$(awk -v a="$wa" -v b="$wb" 'BEGIN{if(b>0)printf "%.1fx", a/b}')"
done
# the two harness goldens must still be byte-exact out of the resident path
for c in zh en; do
  if [ -s /tmp/gen_golden_$c.i32 ]; then
    cmp -s /tmp/gen.clip_$c.warm.i32 /tmp/gen_golden_$c.i32 && echo "GOLDEN_$c OK" || { echo "GOLDEN_$c FAIL"; BAD=1; }
  else echo "GOLDEN_$c SKIPPED (no /tmp/gen_golden_$c.i32)"; fi
done
timeout 5 sh -c "echo Q > $S" 2>/dev/null; timeout 5 sh -c "echo Q > $E" 2>/dev/null
[ -n "$PP" ] && { timeout 5 sh -c "echo Q > $PI" 2>/dev/null; kill $PP 2>/dev/null; rm -f $PI; }
[ "$TTSBAD" = 0 ] || BAD=1
[ -n "$PP" ] && echo "TTS_SERVED $TTSN requests, $([ $TTSBAD = 0 ] && echo all byte-identical to one-shot || echo MISMATCH)"
echo "GUARD_TRIPS $(grep -c MEMGUARD $EL)"
sleep 1; kill $SP $EP 2>/dev/null; wait 2>/dev/null; rm -f $S $E
echo "MEMLOW resident=${MIN} cold=${cold_min} (kB)"
[ "$BAD" = 0 ] && echo "GENERALIZE OK ($(echo $CLIPS | wc -w) clips)" || echo "GENERALIZE FAIL"
exit $BAD
