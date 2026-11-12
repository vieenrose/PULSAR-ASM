#!/bin/bash
# .auto/combined/measure.sh -- combined conversation RTF: LFM3 S2T (speech -> text) plus
# PrimeTTS v2.1 (text -> speech), both on the Jetson Nano (picard).
#
# A "turn" is one conversational exchange:
#     user audio (9.0 s, zh or en)  --LFM3 ASR-->  transcript
#     reply audio (3.616 s)         <--PrimeTTS--  reply ids (T81)
#
# Primary metric (lower is better):
#     combined_rtf = (asr_wall_s + tts_wall_s) / (audio_in_s + audio_out_s)
# i.e. wall time per second of audio the system has to handle, both directions. Averaged
# over the zh and en turns. Secondaries: s2t_rtf (asr_wall / audio_in), tts_rtf
# (tts_wall / audio_out), reply_rtf ((asr+tts) / audio_in: the "answer within the length of
# the user's utterance" number), and ttfa_ms (time to first audio byte of the reply).
#
# ASR mode: ASR_MODE=cold (default) runs the shipped one-shot CLI, which pays the whole
# 1.9 GB backbone load per turn; ASR_MODE=warm uses the resident service once it exists
# (.auto/combined/warm.sh deliberately fails loudly until then, so a "warm" number can never be
# produced by accident).
#
# Quality gates (a run that fails them emits no METRIC lines):
#   - ASR transcript ids == the frozen goldens (.auto/combined/golden_<lang>.i32)
#   - PrimeTTS parity driver reports SNDR >= 60 dB on the reply case
set -uo pipefail

NANO="${NANO:-picard@100.84.151.116}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ADIR="$REPO/.auto/combined"
ASR_MODE="${ASR_MODE:-cold}"
LFM_DIR="${LFM_DIR:-/home/picard/lfm3}"
PT_DIR="${PT_DIR:-/home/picard/primetts}"
REPLY_CASE="${REPLY_CASE:-T81}"
CLIPS="${CLIPS:-zh:/tmp/clip_zh.wav en:/tmp/clip_en.wav}"   # lang:path
ASR_MAXTOK="${ASR_MAXTOK:-96}"

fail() { echo "measure: $1" >&2; exit 2; }

# ---- sync + build every binary this harness measures -----------------------------------------
# A gate is only worth what the binary under test is worth. This harness used to measure whatever
# happened to be on the board, and a stale lfm3_asr silently ignored the new --enc-in flags and
# turned a "resident encoder" run into a one-shot run with every gate still green -- #37's lesson
# (a gate that runs a stale binary is vacuous), re-learned the hard way. LFM_SKIP_BUILD=1 reuses
# what is there, for a quick repeat of an unchanged tree.
if [ "${LFM_SKIP_BUILD:-0}" != 1 ]; then
  SRC="$REPO/pulsar_arm/asm"
  scp -q "$SRC"/lfm3_asr.c "$SRC"/lfm3_tok.c "$SRC"/lfm3_tok.h "$SRC"/lfm3_mel.c "$SRC"/lfm3_mel.S \
       "$SRC"/lfm3_wav.c "$SRC"/lfm3_nanode.cu "$SRC"/lfm3_qlow.cuh "$SRC"/lfm3_enc_gpu.cu \
       "$NANO:$LFM_DIR/" \
       || fail "cannot sync sources to $NANO"
  BUILD=$(ssh "$NANO" "LFM_DIR='$LFM_DIR' bash -s" <<'BUILDEOF' 2>&1 | grep -v setlocale
set -u
cd "$LFM_DIR" || { echo "BUILD_FAIL no_directory"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
ok=1
# rm -f first: nvcc exits 0 on a link failure and leaves the PREVIOUS binary in place, so a
# "test -x" after the build proves nothing unless the old one was removed.
build_cc() { rm -f "$1.new"; gcc -O2 -std=gnu11 -pthread -o "$1.new" $2 -I. -lm >"/tmp/bf_$1.log" 2>&1 \
    && mv -f "$1.new" "$1" && echo "BUILT $1" \
    || { echo "BUILD_FAIL $1"; grep -E "error" "/tmp/bf_$1.log" | head -4; ok=0; }; }
build_cu() { rm -f "$1.new"; nvcc $3 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
    -Xptxas -O3,-dlcm=ca -o "$1.new" $2 -lz >"/tmp/bf_$1.log" 2>&1 \
    && mv -f "$1.new" "$1" && echo "BUILT $1" \
    || { echo "BUILD_FAIL $1"; grep -iE "error" "/tmp/bf_$1.log" | head -4; ok=0; }; }
build_cc lfm3_asr "lfm3_asr.c lfm3_tok.c"
build_cc lfm3_mel "lfm3_mel.c lfm3_mel.S"   # the mel primitives are AArch64 asm
build_cc lfm3_wav lfm3_wav.c
build_cu lfm3_nanode lfm3_nanode.cu ""
build_cu lfm3_enc_gpu lfm3_enc_gpu.cu "-std=c++11"
[ "$ok" = 1 ] && echo BUILD_ALL_OK
BUILDEOF
)
  echo "$BUILD" | grep -q BUILD_ALL_OK || { echo "$BUILD"; fail "the ASR/TTS-side binaries did not build from $REPO" ; }
  echo "  $BUILD" | tr '\n' ' '; echo
fi

# ---- Nano memory guard (the board hard-hangs when RAM is exhausted)
free_mb=$(ssh -o ConnectTimeout=8 "$NANO" 'free -m | sed -n 2p | tr -s " " | cut -d" " -f4' 2>/dev/null)
[ -n "$free_mb" ] || fail "cannot read free RAM on $NANO"
[ "$free_mb" -ge 60 ] || fail "Nano free RAM ${free_mb} MB < 60 MB guard"

[ "$ASR_MODE" = cold ] || [ "$ASR_MODE" = warm ] || fail "ASR_MODE=$ASR_MODE (cold | warm)"

# ---- fixture checks: goldens must exist and the clips must be the frozen ones
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  [ -s "$ADIR/golden_$lang.i32" ] || fail "missing $ADIR/golden_$lang.i32 (freeze it with: lfm3_asr --out-ids)"
done

OUT=$(ssh "$NANO" "CLIPS='$CLIPS' LFM_DIR='$LFM_DIR' PT_DIR='$PT_DIR' REPLY_CASE='$REPLY_CASE' ASR_MAXTOK='$ASR_MAXTOK' ASR_MODE='$ASR_MODE' PFRUN='${PFRUN:-32}' LFM_ENC_RESIDENT='${LFM_ENC_RESIDENT:-1}' PT_RESIDENT='${PT_RESIDENT:-1}' bash -s" <<'REMOTE' 2>&1 | grep -v setlocale
set -u
cd "$LFM_DIR" || exit 2
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  python3 -c "import wave,sys; w=wave.open(sys.argv[2]); print('CLIPSEC %s %.3f' % (sys.argv[1], w.getnframes()/w.getframerate()))" "$lang" "$clip"
done
SRV_IN=/tmp/cmb.srv.in; SRV_LOG=/tmp/cmb.srv.log
  # A service left behind by a crashed or interrupted run holds its ~2 GB device arena, and the
  # next boot then waits forever for memory that will never drain. Kill only OUR services (the
  # flags are specific to this harness) and wait for the arena to actually come back.
  pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
  i=0; while [ $i -lt 40 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
  echo "PREFREE $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB after killing stale services"
ENC_IN=/tmp/cmb.enc.in; ENC_LOG=/tmp/cmb.enc.log
PT_IN=/tmp/cmb.pt.in; PT_LOG=/tmp/cmb.pt.log
if [ "$ASR_MODE" = warm ]; then
  # The resident service: one lfm3_nanode --clip-stdin for the whole session. It pays the
  # 1.946 GB blob read (21.2 s of eMMC at 92 MB/s, device-bound) plus parse/CRC/upload once, and
  # prints CLIP_AWAIT n when it is ready for a turn. --clip-fifo (not a redirected stdin) is what
  # makes it a SERVICE rather than a batch: each ASR turn is its own client process, so when the
  # last writer closes the fifo the read side sees EOF -- the end of a turn's writer, not of the
  # session -- and the server reopens the path and waits for the next caller. Its load is
  # reported, never hidden: it is a boot cost, not a per-turn one.
  rm -f $SRV_IN $SRV_LOG; mkfifo $SRV_IN; : > $SRV_LOG
  # Cold boot, so SERVICELOAD is a real first-invocation number and comparable between arms
  # (the same posix_fadvise trick the S2T harness uses; a previous run or probe can leave the
  # 1.946 GB blob cached and make the load look 20 s faster than it is).
  python3 -c "
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-enc.lfm3q8','tok.bin','mel.bin']:
    try:
        fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
    except OSError: pass
" 2>/dev/null
  t0=$(date +%s.%N)
  # --prefill-run is the same flag the one-shot path passes (the prompt goes through the
  # backbone 32 positions at a time, one launch per GEMV sharing each weight load: 34.8 ms per
  # position against 87.5 ms unbatched, bit-identical per its own note). The service, not the
  # client, runs the prefill, so it is the service that has to be started with it.
  ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN ${PFRUN:+--prefill-run $PFRUN} > $SRV_LOG 2>&1 &
  SRV=$!
  i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null || { echo "SERVICE_FAIL (see $SRV_LOG)"; kill $SRV 2>/dev/null; exit 1; }
  t1=$(date +%s.%N)
  python3 -c "print('SERVICELOAD %.3f' % ($t1-$t0))"
  # Both services are up before the first turn; the encoder's own load is part of boot too.
  # The audio encoder resident too, for the same reason one level down: its process costs ~1.9 s
  # of CUDA init + blob parse/CRC/upload against a ~0.2 s pass (its own ENCODER total line says
  # 122 ms of GPU for this clip). The server takes one request line per turn
  # "<mel> <outdir> auto" on the fifo and answers "ENC_DONE n rc=.."; its project outputs are
  # byte-identical to the one-shot path (verified on proj_out.bin), and the ASR's own per-turn
  # encoder spawn stays as the ASR_MODE=cold arm.
  # LFM_ENC_RESIDENT=0 keeps the per-turn encoder spawn: the A/B control arm for the service.
  # ENCFLAGS must be set here, inside the warm block, and not next to the ENC_IN declaration --
  # an earlier draft put its `if` before `if [ "$ASR_MODE" = warm ]`, which swallowed that line.
  ENCFLAGS=""
  if [ "${LFM_ENC_RESIDENT:-1}" = 1 ]; then ENCFLAGS="--enc-in $ENC_IN --enc-log $ENC_LOG"; fi
  rm -f $ENC_IN $ENC_LOG; mkfifo $ENC_IN; : > $ENC_LOG
  if [ "${LFM_ENC_RESIDENT:-1}" = 1 ]; then
    ./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 &
    ENCS=$!
    i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
    grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null || { echo "ENCSERVICE_FAIL (see $ENC_LOG)"; kill $ENCS 2>/dev/null; exit 1; }
  fi
  # Untimed warm-up turn. 'Ready to accept a turn' is not the same as 'the backbone is resident':
  # with --prefill-run the loader parses the blob and defers the upload to the first prefill, so
  # without this the user's FIRST question absorbs the blob read and the upload. A deployed
  # service is started and poked before anyone talks to it, so the warm-up is part of boot here
  # and is reported as WARMUP rather than being hidden.
  first=${CLIPS%% *}; warmclip=${first#*:}
  t2=$(date +%s.%N)
  ./lfm3_asr . "$warmclip" "$ASR_MAXTOK" --out-ids /tmp/cmb.warm.i32 \
    --serve-in $SRV_IN --serve-log $SRV_LOG $ENCFLAGS > /tmp/cmb.warm.log 2>&1 || echo "SERVICE_FAIL warm-up"
  t3=$(date +%s.%N)
  python3 -c "print('WARMUP %.3f' % ($t3-$t2))"
fi
# MemAvailable low-water watcher: the board hard-hangs when RAM is exhausted (run 281), and three
# resident GPU processes now coexist with the page cache, so a run that quietly pushes the floor
# down must be visible in the log rather than discovered by a hang.
( LOW=/tmp/cmb.lowwater; : > $LOW
  while :; do awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo >> $LOW; sleep 0.5; done ) &
WATCH=$!
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  rm -f "/tmp/cmb.$lang.i32" "/tmp/cmb.$lang.log"
  t0=$(date +%s.%N)
  if [ "$ASR_MODE" = warm ]; then
    ./lfm3_asr . "$clip" "$ASR_MAXTOK" --out-ids "/tmp/cmb.$lang.i32" \
      --serve-in $SRV_IN --serve-log $SRV_LOG $ENCFLAGS > "/tmp/cmb.$lang.log" 2>&1 || { echo "ASR_FAIL $lang"; continue; }
  else
    ./lfm3_asr . "$clip" "$ASR_MAXTOK" --out-ids "/tmp/cmb.$lang.i32" > "/tmp/cmb.$lang.log" 2>&1 || { echo "ASR_FAIL $lang"; continue; }
  fi
  t1=$(date +%s.%N)
  python3 -c "print('ASR %s %.3f' % ('$lang', $t1-$t0))"
  tail -1 "/tmp/cmb.$lang.log" | grep -q TRANSCRIPT || echo "ASR_NOTE $lang: no transcript line"
done
if [ "$ASR_MODE" = warm ]; then
  echo "ENCREQ $(grep -c ENC_DONE $ENC_LOG 2>/dev/null)"
  echo "PTREQ $(grep -c PT_DONE $PT_LOG 2>/dev/null || echo 0)"
  for i in $(seq 1 40); do kill -0 $SRV 2>/dev/null || break; timeout 5 sh -c "echo Q > $SRV_IN" 2>/dev/null; sleep 0.25; done
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -f $SRV_IN
  # Opening a fifo for write blocks until a reader exists, so this must be guarded (and timed
  # out): in the LFM_ENC_RESIDENT=0 control arm no encoder service is running and a bare
  # `echo Q > $ENC_IN` hangs the whole run in teardown.
  if [ "${LFM_ENC_RESIDENT:-1}" = 1 ]; then timeout 5 sh -c "echo Q > $ENC_IN" 2>/dev/null; kill $ENCS 2>/dev/null; wait $ENCS 2>/dev/null; fi
  rm -f $ENC_IN
fi
kill $WATCH 2>/dev/null
echo "LOWWATER_MB $(sort -n /tmp/cmb.lowwater 2>/dev/null | head -1)"
cd "$PT_DIR" || exit 2
# The reference is ALWAYS the one-shot driver, run untimed: it is the thing the SNDR gate has
# always certified, and it is what the served waveform is compared against byte for byte.
rm -f /tmp/cmb.ref.npy /tmp/cmb.tts.npy
./build/pt_v21_nano --container build/v21.ptk --case "tests/golden_v21/$REPLY_CASE" \
  --out /tmp/cmb.ref.npy > /tmp/cmb.tts_ref.log 2>&1 || echo "TTS_REF_FAIL"
grep -o "SNDR=[0-9.]*" /tmp/cmb.tts_ref.log | tail -1 | cut -d= -f2 | sed 's/^/TTS_SNDRdB /'
if [ "${PT_RESIDENT:-1}" = 1 ] && [ "$ASR_MODE" = warm ]; then
  # Resident TTS: a one-shot process pays ~0.4 s of container record pass + CUDA context + weight
  # upload + clock ramp before it computes anything, and in a conversation that is per turn. One
  # server per session, one request line per turn ("<case dir> <out.npy>"), answered by
  # "PT_DONE n rc=.. sndr=..". Same run_case() as the one-shot, and the waveform is gated against
  # the reference above byte for byte, so the metric arm and the certified arm cannot drift apart.
  rm -f $PT_IN $PT_LOG; mkfifo $PT_IN; : > $PT_LOG
  ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $PT_IN > $PT_LOG 2>&1 &
  PTS=$!
  i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $PT_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  grep -q "PT_AWAIT 0" $PT_LOG 2>/dev/null || { echo "PTSERVICE_FAIL (see $PT_LOG)"; kill $PTS 2>/dev/null; exit 1; }
  # grep -c already prints 0, and a `|| echo 0` there would append a SECOND line to the
  # substitution: the pattern "PT_DONE <n>\n<m>" then acts as two alternates, one of which is a
  # bare space that matches every line of the log, so the wait loop returns in 0.017 s and the
  # TTS time measures nothing. (It did: tts_rtf 0.0047 for 3.6 s of audio.)
  npt=$(grep -c PT_DONE $PT_LOG 2>/dev/null); [ -n "$npt" ] || npt=0
  t0=$(date +%s.%N)
  timeout 60 sh -c "echo 'tests/golden_v21/$REPLY_CASE /tmp/cmb.tts.npy' > $PT_IN" || echo "PT_WRITE_FAIL"
  i=0; while [ $i -lt 240 ]; do grep -q "PT_DONE $npt " $PT_LOG 2>/dev/null && break; sleep 0.1; i=$((i+1)); done
  t1=$(date +%s.%N)
  python3 -c "print('TTS %.3f' % ($t1-$t0))"
  grep -E "^PT_DONE $npt " $PT_LOG | tail -1 | grep -o "sndr=[0-9.]*" | cut -d= -f2 | sed 's/^/TTS_SNDRdB /'
  if cmp -s /tmp/cmb.tts.npy /tmp/cmb.ref.npy; then echo "GATE_TTS_SERVE_BYTES_OK"; else echo "GATE_TTS_SERVE_BYTES_FAIL"; fi
  timeout 5 sh -c "echo Q > $PT_IN" 2>/dev/null; kill $PTS 2>/dev/null; wait $PTS 2>/dev/null; rm -f $PT_IN
else
  t0=$(date +%s.%N)
  ./build/pt_v21_nano --container build/v21.ptk --case "tests/golden_v21/$REPLY_CASE" > /tmp/cmb.tts.log 2>&1 || echo "TTS_FAIL"
  t1=$(date +%s.%N)
  python3 -c "print('TTS %.3f' % ($t1-$t0))"
fi
REMOTE
)
echo "$OUT" | grep -E "^(ASR|TTS|CLIPSEC|SERVICELOAD|WARMUP|ENCREQ|PTREQ|LOWWATER_MB|GATE_TTS_SERVE)" | sed 's/^/  /'
echo "$OUT" | grep -q "ASR_FAIL\|TTS_FAIL\|TTS_REF_FAIL\|PTSERVICE_FAIL\|PT_WRITE_FAIL\|SERVICE_FAIL\|ENCSERVICE_FAIL" && { echo "GATE FAIL: a stage failed"; echo "$OUT" | tail -20; exit 1; }

# ---- quality gates
ok=1
for spec in $CLIPS; do
  lang=${spec%%:*}
  scp -q "$NANO:/tmp/cmb.$lang.i32" "/tmp/cmb.$lang.i32" 2>/dev/null || { echo "GATE FAIL: no ids for $lang"; ok=0; continue; }
  if cmp -s "/tmp/cmb.$lang.i32" "$ADIR/golden_$lang.i32"; then
    echo "GATE_ASR_${lang}_OK"
  else
    echo "GATE_ASR_${lang}_FAIL (transcript differs from $ADIR/golden_$lang.i32)"; ok=0
  fi
done
if echo "$OUT" | grep -q "GATE_TTS_SERVE_BYTES_FAIL"; then
  echo "GATE_TTS_SERVE_BYTES_FAIL (the served waveform differs from the one-shot reference)"; ok=0
elif echo "$OUT" | grep -q "GATE_TTS_SERVE_BYTES_OK"; then echo "GATE_TTS_SERVE_BYTES_OK"; fi
sndr=$(echo "$OUT" | grep -o "TTS_SNDRdB [0-9.]*" | awk '{print $2}' | tail -1)
if [ -n "${sndr:-}" ] && awk -v s="$sndr" 'BEGIN{exit !(s>=60)}'; then echo "GATE_TTS_SNDR_OK ($sndr dB)"; else echo "GATE_TTS_SNDR_FAIL (${sndr:-none})"; ok=0; fi
[ "$ok" = 1 ] || { echo "GATES FAILED"; exit 1; }

# ---- metrics
python3 - "$OUT" "$CLIPS" <<'PY'
import math, sys
out, clips = sys.argv[1], sys.argv[2].split()
sec = {}
for l in out.splitlines():
    p = l.split()
    if len(p) == 3 and p[0] == "CLIPSEC":
        sec[p[1]] = float(p[2])
asr = {}
for l in out.splitlines():
    p = l.split()
    if len(p) == 3 and p[0] == "ASR":
        asr[p[1]] = float(p[2])
tts = None
for l in out.splitlines():
    p = l.split()
    if len(p) == 2 and p[0] == "TTS":
        tts = float(p[1])
AOUT = 3.616  # T81 reply length
if tts is None or not asr:
    print("measure: no timings", file=sys.stderr); sys.exit(2)
crs, srs, trs, rrs = [], [], [], []
for spec in clips:
    lang = spec.split(":")[0]
    a_in = sec[lang]
    crs.append((asr[lang] + tts) / (a_in + AOUT))
    srs.append(asr[lang] / a_in)
    trs.append(tts / AOUT)
    rrs.append((asr[lang] + tts) / a_in)
mean = lambda v: sum(v) / len(v)
print("METRIC combined_rtf=%.4f" % mean(crs))
print("METRIC s2t_rtf=%.4f" % mean(srs))
print("METRIC tts_rtf=%.4f" % mean(trs))
print("METRIC reply_rtf=%.4f" % mean(rrs))
print("METRIC asr_wall_s=%.3f" % mean(list(asr.values())))
print("METRIC tts_wall_s=%.3f" % tts)
PY
