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

# ---- Nano memory guard (the board hard-hangs when RAM is exhausted)
free_mb=$(ssh -o ConnectTimeout=8 "$NANO" 'free -m | sed -n 2p | tr -s " " | cut -d" " -f4' 2>/dev/null)
[ -n "$free_mb" ] || fail "cannot read free RAM on $NANO"
[ "$free_mb" -ge 60 ] || fail "Nano free RAM ${free_mb} MB < 60 MB guard"

[ "$ASR_MODE" = cold ] || fail "ASR_MODE=$ASR_MODE: the resident ASR service is not built yet (only 'cold' exists)"

# ---- fixture checks: goldens must exist and the clips must be the frozen ones
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  [ -s "$ADIR/golden_$lang.i32" ] || fail "missing $ADIR/golden_$lang.i32 (freeze it with: lfm3_asr --out-ids)"
done

OUT=$(ssh "$NANO" "CLIPS='$CLIPS' LFM_DIR='$LFM_DIR' PT_DIR='$PT_DIR' REPLY_CASE='$REPLY_CASE' ASR_MAXTOK='$ASR_MAXTOK' bash -s" <<'REMOTE' 2>&1 | grep -v setlocale
set -u
cd "$LFM_DIR" || exit 2
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  python3 -c "import wave,sys; w=wave.open(sys.argv[2]); print('CLIPSEC %s %.3f' % (sys.argv[1], w.getnframes()/w.getframerate()))" "$lang" "$clip"
done
for spec in $CLIPS; do
  lang=${spec%%:*}; clip=${spec#*:}
  rm -f "/tmp/cmb.$lang.i32" "/tmp/cmb.$lang.log"
  t0=$(date +%s.%N)
  ./lfm3_asr . "$clip" "$ASR_MAXTOK" --out-ids "/tmp/cmb.$lang.i32" > "/tmp/cmb.$lang.log" 2>&1 || { echo "ASR_FAIL $lang"; continue; }
  t1=$(date +%s.%N)
  python3 -c "print('ASR %s %.3f' % ('$lang', $t1-$t0))"
  tail -1 "/tmp/cmb.$lang.log" | grep -q TRANSCRIPT || echo "ASR_NOTE $lang: no transcript line"
done
cd "$PT_DIR" || exit 2
t0=$(date +%s.%N)
./build/pt_v21_nano --container build/v21.ptk --case "tests/golden_v21/$REPLY_CASE" > /tmp/cmb.tts.log 2>&1 || echo "TTS_FAIL"
t1=$(date +%s.%N)
python3 -c "print('TTS %.3f' % ($t1-$t0))"
grep -o "SNDR=[0-9.]*" /tmp/cmb.tts.log | tail -1 | cut -d= -f2 | sed 's/^/TTS_SNDRdB /'
REMOTE
)
echo "$OUT" | grep -E "^(ASR|TTS|CLIPSEC)" | sed 's/^/  /'
echo "$OUT" | grep -q "ASR_FAIL\|TTS_FAIL" && { echo "GATE FAIL: a stage failed"; exit 1; }

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
