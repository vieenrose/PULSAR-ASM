#!/bin/bash
# .auto/checks.sh -- quality gates for the combined conversation objective
# (LFM3 S2T + PrimeTTS v2.1 on the Nano). Runs after every passing benchmark; a run that
# fails here cannot be kept.
#
# Gates:
#   1. Fixture integrity: the frozen ASR goldens and the two conversation clips are the
#      bytes this session pinned (md5 sidecar). Changing them is a deliberate act.
#   2. ASR fidelity: the current lfm3 binary still reproduces the frozen transcript ids for
#      the zh and en conversation clips (the cold one-shot path; a warm service must be
#      bit-identical to it, which is what this gate means).
#   3. TTS fidelity: pt_v21_nano on the reply case is inside the parity envelope
#      (SNDR >= 60 dB), i.e. the audio is still the validated model, not a shortcut.
set -uo pipefail
NANO="${NANO:-picard@100.84.151.116}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ADIR="$REPO/.auto/combined"
LFM_DIR="${LFM_DIR:-/home/picard/lfm3}"
PT_DIR="${PT_DIR:-/home/picard/primetts}"

fail() { echo "CHECKS FAIL: $1"; exit 1; }

# ---- 1. fixture integrity
[ -s "$ADIR/fixtures.md5" ] || fail "missing $ADIR/fixtures.md5"
( cd "$ADIR" && md5sum -c fixtures.md5 --quiet ) || fail "fixtures changed (see $ADIR/fixtures.md5)"

# ---- 2. ASR fidelity on both conversation clips
scp -q "$ADIR/clip_zh.wav" "$ADIR/clip_en.wav" "$NANO:/tmp/" 2>/dev/null || fail "cannot sync the clips"
for lang in zh en; do
  ssh "$NANO" "cd $LFM_DIR && ./lfm3_asr . /tmp/clip_$lang.wav 96 --out-ids /tmp/chk.$lang.i32" >/dev/null 2>&1 \
    || fail "lfm3_asr failed on clip_$lang"
  scp -q "$NANO:/tmp/chk.$lang.i32" "/tmp/chk.$lang.i32" 2>/dev/null || fail "no ids for $lang"
  cmp -s "/tmp/chk.$lang.i32" "$ADIR/golden_$lang.i32" || fail "ASR transcript for $lang differs from the golden"
  echo "  ASR $lang: ids match golden"
done

# ---- 3. TTS fidelity
sndr=$(ssh "$NANO" "cd $PT_DIR && ./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/T81" 2>/dev/null \
       | grep -o "SNDR=[0-9.]*" | tail -1 | cut -d= -f2)
[ -n "${sndr:-}" ] || fail "TTS parity run produced no SNDR"
awk -v s="$sndr" 'BEGIN{exit !(s>=60)}' || fail "TTS SNDR ${sndr} dB < 60 dB"
echo "  TTS: SNDR ${sndr} dB"
echo "CHECKS OK"
