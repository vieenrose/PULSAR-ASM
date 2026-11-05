#!/usr/bin/env bash
# checks_lfm3_audio.sh -- S9 single-command gate for the LFM3-Audio-2B port.
#
# Runs every component gate that is currently certified, in dependency order:
#   S1 tokenizer, S2 mel, S2b wav, S3 encoder, S4 depth (if blob present),
#   S5 codec (if blob present), S6 controller mode machine, M1 ASR end to end.
# Exits non-zero on the first failure. Set NANO / NANO_DIR / LFM3_ENV as for
# the individual checks. With --help prints this and exits.
set -u
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '1,30p' "$0"; exit 0
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${LFM3_PY:-/home/louis/venvs/lfm3/bin/python}"
fail=0
run() { echo "===== $* ====="; "$@"; local rc=$?; if [ $rc -ne 0 ]; then echo "FAIL: $* (rc=$rc)"; fail=1; fi; return 0; }
run "$PY" "$HERE/checks_tok.py"
run "$PY" "$HERE/checks_mel.py"
run "$PY" "$HERE/checks_wav.py"
run "$PY" "$HERE/checks_asr.py"
echo "===== summary ====="
if [ $fail -ne 0 ]; then echo "LFM3 AUDIO GATE: FAIL"; exit 1; fi
echo "LFM3 AUDIO GATE: PASS"
