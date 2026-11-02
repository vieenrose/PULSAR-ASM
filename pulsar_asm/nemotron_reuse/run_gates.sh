#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Run the full gate suite against the committed golden references.
#
#   ./run_gates.sh gpu   # GPU model + CPU iSTFT, reference STFT fed (model gate)
#   ./run_gates.sh cpu   # CPU model + CPU iSTFT, engine STFT (end-to-end)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
GOLD="$DIR/tests/golden"
BLOB="$GOLD/reuse.blob"
MODE="${1:-gpu}"
OUT="${OUT:-/tmp/reuse_gates}"
mkdir -p "$OUT"

run_rate() {
  local r="$1"
  local ref="$GOLD/ref_out_$r"
  local wav="$GOLD/fixtures/mic_test2_$r.wav"
  local F T
  case "$r" in
    8000)  F=161 T=1021 ;;
    16000) F=321 T=1021 ;;
    44100) F=883 T=1023 ;;
    48000) F=961 T=1020 ;;
    *) echo "unknown rate $r"; return 1 ;;
  esac
  echo "############ $r Hz (F=$F T=$T) ############"
  if [ ! -f "$ref/in_mag.raw" ]; then
    echo "missing $ref/in_mag.raw - the model gate must be fed the REFERENCE (PyTorch) STFT." >&2
    echo "Regenerate with ref/ref_full.py; do NOT substitute engine STFT (tests/stft_dump.c)." >&2
    return 1
  fi
  if [ "$MODE" = "gpu" ]; then
    # model gate: feed the reference STFT so the model+iSTFT are measured alone
    REUSE_MAG="$ref/in_mag.raw" REUSE_PHA="$ref/in_pha.raw" \
    REUSE_DUMP_AMP="$OUT/c_amp.raw" REUSE_DUMP_PHA="$OUT/c_pha.raw" \
    REUSE_DUMP_AUDIO="$OUT/c_audio.f32" \
      "$DIR/engine/reuse_nano_gpu" "$BLOB" "$wav" /dev/null 3
  else
    REUSE_DUMP_INMAG="$OUT/c_inmag.raw" REUSE_DUMP_INPHA="$OUT/c_inpha.raw" \
    REUSE_DUMP_AMP="$OUT/c_amp.raw" REUSE_DUMP_PHA="$OUT/c_pha.raw" \
    REUSE_DUMP_AUDIO="$OUT/c_audio.f32" \
      "$DIR/engine/reuse_nano_cpu" "$BLOB" "$wav" /dev/null full
  fi
  python3 "$DIR/tests/gate_full.py" --ref "$ref" --c "$OUT" \
    --mode "$([ "$MODE" = "gpu" ] && echo model || echo e2e)" \
    --label "$MODE $r Hz"
  echo
}

for r in ${RATES:-8000 16000}; do run_rate "$r"; done
