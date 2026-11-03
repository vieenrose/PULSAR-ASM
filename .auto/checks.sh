#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Accuracy gates. "Preserving accuracy" is the hard constraint, so this runs
# after every passing benchmark and a failure blocks `keep`.
#
# Gates are run with the reference (PyTorch) STFT fed, which is the established
# protocol for measuring the model + iSTFT on their own; thresholds are fixed
# and must never be relaxed:
#   complex spectrum rel-rms <= 1e-4, waveform SNR >= 50 dB, at 8 kHz and 16 kHz.
# Reference data is the committed tests/golden tree (read-only).
set -euo pipefail

ROOT=/tmp/pulsar-asm/pulsar_asm/nemotron_reuse
NVCC="${NVCC:-/usr/local/cuda-10.2/bin/nvcc}"
cd "$ROOT"
BIN=engine/reuse_nano_gpu
BLOB=tests/golden/reuse.blob
D=/tmp/reuse_gate
mkdir -p "$D"
export REUSE_GPU_ISTFT=1

for r in 8000 16000; do
  case "$r" in 8000) ws=1024;; 16000) ws=1280;; esac
  REUSE_WS_MB=$ws \
  REUSE_MAG="tests/golden/ref_out_$r/in_mag.raw" \
  REUSE_PHA="tests/golden/ref_out_$r/in_pha.raw" \
  REUSE_DUMP_AMP="$D/c_amp.raw" REUSE_DUMP_PHA="$D/c_pha.raw" \
  REUSE_DUMP_AUDIO="$D/c_audio.f32" \
    ./"$BIN" "$BLOB" "tests/golden/fixtures/mic_test2_$r.wav" /dev/null 1 >/dev/null
  python3 tests/gate_full.py --ref "tests/golden/ref_out_$r" --c "$D" --mode model \
    --label "$r Hz" | tail -2
done

# The Nano is the deployment target: sm_53 must keep compiling.
"$NVCC" -O3 -Xcompiler -fopenmp -arch=sm_53 -Iengine -c engine/reuse_gpu.cu \
  -o /tmp/reuse_sm53_check.o 2>/tmp/reuse_sm53.err || {
    echo "sm_53 COMPILE FAILED"; grep -E "error" /tmp/reuse_sm53.err | head -15; exit 1; }
echo "gates ok"
