#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# RTF benchmark. RTF = processing wall time / input audio duration.
#
# Reports, per fixture: model-only RTF, STFT/iSTFT RTF, total RTF, and the
# first-call (cold) latency. Locks the Nano to its max clocks when possible.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BLOB="${BLOB:-$DIR/tests/golden/reuse.blob}"
# 3 GiB shared-memory GPU on Nano Gen1: 1 GiB+ arenas get over-committed and
# fault ("launch timed out"). 640 MiB covers the encoder peak (~560 MiB).
export REUSE_WS_MB="${REUSE_WS_MB:-640}"
# On Nano Gen1 the async path dies with a Tegra poisoned-page MMU fault
# (dmesg: "fake mmu fault ... type 15 (poisoned)") once the launch queue gets
# deep; serialising launches avoids it. Launch overhead is ~0.1 s against a
# ~225 s model, so this does not materially affect RTF.
if [ -e /etc/nv_tegra_release ]; then export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"; fi
BIN="${BIN:-$DIR/engine/reuse_nano_gpu}"
REPS="${REPS:-5}"
FIXTURES="${@:-$DIR/tests/golden/fixtures/mic_test2_8000.wav $DIR/tests/golden/fixtures/mic_test2_16000.wav}"

if [ ! -x "$BIN" ]; then echo "no binary at $BIN (run ./build_nano.sh)"; exit 1; fi

# Jetson: request max clocks; ignore failures on non-Nano hosts.
if command -v nvpmodel >/dev/null 2>&1; then nvpmodel -q 2>/dev/null | head -3 || true; fi
if command -v jetson_clocks >/dev/null 2>&1; then jetson_clocks 2>/dev/null || true; fi

for w in $FIXTURES; do
  [ -f "$w" ] || { echo "skip missing $w"; continue; }
  echo "=== $w ==="
  # cold: single rep includes first-call latency
  "$BIN" "$BLOB" "$w" /dev/null 1 2>&1 | sed 's/^/[cold] /'
  # warm: REPS reps, best/worst reported by the driver
  "$BIN" "$BLOB" "$w" /dev/null "$REPS" 2>&1 | sed 's/^/[warm] /'
done

echo
echo "# RTF definition: RTF_model = model wall time / audio duration."
echo "# RTF_total additionally counts the DFT-GEMM STFT/iSTFT (currently CPU)."
echo "# RTF < 1.0 means faster than real time."
