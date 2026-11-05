#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# RTF benchmark for the RE-USE Nano port.
#   RTF = processing wall time / input audio duration.
# Primary metric = rtf_total for the 8 kHz benchmark fixture, measured on the
# FULL pipeline (engine STFT -> GPU model -> GPU iSTFT). Nothing is fed in from
# the reference, and every phase is reported separately so cost cannot be hides
# by shifting it into a phase that is not counted.
set -euo pipefail

ROOT=/tmp/pulsar-asm/pulsar_asm/nemotron_reuse
AUTO=/tmp/pulsar-asm/.auto
NVCC="${NVCC:-/usr/local/cuda-10.2/bin/nvcc}"
BIN=engine/reuse_nano_gpu
BLOB=tests/golden/reuse.blob
cd "$ROOT"

# --- pre-check: build (fast syntax/compile failure) ------------------------
if ! "$NVCC" -O3 -Xcompiler -fopenmp -arch=sm_61 -Iengine -o "$BIN" \
      engine/reuse_host_gpu.c engine/reuse_gpu.cu engine/blob.c engine/wav_io.c \
      engine/stft_dft.o engine/istft_dft.o -lm 2>"$AUTO/build.err"; then
  echo "BUILD FAILED"
  grep -E "error" "$AUTO/build.err" | head -20
  exit 1
fi

# Dev-host protocol. Fixed: GPU iSTFT on, full 1 GiB arena, no reference STFT.
export REUSE_GPU_ISTFT=1
export REUSE_WS_MB=1024
unset REUSE_MAG REUSE_PHA REUSE_KPROF GPU_DEBUG REUSE_NAIVE_OP || true



# --- benchmark fixture (the scored workload) ------------------------------
o=$(./"$BIN" "$BLOB" tests/golden/fixtures/mic_test2_8000.wav /dev/null 3 2>&1)
rtf_total=$(grep '^total(' <<<"$o" | awk '{for(i=1;i<=NF;i++) if($i~/^RTF_total/) print $(i+1)}')
rtf_model=$(grep '^model best' <<<"$o" | awk '{for(i=1;i<=NF;i++) if($i~/^RTF_model/) print $(i+1)}')
rtf_stft=$(grep '^stft(' <<<"$o" | awk '{for(i=1;i<=NF;i++) if($i=="RTF") {print $(i+1); exit}}')
rtf_istft=$(grep '^istft(' <<<"$o" | awk '{for(i=1;i<=NF;i++) if($i=="RTF") {print $(i+1); exit}}')
ph=$(grep '^phases:' <<<"$o")
enc=$(grep -oE 'encoder [0-9.]+' <<<"$ph" | awk '{print $2}')
tower=$(grep -oE '30xTFMamba [0-9.]+' <<<"$ph" | awk '{print $2}')
dec=$(grep -oE 'decoders [0-9.]+' <<<"$ph" | awk '{print $2}')
head=$(grep -oE 'heads [0-9.]+' <<<"$ph" | awk '{print $2}')

[ -n "$rtf_total" ] || { echo "NO RESULT from benchmark run"; tail -6 <<<"$o"; exit 1; }

printf 'phases: encoder %s s | tower %s s | decoders %s s | heads %s s\n' \
  "$enc" "$tower" "$dec" "$head"

# --- held-out shapes (NOT scored; anti-overfit monitors) ------------------
# Different clip, length and sample rate. Tuning must improve these too; if
# only rtf_total moves, the change is fixture-specific.
h1=$(./"$BIN" "$BLOB" "$AUTO/fixtures/held_8k.wav"  /dev/null 2 2>&1)
h2=$(./"$BIN" "$BLOB" "$AUTO/fixtures/held_12k.wav" /dev/null 2 2>&1)
# the 16 kHz clip needs a bigger arena (encoder peak scales with T*F)
h3=$(REUSE_WS_MB=1280 ./"$BIN" "$BLOB" "$AUTO/fixtures/held_16k.wav" /dev/null 2 2>&1)
rtf_h1=$(grep '^total(' <<<"$h1" | awk '{for(i=1;i<=NF;i++) if($i~/^RTF_total/) print $(i+1)}')
rtf_h2=$(grep '^total(' <<<"$h2" | awk '{for(i=1;i<=NF;i++) if($i~/^RTF_total/) print $(i+1)}')
rtf_h3=$(grep '^total(' <<<"$h3" | awk '{for(i=1;i<=NF;i++) if($i~/^RTF_total/) print $(i+1)}')

echo "METRIC rtf_total=$rtf_total"
echo "METRIC rtf_model=$rtf_model"
echo "METRIC rtf_stft=$rtf_stft"
echo "METRIC rtf_istft=$rtf_istft"
echo "METRIC enc_s=$enc"
echo "METRIC tower_s=$tower"
echo "METRIC dec_s=$dec"
echo "METRIC heads_s=$head"
echo "METRIC rtf_held8k=$rtf_h1"
echo "METRIC rtf_held12k=$rtf_h2"
echo "METRIC rtf_held16k=$rtf_h3"
