#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Build the RE-USE Nano port.
#   ./build_nano.sh            -> CUDA build for Jetson Nano Gen1 (sm_53, CUDA 10.2 / JetPack R32.5.1)
#   ./build_nano.sh cpu        -> CPU reference build (fp32 + -DREUSE_DOUBLE_ACC accumulators)
#   ./build_nano.sh host 61    -> CUDA build for a dev host GPU (e.g. arch 61)
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/engine"

MODE="${1:-cuda}"
ARCH="${2:-53}"

COMMON="blob.c wav_io.c"
CPU_STFT="stft_dft.o istft_dft.o"

build_cpu_objs() {
  [ -f stft_dft.o ]  || gcc -O2 -fopenmp -x c -c stft_dft.cu  -o stft_dft.o  -I.
  [ -f istft_dft.o ] || gcc -O2 -fopenmp -x c -c istft_dft.cu -o istft_dft.o -I.
}

if [ "$MODE" = "cpu" ]; then
  echo "[cpu] building CPU reference (double accumulators)"
  build_cpu_objs
  gcc -O2 -fopenmp -DREUSE_DOUBLE_ACC -o reuse_nano_cpu \
      reuse_host.c $COMMON conv2d_full.c instancenorm.c tfmamba.c model_full.c \
      $CPU_STFT -I. -lm
  echo "-> ./reuse_nano_cpu <blob> <in.wav> <out.pcm> full"
  exit 0
fi

NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
  NVCC="$(command -v nvcc 2>/dev/null || true)"
  [ -x "$NVCC" ] || NVCC=/usr/local/cuda-10.2/bin/nvcc
fi
if [ ! -x "$NVCC" ]; then
  echo "nvcc not found (set NVCC=/path/to/nvcc); use './build_nano.sh cpu'" >&2
  exit 1
fi

echo "[cuda] building with $NVCC for sm_${ARCH}"
build_cpu_objs
# STFT/iSTFT stay on the CPU for now (DFT-GEMM, no cuFFT dependency); the model is on the GPU.
"$NVCC" -O3 -Xcompiler -fopenmp -arch=sm_${ARCH} -o reuse_nano_gpu \
     reuse_host_gpu.c reuse_gpu.cu $COMMON $CPU_STFT -lm
echo "-> ./reuse_nano_gpu <blob> <in.wav> <out.pcm> [reps]"
echo "   env: REUSE_MAG/REUSE_PHA to feed a reference STFT (model-only gate),"
echo "        REUSE_DUMP_AMP/PHA/AUDIO for gate dumps, REUSE_WS_MB for the arena."
