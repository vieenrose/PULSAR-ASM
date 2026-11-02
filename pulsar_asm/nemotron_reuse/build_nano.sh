#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/engine"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found"
  exit 1
fi
nvcc -O3 -arch=sm_53 -std=c++11 -lineinfo reuse.cu reuse_host.c blob.c wav_io.c -lm -o reuse_nano 2>&1 | tail -60
