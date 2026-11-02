// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#pragma once

// DFT-GEMM STFT/iSTFT: precompute DFT matrix on host, multiply by GEMM
// For sm_53 implement with shared memory tiled matmul or simple
