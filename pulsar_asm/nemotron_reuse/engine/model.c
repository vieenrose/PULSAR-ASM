// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void conv2d_1x1(const float *x, int N, int C_in, int H, int W,
                const float *w, const float *b,
                float *y, int C_out);
void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps);
void mamba_scan_fwd(const float *x, const float *dt, const float *A_log, const float *D,
                    const float *B, const float *C,
                    float *y, int Bsz, int L, int D_in, int N);

// Forward declarations
