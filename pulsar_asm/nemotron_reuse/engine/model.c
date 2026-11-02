// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "model.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void conv2d_1x1(const float *x, int N, int C_in, int H, int W,
                const float *w, const float *b,
                float *y, int C_out);
void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps);

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F, int T,
                   float *out_mag, float *out_pha) {
    // Placeholder: copy input for now
    memcpy(out_mag, mag_in, (size_t)B*F*T*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*F*T*sizeof(float));
    (void)blob;
}
