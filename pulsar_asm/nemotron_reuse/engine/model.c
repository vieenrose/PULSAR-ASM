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
void mamba_scan_fwd(const float *x, const float *dt, const float *A_log, const float *D,
                    const float *B, const float *C,
                    float *y, int Bsz, int L, int D_in, int N);

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

static const float *find_tensor(const reuse_blob_t *blob, const char *name, size_t *nelem) {
    const reuse_tensor_t *t = reuse_blob_find(blob, name);
    if (!t) return NULL;
    if (nelem) *nelem = t->nelem;
    return t->data;
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F, int T,
                   float *out_mag, float *out_pha) {
    (void)blob;
    // For now pass through - full implementation would be very large in C
    // Will implement incrementally
    memcpy(out_mag, mag_in, (size_t)B*F*T*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*F*T*sizeof(float));
}
