// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static const float *find_tensor(const reuse_blob_t *blob, const char *name, size_t *nelem) {
    const reuse_tensor_t *t = reuse_blob_find(blob, name);
    if (!t) { if (nelem)*nelem=0; return NULL; }
    if (nelem)*nelem=t->nelem;
    return t->data;
}

void conv1d_causal(const float *x, const float *w, const float *b,
                   float *y, int B, int L, int C_in, int C_out, int k) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < L; t++) {
            for (int co = 0; co < C_out; co++) {
                float acc = b ? b[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kt = 0; kt < k; kt++) {
                        int tt = t - (k - 1) + kt;
                        if (tt < 0) continue;
                        acc += x[b*L*C_in + tt*C_in + ci] * w[co*C_in*k + ci*k + kt];
                    }
                }
                y[b*L*C_out + t*C_out + co] = acc;
            }
        }
    }
}
