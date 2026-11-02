// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

void conv1d_causal(const float *x, const float *w, const float *b,
                   float *y, int B, int L, int C_in, int C_out, int k) {
    for (int bn = 0; bn < B; bn++) {
        for (int t = 0; t < L; t++) {
            for (int co = 0; co < C_out; co++) {
                float acc = 0.0f;
                if (b) acc = b[co];
                for (int ci = 0; ci < C_in; ci++) {
                    for (int kt = 0; kt < k; kt++) {
                        int tt = t - (k - 1) + kt;
                        if (tt < 0) continue;
                        acc += x[bn*L*C_in + tt*C_in + ci] * w[co*C_in*k + ci*k + kt];
                    }
                }
                y[bn*L*C_out + t*C_out + co] = acc;
            }
        }
    }
}
