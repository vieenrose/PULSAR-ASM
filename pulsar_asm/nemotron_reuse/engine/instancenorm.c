// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include <math.h>
#include <omp.h>

#ifdef REUSE_DOUBLE_ACC
typedef double acc_t;
#else
typedef float acc_t;
#endif

void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            acc_t mean = 0.0;
            const int base = n*C*H*W + c*H*W;
            for (int i = 0; i < H*W; i++) mean += (acc_t)x[base + i];
            mean /= (acc_t)(H*W);
            acc_t var = 0.0;
            for (int i = 0; i < H*W; i++) {
                acc_t d = (acc_t)x[base + i] - mean;
                var += d*d;
            }
            var /= (acc_t)(H*W);
            const float invstd = (float)(1.0 / sqrt(var + (acc_t)eps));
            float ww = w ? w[c] : 1.0f;
            float bb = b ? b[c] : 0.0f;
            float aa = alpha ? alpha[c] : 0.25f;
            for (int i = 0; i < H*W; i++) {
                float val = (x[base + i] - mean) * invstd * ww + bb;
                if (val < 0.0f) val = aa * val;
                y[base + i] = val;
            }
        }
    }
}
