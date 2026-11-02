// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include <math.h>

void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            // mean
            float mean = 0.0f;
            int base = n*C*H*W + c*H*W;
            for (int i = 0; i < H*W; i++) mean += x[base + i];
            mean /= (H*W);
            // var
            float var = 0.0f;
            for (int i = 0; i < H*W; i++) {
                float d = x[base + i] - mean;
                var += d*d;
            }
            var /= (H*W);
            float invstd = 1.0f / sqrtf(var + eps);
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
