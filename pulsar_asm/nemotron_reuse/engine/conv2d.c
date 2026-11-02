// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include <string.h>
#include <math.h>

void conv2d_1x1(const float *x, int N, int C_in, int H, int W,
                const float *w, const float *b,
                float *y, int C_out) {
    for (int n = 0; n < N; n++) {
        for (int ho = 0; ho < H; ho++) {
            for (int wo = 0; wo < W; wo++) {
                for (int co = 0; co < C_out; co++) {
                    float acc = b ? b[co] : 0.0f;
                    int idx_in = n*C_in*H*W + 0*H*W + ho*W + wo;
                    int idx_w = co*C_in*1*1 + 0*1*1;
                    for (int ci = 0; ci < C_in; ci++) {
                        acc += x[idx_in + ci*H*W] * w[idx_w + ci];
                    }
                    y[n*C_out*H*W + co*H*W + ho*W + wo] = acc;
                }
            }
        }
    }
}
