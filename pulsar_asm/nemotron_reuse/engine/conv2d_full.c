// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include <string.h>
#include <math.h>
#include <omp.h>

void conv2d_general(const float *x, int N, int C_in, int H, int W,
                    const float *w, const float *b, int C_out,
                    int kH, int kW, int padH, int padW, int dilH, int dilW,
                    int strideH, int strideW,
                    float *y) {
    int H_out = (H + 2*padH - dilH*(kH-1) - 1)/strideH + 1;
    int W_out = (W + 2*padW - dilW*(kW-1) - 1)/strideW + 1;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < C_out; co++) {
            for (int ho = 0; ho < H_out; ho++) {
                for (int wo = 0; wo < W_out; wo++) {
                    float acc = b ? b[co] : 0.0f;
                    for (int ci = 0; ci < C_in; ci++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int h_in = ho*strideH - padH + kh*dilH;
                                int w_in = wo*strideW - padW + kw*dilW;
                                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
                                acc += x[n*C_in*H*W + ci*H*W + h_in*W + w_in] * w[co*C_in*kH*kW + ci*kH*kW + kh*kW + kw];
                            }
                        }
                    }
                    y[n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo] = acc;
                }
            }
        }
    }
}
