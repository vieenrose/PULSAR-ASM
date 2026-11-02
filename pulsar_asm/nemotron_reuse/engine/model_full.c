// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "model.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void conv2d_general(const float *x, int N, int C_in, int H, int W,
                    const float *w, const float *b, int C_out,
                    int kH, int kW, int padH, int padW, int dilH, int dilW,
                    int strideH, int strideW,
                    float *y);
void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps);

static const float *find_tensor(const reuse_blob_t *blob, const char *name) {
    const reuse_tensor_t *t = reuse_blob_find(blob, name);
    if (!t) return NULL;
    return t->data;
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F_in, int T_in,
                   float *out_mag, float *out_pha) {
    int hid = 64;
    float *x = (float *)malloc((size_t)B*2*T_in*F_in*sizeof(float));
    for (int n = 0; n < B; n++) {
        for (int t = 0; t < T_in; t++) {
            for (int f = 0; f < F_in; f++) {
                x[n*2*T_in*F_in + 0*T_in*F_in + t*F_in + f] = mag_in[n*F_in*T_in + f*T_in + t];
                x[n*2*T_in*F_in + 1*T_in*F_in + t*F_in + f] = pha_in[n*F_in*T_in + f*T_in + t];
            }
        }
    }
    float *x1 = (float *)calloc((size_t)B*hid*T_in*F_in, sizeof(float));
    const float *w1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.weight");
    const float *b1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.bias");
    const float *nw1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.weight");
    const float *nb1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.bias");
    const float *pa1 = find_tensor(blob, "dense_encoder.dense_conv_1.2.weight");
    if (w1 && b1) conv2d_general(x, B, 2, T_in, F_in, w1, b1, hid, 1, 1, 0, 0, 1, 1, 1, 1, x1);
    free(x);
    float *x1n = (float *)calloc((size_t)B*hid*T_in*F_in, sizeof(float));
    if (nw1 && nb1 && pa1) instancenorm2d_prelu(x1, x1n, B, hid, T_in, F_in, nw1, nb1, pa1, 1e-5);
    else memcpy(x1n, x1, (size_t)B*hid*T_in*F_in*sizeof(float));
    free(x1);
    
    int T2 = (T_in - 1)/4 + 1;
    int F2 = (F_in - 3)/2 + 1;
    if (T2 < 1) T2 = 1;
    if (F2 < 1) F2 = 1;
    float *x2 = (float *)calloc((size_t)B*hid*T2*F2, sizeof(float));
    const float *w2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.weight");
    const float *b2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.bias");
    const float *nw2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.weight");
    const float *nb2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.bias");
    const float *pa2 = find_tensor(blob, "dense_encoder.dense_conv_2.2.weight");
    if (w2 && b2) conv2d_general(x1n, B, hid, T_in, F_in, w2, b2, hid, 1, 3, 0, 1, 1, 1, 4, 2, x2);
    free(x1n);
    float *x2n = (float *)calloc((size_t)B*hid*T2*F2, sizeof(float));
    if (nw2 && nb2 && pa2) instancenorm2d_prelu(x2, x2n, B, hid, T2, F2, nw2, nb2, pa2, 1e-5);
    else memcpy(x2n, x2, (size_t)B*hid*T2*F2*sizeof(float));
    free(x2);
    
    memcpy(out_mag, mag_in, (size_t)B*F_in*T_in*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*F_in*T_in*sizeof(float));
    free(x2n);
}
