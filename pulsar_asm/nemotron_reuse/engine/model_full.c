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
void linear_fwd(const float *x, const float *w, const float *b,
                 float *y, int B, int In, int Out);

static const float *find_tensor(const reuse_blob_t *blob, const char *name, size_t *nelem) {
    const reuse_tensor_t *t = reuse_blob_find(blob, name);
    if (!t) return NULL;
    if (nelem) *nelem = t->nelem;
    return t->data;
}

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

// DenseBlock
static void dense_block_forward(const reuse_blob_t *blob, float *x, int N, int C_in, int T, int F, int block_idx, const char *prefix) {
    // simplified - will be expanded
    (void)blob; (void)x; (void)N; (void)C_in; (void)T; (void)F; (void)block_idx; (void)prefix;
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F, int T,
                   float *out_mag, float *out_pha) {
    // Full implementation in progress - matches PyTorch reference structure
    // DenseEncoder: conv1x1 2->64, DenseBlock x4, conv (1,3) stride (4,2)
    int hid = 64;
    // input [B,2,T,F] -> concat mag,pha
    float *x = (float *)malloc((size_t)B*2*T*F*sizeof(float));
    for (int n = 0; n < B; n++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                x[n*2*T*F + 0*T*F + t*F + f] = mag_in[n*F*T + f*T + t];
                x[n*2*T*F + 1*T*F + t*F + f] = pha_in[n*F*T + f*T + t];
            }
        }
    }
    // dense_conv_1
    float *x1 = (float *)malloc((size_t)B*hid*T*F*sizeof(float));
    const float *w1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.weight", NULL);
    const float *b1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.bias", NULL);
    const float *nw1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.weight", NULL);
    const float *nb1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.bias", NULL);
    const float *pa1 = find_tensor(blob, "dense_encoder.dense_conv_1.2.weight", NULL);
    conv2d_general(x, B, 2, T, F, w1, b1, hid, 1, 1, 0, 0, 1, 1, 1, 1, x1);
    // IN + PReLU
    float *x1n = (float *)malloc((size_t)B*hid*T*F*sizeof(float));
    instancenorm2d_prelu(x1, x1n, B, hid, T, F, nw1, nb1, pa1, 1e-5);
    free(x); free(x1);
    
    // DenseBlock depth 4 - simplified pass
    float *xb = (float *)malloc((size_t)B*hid*T*F*sizeof(float));
    memcpy(xb, x1n, (size_t)B*hid*T*F*sizeof(float));
    free(x1n);
    
    // dense_conv_2 stride (4,2)
    int T2 = (T - 1)/4 + 1;
    int F2 = (F - 3)/2 + 1;
    float *x2 = (float *)malloc((size_t)B*hid*T2*F2*sizeof(float));
    const float *w2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.weight", NULL);
    const float *b2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.bias", NULL);
    const float *nw2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.weight", NULL);
    const float *nb2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.bias", NULL);
    const float *pa2 = find_tensor(blob, "dense_encoder.dense_conv_2.2.weight", NULL);
    conv2d_general(xb, B, hid, T, F, w2, b2, hid, 1, 3, 0, 1, 1, 1, 4, 2, x2);
    free(xb);
    float *x2n = (float *)malloc((size_t)B*hid*T2*F2*sizeof(float));
    instancenorm2d_prelu(x2, x2n, B, hid, T2, F2, nw2, nb2, pa2, 1e-5);
    free(x2);
    
    // 30 TFMamba blocks - placeholder
    float *xm = (float *)malloc((size_t)B*hid*T2*F2*sizeof(float));
    memcpy(xm, x2n, (size_t)B*hid*T2*F2*sizeof(float));
    free(x2n);
    
    // Decoders
    float *yd = (float *)malloc((size_t)B*hid*T2*F2*sizeof(float));
    memcpy(yd, xm, (size_t)B*hid*T2*F2*sizeof(float));
    free(xm);
    
    // Mag decoder final
    float *mag_d = (float *)malloc((size_t)B*1*T*F*sizeof(float));
    memcpy(mag_d, mag_in, (size_t)B*F*T*sizeof(float));
    // simple upsample not exact
    memcpy(out_mag, mag_in, (size_t)B*F*T*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*F*T*sizeof(float));
    free(x1n); free(yd); free(mag_d);
}
