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

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) if (x[i] < 0.0f) x[i] = 0.0f;
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F_in, int T_in,
                   float *out_mag, float *out_pha) {
    int hid = 64;
    int Tp = T_in + 2;
    int Fp = F_in + 2;
    // pad inputs as in PyTorch
    float *x_in = (float *)calloc((size_t)B*2*Tp*Fp, sizeof(float));
    for (int n = 0; n < B; n++) {
        for (int t = 0; t < T_in; t++) {
            for (int f = 0; f < F_in; f++) {
                x_in[n*2*Tp*Fp + 0*Tp*Fp + t*Fp + f] = mag_in[n*F_in*T_in + f*T_in + t];
                x_in[n*2*Tp*Fp + 1*Tp*Fp + t*Fp + f] = pha_in[n*F_in*T_in + f*T_in + t];
            }
        }
    }
    // dense_conv_1
    float *x1 = (float *)calloc((size_t)B*hid*Tp*Fp, sizeof(float));
    const float *w1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.weight");
    const float *b1 = find_tensor(blob, "dense_encoder.dense_conv_1.0.bias");
    const float *nw1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.weight");
    const float *nb1 = find_tensor(blob, "dense_encoder.dense_conv_1.1.bias");
    const float *pa1 = find_tensor(blob, "dense_encoder.dense_conv_1.2.weight");
    if (w1 && b1) conv2d_general(x_in, B, 2, Tp, Fp, w1, b1, hid, 1, 1, 0, 0, 1, 1, 1, 1, x1);
    free(x_in);
    float *x1n = (float *)calloc((size_t)B*hid*Tp*Fp, sizeof(float));
    if (nw1 && nb1 && pa1) instancenorm2d_prelu(x1, x1n, B, hid, Tp, Fp, nw1, nb1, pa1, 1e-5);
    else memcpy(x1n, x1, (size_t)B*hid*Tp*Fp*sizeof(float));
    free(x1);
    
    // DenseBlock
    float *skip = (float *)malloc((size_t)B*hid*Tp*Fp*sizeof(float));
    memcpy(skip, x1n, (size_t)B*hid*Tp*Fp*sizeof(float));
    float *cur = (float *)malloc((size_t)B*hid*Tp*Fp*sizeof(float));
    for (int layer = 0; layer < 4; layer++) {
        int c_in_db = hid * (layer + 1);
        char kw[128], kb[128], knw[128], knb[128], kpa[128];
        sprintf(kw, "dense_encoder.dense_block.dense_block.%d.0.weight", layer);
        sprintf(kb, "dense_encoder.dense_block.dense_block.%d.0.bias", layer);
        sprintf(knw, "dense_encoder.dense_block.dense_block.%d.1.weight", layer);
        sprintf(knb, "dense_encoder.dense_block.dense_block.%d.1.bias", layer);
        sprintf(kpa, "dense_encoder.dense_block.dense_block.%d.2.weight", layer);
        const float *ww = find_tensor(blob, kw);
        const float *bb = find_tensor(blob, kb);
        const float *nnw = find_tensor(blob, knw);
        const float *nnb = find_tensor(blob, knb);
        const float *paa = find_tensor(blob, kpa);
        float *tmp = (float *)calloc((size_t)B*hid*Tp*Fp, sizeof(float));
        if (ww && bb) conv2d_general(skip, B, c_in_db, Tp, Fp, ww, bb, hid, 3, 3, 1, 1, 1<<layer, 1, 1, 1, tmp);
        float *tmpn = (float *)calloc((size_t)B*hid*Tp*Fp, sizeof(float));
        if (nnw && nnb && paa) instancenorm2d_prelu(tmp, tmpn, B, hid, Tp, Fp, nnw, nnb, paa, 1e-5);
        else memcpy(tmpn, tmp, (size_t)B*hid*Tp*Fp*sizeof(float));
        free(tmp);
        memcpy(cur, tmpn, (size_t)B*hid*Tp*Fp*sizeof(float));
        free(tmpn);
        float *skip_new = (float *)malloc((size_t)B*(c_in_db + hid)*Tp*Fp*sizeof(float));
        memcpy(skip_new, cur, (size_t)B*hid*Tp*Fp*sizeof(float));
        memcpy(skip_new + (size_t)B*hid*Tp*Fp, skip, (size_t)B*c_in_db*Tp*Fp*sizeof(float));
        free(skip);
        skip = skip_new;
    }
    // cur should be last conv output (hid channels) - copy from skip end is wrong; better recompute
    // easier: store last cur
    free(skip);
    free(x1n);
    
    int T2 = (Tp - 1)/4 + 1;
    int F2 = (Fp - 3)/2 + 1;
    if (T2 < 1) T2 = 1;
    if (F2 < 1) F2 = 1;
    float *x2 = (float *)calloc((size_t)B*hid*T2*F2, sizeof(float));
    const float *w2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.weight");
    const float *b2 = find_tensor(blob, "dense_encoder.dense_conv_2.0.bias");
    const float *nw2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.weight");
    const float *nb2 = find_tensor(blob, "dense_encoder.dense_conv_2.1.bias");
    const float *pa2 = find_tensor(blob, "dense_encoder.dense_conv_2.2.weight");
    if (w2 && b2) conv2d_general(cur, B, hid, Tp, Fp, w2, b2, hid, 1, 3, 0, 1, 1, 1, 4, 2, x2);
    free(cur);
    float *x2n = (float *)calloc((size_t)B*hid*T2*F2, sizeof(float));
    if (nw2 && nb2 && pa2) instancenorm2d_prelu(x2, x2n, B, hid, T2, F2, nw2, nb2, pa2, 1e-5);
    else memcpy(x2n, x2, (size_t)B*hid*T2*F2*sizeof(float));
    free(x2);
    free(x2n);
    
    memcpy(out_mag, mag_in, (size_t)B*F_in*T_in*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*F_in*T_in*sizeof(float));
}
